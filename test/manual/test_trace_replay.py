#!/usr/bin/env python3
"""Trace-driven eviction replay harness.

Loads JSON trace data, converts each trace record to a ScenarioEvent,
trains a ConversationRoundPredictor (optional), and replays the trace
against LRU / SLRU / BusinessAware eviction policies on a simulated
RadixCache.  Reports hit rate, regret count, extra prefill cost, and
per-bucket / per-tenant breakdowns.

Trace format (JSON array, one object per request):

    {
      "user_id": "u123",
      "tenant": "tenant_a",
      "biz_type": "search",
      "sla_class": "standard",
      "round": 3,
      "total_rounds": 7,
      "token_ids": [1, 2, 3, 4, 5],
      "timestamp": 1700000000,
      "reload_cost": 12.0,
      "is_final_round": false
    }

Usage:
    uv run --python 3.12 --with torch,orjson python test/manual/test_trace_replay.py \
        --trace /path/to/trace.json
    uv run --python 3.12 --with torch,orjson python test/manual/test_trace_replay.py \
        --trace /path/to/trace.json --no-predictor
"""

from __future__ import annotations

import argparse
import enum
import hashlib
import json
import sys
import time
import types
import unittest.mock
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# --------------------------------------------------------------------------- #
# Minimal local sglang import bootstrap + runtime stubs.
# Copied from test_business_aware_eviction_replay.py so that this file can
# run independently.  See that file for detailed rationale.
# --------------------------------------------------------------------------- #


def _bootstrap_local_sglang_import() -> None:
    if "sglang" in sys.modules:
        return
    repo_root = Path(__file__).resolve().parents[2]
    package_root = repo_root / "python" / "sglang"
    if not package_root.exists():
        raise RuntimeError(f"Cannot locate local sglang package root: {package_root}")
    sglang_stub = types.ModuleType("sglang")
    sglang_stub.__path__ = [str(package_root)]
    sys.modules["sglang"] = sglang_stub


def _install_minimal_runtime_stubs() -> None:
    if "sglang.srt.mem_cache.allocator" not in sys.modules:
        allocator_mod = types.ModuleType("sglang.srt.mem_cache.allocator")

        class BaseTokenToKVPoolAllocator:
            pass

        allocator_mod.BaseTokenToKVPoolAllocator = BaseTokenToKVPoolAllocator
        sys.modules[allocator_mod.__name__] = allocator_mod

    if "sglang.srt.mem_cache.memory_pool" not in sys.modules:
        memory_pool_mod = types.ModuleType("sglang.srt.mem_cache.memory_pool")

        class ReqToTokenPool:
            pass

        memory_pool_mod.ReqToTokenPool = ReqToTokenPool
        sys.modules[memory_pool_mod.__name__] = memory_pool_mod

    if "sglang.srt.observability.metrics_collector" not in sys.modules:
        metrics_mod = types.ModuleType("sglang.srt.observability.metrics_collector")

        class RadixCacheMetricsCollector:
            def __init__(self, *args, **kwargs):
                pass

            def observe_eviction_duration(self, *args, **kwargs):
                pass

            def increment_eviction_num_tokens(self, *args, **kwargs):
                pass

        metrics_mod.RadixCacheMetricsCollector = RadixCacheMetricsCollector
        sys.modules[metrics_mod.__name__] = metrics_mod

    if "sglang.srt.disaggregation.kv_events" not in sys.modules:
        kv_events_mod = types.ModuleType("sglang.srt.disaggregation.kv_events")

        class StorageMedium(str, enum.Enum):
            GPU = "GPU"
            CPU = "CPU_PINNED"
            DISK = "DISK"
            EXTERNAL = "EXTERNAL"

        @dataclass
        class BlockStored:
            block_hashes: list[int]
            parent_block_hash: int | None
            token_ids: list[int]
            block_size: int
            lora_id: int | None
            medium: str | None = None

        @dataclass
        class BlockRemoved:
            block_hashes: list[int]
            medium: str | None = None

        @dataclass
        class AllBlocksCleared:
            pass

        kv_events_mod.StorageMedium = StorageMedium
        kv_events_mod.BlockStored = BlockStored
        kv_events_mod.BlockRemoved = BlockRemoved
        kv_events_mod.AllBlocksCleared = AllBlocksCleared
        sys.modules[kv_events_mod.__name__] = kv_events_mod

    if "sglang.srt.mem_cache.utils" not in sys.modules:
        utils_mod = types.ModuleType("sglang.srt.mem_cache.utils")

        def hash_str_to_int64(hash_str: str) -> int:
            digest = hashlib.sha256(hash_str.encode("utf-8")).digest()
            return int.from_bytes(digest[:8], byteorder="big", signed=False)

        utils_mod.hash_str_to_int64 = hash_str_to_int64
        sys.modules[utils_mod.__name__] = utils_mod


_bootstrap_local_sglang_import()
_install_minimal_runtime_stubs()

import torch  # noqa: E402

from sglang.srt.mem_cache.base_prefix_cache import (  # noqa: E402
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.conversation_predictor import (  # noqa: E402
    ConversationRoundPredictor,
)
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode  # noqa: E402


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #


@dataclass
class ScenarioEvent:
    """A single replayable event derived from a trace record."""

    key: str
    token_ids: List[int]
    hot_bucket_score: float
    time_window_score: float
    estimated_reload_cost: float
    estimated_reuse_prefix_len: float
    business_complete: bool = False
    biz_type: str = "default"
    sla_class: str = "standard"
    tenant: str = "default"

    # Extra metadata kept for regret / bucket analysis.
    user_id: str = ""
    reload_cost_raw: float = 0.0


@dataclass
class TraceReplayConfig:
    """Configuration for a single trace replay run."""

    trace_path: str
    cache_capacity: int = 16
    policies: List[str] = None  # type: ignore[assignment]
    use_predictor: bool = True

    def __post_init__(self):
        if self.policies is None:
            self.policies = ["lru", "slru", "business_aware"]


# --------------------------------------------------------------------------- #
# Trace loading and conversion
# --------------------------------------------------------------------------- #


def load_trace(path: str) -> List[dict]:
    """Load a JSON trace file.  Returns a list of trace-record dicts.

    Fail-closed: corrupt JSON -> empty list.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return []
        return data
    except Exception:
        return []


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    return bool(value) if value is not None else default


def trace_to_events(
    trace: List[dict],
    use_predictor: bool = True,
) -> List[ScenarioEvent]:
    """Convert trace records to ScenarioEvent sequence.

    If ``use_predictor`` is True:
        - First pass: train ConversationRoundPredictor on completed
          conversations (records where ``is_final_round`` is True, using
          ``total_rounds`` as the completed round count).
        - Second pass: for each record, call ``predict_to_metadata`` to
          derive BusinessMetadata fields.

    If ``use_predictor`` is False:
        - Use ``total_rounds - round`` directly as
          ``estimated_reuse_prefix_len``.
        - Use ``is_final_round`` as ``business_complete``.

    Fail-closed: corrupt trace records are silently skipped.
    """
    # Sort by timestamp (stable sort preserves insertion order for ties).
    indexed = [(idx, rec) for idx, rec in enumerate(trace)]
    indexed.sort(key=lambda pair: (_safe_int(pair[1].get("timestamp"), 0), pair[0]))

    # --- Training pass (predictor mode only) ---
    predictor: Optional[ConversationRoundPredictor] = None
    if use_predictor:
        predictor = ConversationRoundPredictor()
        for _, rec in indexed:
            if not _safe_bool(rec.get("is_final_round")):
                continue
            tenant = str(rec.get("tenant", "default"))
            user_id = str(rec.get("user_id", ""))
            total_rounds = _safe_int(rec.get("total_rounds"))
            if total_rounds > 0:
                predictor.record_round(tenant, user_id, total_rounds)

    # --- Conversion pass ---
    events: List[ScenarioEvent] = []
    for seq_no, (orig_idx, rec) in enumerate(indexed):
        token_ids = rec.get("token_ids")
        if not isinstance(token_ids, list) or len(token_ids) == 0:
            continue  # skip corrupt records

        tenant = str(rec.get("tenant", "default"))
        user_id = str(rec.get("user_id", ""))
        biz_type = str(rec.get("biz_type", "default"))
        sla_class = str(rec.get("sla_class", "standard"))
        reload_cost = _safe_float(rec.get("reload_cost"))
        round_num = _safe_int(rec.get("round"))
        total_rounds = _safe_int(rec.get("total_rounds"))
        is_final = _safe_bool(rec.get("is_final_round"))

        key = f"{tenant}:{user_id}:r{round_num}:{seq_no}"

        if use_predictor and predictor is not None:
            meta = predictor.predict_to_metadata(
                tenant=tenant,
                user_id=user_id,
                current_round=round_num,
                reload_cost=reload_cost,
            )
            events.append(
                ScenarioEvent(
                    key=key,
                    token_ids=list(token_ids),
                    hot_bucket_score=meta["hot_bucket_score"],
                    time_window_score=meta["time_window_score"],
                    estimated_reload_cost=meta["estimated_reload_cost"],
                    estimated_reuse_prefix_len=meta["estimated_reuse_prefix_len"],
                    business_complete=meta["business_complete"],
                    biz_type=biz_type,
                    sla_class=sla_class,
                    tenant=tenant,
                    user_id=user_id,
                    reload_cost_raw=reload_cost,
                )
            )
        else:
            remaining = max(0, total_rounds - round_num)
            events.append(
                ScenarioEvent(
                    key=key,
                    token_ids=list(token_ids),
                    hot_bucket_score=0.0,
                    time_window_score=0.0,
                    estimated_reload_cost=reload_cost,
                    estimated_reuse_prefix_len=float(remaining),
                    business_complete=is_final,
                    biz_type=biz_type,
                    sla_class=sla_class,
                    tenant=tenant,
                    user_id=user_id,
                    reload_cost_raw=reload_cost,
                )
            )

    return events


# --------------------------------------------------------------------------- #
# Replay engine
# --------------------------------------------------------------------------- #


class RecordingAllocator(unittest.mock.Mock):
    """Minimal mock allocator for simulated RadixCache."""

    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")
        self.freed_sequences: List[List[int]] = []

    def free(self, free_index: torch.Tensor):  # type: ignore[override]
        if free_index is None:
            self.freed_sequences.append([])
            return
        self.freed_sequences.append(free_index.detach().cpu().tolist())


def _insert_event(cache: RadixCache, event: ScenarioEvent):
    """Insert a token sequence into the cache and attach business metadata."""
    key = RadixKey(event.token_ids)
    value = torch.tensor(event.token_ids, dtype=torch.int64)
    cache.insert(InsertParams(key=key, value=value))
    time.sleep(0.001)  # ensure monotonic clock advances
    match_result = cache.match_prefix(MatchPrefixParams(key=key))
    node = match_result.last_device_node
    cache.set_business_metadata(
        node,
        hot_bucket_score=event.hot_bucket_score,
        time_window_score=event.time_window_score,
        estimated_reload_cost=event.estimated_reload_cost,
        estimated_reuse_prefix_len=event.estimated_reuse_prefix_len,
        business_complete=event.business_complete,
        biz_type=event.biz_type,
        sla_class=event.sla_class,
        tenant=event.tenant,
    )
    return node


def _match_full(cache: RadixCache, token_ids: List[int]) -> bool:
    result = cache.match_prefix(MatchPrefixParams(key=RadixKey(token_ids)))
    return len(result.device_indices) == len(token_ids)


def _collect_leaf_token_sets(cache: RadixCache) -> set:
    """Return the set of token-tuples for all live leaf nodes."""
    leaves = set()
    stack = [cache.root_node]
    while stack:
        node = stack.pop()
        if node is not cache.root_node and len(node.children) == 0 and not node.evicted:
            leaves.add(tuple(node.key.token_ids))
        for child in node.children.values():
            stack.append(child)
    return leaves


def _run_single_policy(
    policy: str,
    events: List[ScenarioEvent],
    cache_capacity: int,
) -> dict:
    """Replay ``events`` against a single eviction policy.

    Periodically triggers eviction when the evictable size exceeds
    ``cache_capacity``.  Tracks hit/miss, regret, extra prefill cost,
    per-bucket and per-tenant breakdowns.
    """
    TreeNode.counter = 0
    allocator = RecordingAllocator()
    cache = RadixCache.create_simulated(
        mock_allocator=allocator, eviction_policy=policy
    )

    evict_batch = max(1, cache_capacity // 4)

    total = 0
    hit_count = 0
    miss_count = 0

    # Per-bucket and per-tenant tallies.
    bucket_stats: Dict[str, Dict[str, int]] = {}
    tenant_stats: Dict[str, Dict[str, int]] = {}

    # Eviction tracking.
    eviction_curve: List[Dict[str, Any]] = []  # hit-rate after each eviction
    regret_count = 0
    total_extra_prefill_cost = 0.0

    # Track evicted token sets for regret detection.
    # Each entry: (set_of_evicted_token_tuples, events_after_this_eviction)
    evicted_history: List[set] = []

    # For regret: after each eviction, record evicted token tuples.
    # Then for subsequent events, if a miss matches an evicted tuple, it's a regret.
    all_evicted_tuples: set = set()
    # We also need the reload_cost for each evicted token set — but we don't
    # have that directly.  Instead, we look up the event that had those tokens.
    evicted_tuple_to_reload: Dict[Tuple[int, ...], float] = {}

    for event in events:
        total += 1
        is_hit = _match_full(cache, event.token_ids)

        if is_hit:
            hit_count += 1
        else:
            miss_count += 1
            # Insert the event into the cache.
            _insert_event(cache, event)

            # Check if this miss is a regret (the tokens were previously evicted).
            token_key = tuple(event.token_ids)
            if token_key in all_evicted_tuples:
                regret_count += 1
                total_extra_prefill_cost += evicted_tuple_to_reload.get(
                    token_key, event.reload_cost_raw
                )

        # Bucket and tenant stats.
        b = bucket_stats.setdefault(
            event.biz_type, {"total": 0, "hit": 0, "miss": 0}
        )
        b["total"] += 1
        b["hit" if is_hit else "miss"] += 1

        t = tenant_stats.setdefault(
            event.tenant, {"total": 0, "hit": 0, "miss": 0}
        )
        t["total"] += 1
        t["hit" if is_hit else "miss"] += 1

        # Trigger eviction if cache exceeds capacity.
        if cache.evictable_size > cache_capacity:
            pre_leaves = _collect_leaf_token_sets(cache)
            cache.evict(EvictParams(num_tokens=evict_batch))
            post_leaves = _collect_leaf_token_sets(cache)
            evicted_tuples = pre_leaves - post_leaves
            all_evicted_tuples |= evicted_tuples
            # Map evicted tuples to reload costs from the event.
            for et in evicted_tuples:
                # Find the event that inserted these tokens.
                for ev in events:
                    if tuple(ev.token_ids) == et:
                        evicted_tuple_to_reload[et] = ev.reload_cost_raw
                        break

            # Record hit-rate snapshot after this eviction.
            cur_rate = hit_count / total if total > 0 else 0.0
            eviction_curve.append(
                {
                    "event_index": total - 1,
                    "evicted_tokens": len(evicted_tuples),
                    "cumulative_hit_rate": round(cur_rate, 6),
                }
            )

    overall_hit_rate = hit_count / total if total > 0 else 0.0

    return {
        "policy": policy,
        "cache_capacity": cache_capacity,
        "use_predictor": None,  # filled by caller
        "total_events": total,
        "hit_count": hit_count,
        "miss_count": miss_count,
        "hit_rate": round(overall_hit_rate, 6),
        "regret_count": regret_count,
        "total_extra_prefill_cost": round(total_extra_prefill_cost, 4),
        "eviction_curve": eviction_curve,
        "bucket_stats": bucket_stats,
        "tenant_stats": tenant_stats,
        "eviction_count": len(eviction_curve),
    }


def run_trace_replay(config: TraceReplayConfig) -> List[dict]:
    """Run the full trace replay for all configured policies.

    Returns a list of per-policy result dicts.
    """
    trace = load_trace(config.trace_path)
    if not trace:
        print(f"[WARN] No trace data loaded from {config.trace_path}")
        return []

    events = trace_to_events(trace, use_predictor=config.use_predictor)
    if not events:
        print("[WARN] No events derived from trace (all records corrupt?)")
        return []

    results = []
    for policy in config.policies:
        result = _run_single_policy(
            policy=policy,
            events=events,
            cache_capacity=config.cache_capacity,
        )
        result["use_predictor"] = config.use_predictor
        results.append(result)

    return results


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main():
    parser = argparse.ArgumentParser(
        description="Trace-driven eviction replay harness"
    )
    parser.add_argument(
        "--trace", "-t", required=True, help="Path to trace JSON file"
    )
    parser.add_argument(
        "--cache-capacity", "-c", type=int, default=16,
        help="RadixCache token capacity (default: 16)",
    )
    parser.add_argument(
        "--policies", default="lru,slru,business_aware",
        help="Comma-separated eviction policies (default: lru,slru,business_aware)",
    )
    parser.add_argument(
        "--no-predictor", action="store_true",
        help="Disable ConversationRoundPredictor for business_aware (comparison)",
    )
    args = parser.parse_args()

    config = TraceReplayConfig(
        trace_path=args.trace,
        cache_capacity=args.cache_capacity,
        policies=[p.strip() for p in args.policies.split(",")],
        use_predictor=not args.no_predictor,
    )

    results = run_trace_replay(config)

    # --- Compact comparison summary ---
    print("=" * 72)
    print(
        f"Trace: {args.trace}  "
        f"capacity={args.cache_capacity}  "
        f"predictor={'OFF' if args.no_predictor else 'ON'}"
    )
    print("=" * 72)

    for r in results:
        print(f"\n[{r['policy']}]")
        print(f"  total_events:       {r['total_events']}")
        print(f"  hit/miss:           {r['hit_count']}/{r['miss_count']}")
        print(f"  hit_rate:           {r['hit_rate']:.4f}")
        print(f"  regret_count:       {r['regret_count']}")
        print(f"  extra_prefill_cost: {r['total_extra_prefill_cost']:.2f}")
        print(f"  eviction_count:     {r['eviction_count']}")
        print(f"  bucket_stats:       {r['bucket_stats']}")
        print(f"  tenant_stats:       {r['tenant_stats']}")

    print("\n" + "=" * 72)
    print("Full JSON below:\n")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
