#!/usr/bin/env python3
"""轻量级 eviction replay harness。

目标：
- 不依赖完整 serving stack
- 在 simulated `RadixCache` 上比较不同 eviction policy
- 输出可直接粘贴到文档/实验记录中的 JSON 结果

当前对比策略：
- lru
- slru
- business_aware

使用方式（推荐在 sglang repo 下执行）:
    uv run --python 3.12 --with torch,orjson python test/manual/test_business_aware_eviction_replay.py
    uv run --python 3.12 --with torch,orjson python test/manual/test_business_aware_eviction_replay.py --scenario hotspot
"""

from __future__ import annotations

import argparse
import enum
import hashlib
import json
import time
import sys
import types
import unittest.mock
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List


def _bootstrap_local_sglang_import() -> None:
    """在不安装整个 sglang package 的情况下允许本地 import。

    直接 `import sglang...` 会先执行 `sglang/__init__.py`，那会拉起大量非本实验必需依赖。
    这里注入一个轻量 package stub，只让 Python 去 repo/python/sglang 目录解析子模块。
    """

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
    """注入运行 `mem_cache.radix_cache` 所需的最小依赖。

    目标不是模拟完整 SGLang 运行时，而是把 replay harness 所依赖的
    import 面收缩到：
    - torch
    - radix_cache / evict_policy / base_prefix_cache 本身
    """

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

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode


@dataclass
class ScenarioEvent:
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


@dataclass
class ScenarioSpec:
    warm_events: List[ScenarioEvent]
    replay_events: List[ScenarioEvent]
    post_evict_probes: List[ScenarioEvent]


SCENARIOS: Dict[str, ScenarioSpec] = {
    "hotspot": ScenarioSpec(
        # 前 4 个 event 先填满 cache
        warm_events=[
            ScenarioEvent("hot_a", [1, 2, 3, 4], 3.0, 1.0, 5.0, 16.0),
            ScenarioEvent("hot_b", [1, 2, 3, 5], 2.5, 1.0, 4.0, 16.0),
            ScenarioEvent("cold_a", [8, 9, 10, 11], 0.0, 0.0, 1.0, 4.0),
            ScenarioEvent("cold_b", [12, 13, 14, 15], 0.0, 0.0, 1.0, 4.0),
        ],
        # 再 replay 一轮访问 / 新插入，制造 eviction 前的优先级差异
        replay_events=[
            ScenarioEvent("hot_a", [1, 2, 3, 4], 3.0, 1.0, 5.0, 16.0),
            ScenarioEvent("hot_b", [1, 2, 3, 5], 2.5, 1.0, 4.0, 16.0),
            ScenarioEvent("cold_c", [16, 17, 18, 19], 0.0, 0.0, 1.0, 4.0),
        ],
        # eviction 之后，探测未来马上会不会再次需要热点 key
        post_evict_probes=[
            ScenarioEvent("hot_a_probe", [1, 2, 3, 4], 3.0, 1.0, 5.0, 16.0),
            ScenarioEvent("hot_b_probe", [1, 2, 3, 5], 2.5, 1.0, 4.0, 16.0),
            ScenarioEvent("cold_c_probe", [16, 17, 18, 19], 0.0, 0.0, 1.0, 4.0),
        ],
    ),
    "time_shift": ScenarioSpec(
        warm_events=[
            ScenarioEvent("lunch_a", [21, 22, 23, 24], 2.0, 3.0, 3.0, 12.0),
            ScenarioEvent("lunch_b", [21, 22, 23, 25], 2.0, 3.0, 3.0, 12.0),
            ScenarioEvent("night_a", [31, 32, 33, 34], 1.0, 0.0, 2.0, 8.0),
            ScenarioEvent("done_a", [41, 42, 43, 44], 1.0, 1.0, 2.0, 8.0, True),
        ],
        replay_events=[
            ScenarioEvent("lunch_a", [21, 22, 23, 24], 2.0, 3.0, 3.0, 12.0),
            ScenarioEvent("lunch_b", [21, 22, 23, 25], 2.0, 3.0, 3.0, 12.0),
            ScenarioEvent("night_b", [31, 32, 33, 35], 1.0, 0.0, 2.0, 8.0),
        ],
        post_evict_probes=[
            ScenarioEvent("lunch_a_probe", [21, 22, 23, 24], 2.0, 3.0, 3.0, 12.0),
            ScenarioEvent("lunch_b_probe", [21, 22, 23, 25], 2.0, 3.0, 3.0, 12.0),
            ScenarioEvent("night_b_probe", [31, 32, 33, 35], 1.0, 0.0, 2.0, 8.0),
        ],
    ),
    "recency_vs_value_conflict": ScenarioSpec(
        # valuable_old 很有业务价值，但后续不再被 touch，因此在 eviction 前不是最近访问。
        # recent_low_value 最近被访问过，但业务价值低，而且 business_complete=True。
        warm_events=[
            ScenarioEvent(
                "valuable_old",
                [101, 102, 103, 104],
                hot_bucket_score=20000.0,
                time_window_score=2000.0,
                estimated_reload_cost=20000.0,
                estimated_reuse_prefix_len=20000.0,
                business_complete=False,
                biz_type="premium_rag",
                sla_class="premium",
            ),
            ScenarioEvent(
                "recent_low_value",
                [201, 202, 203, 204],
                hot_bucket_score=0.0,
                time_window_score=0.0,
                estimated_reload_cost=1.0,
                estimated_reuse_prefix_len=4.0,
                business_complete=True,
                biz_type="best_effort_chat",
                sla_class="best_effort",
            ),
            ScenarioEvent(
                "cold_a",
                [301, 302, 303, 304],
                hot_bucket_score=0.0,
                time_window_score=0.0,
                estimated_reload_cost=1.0,
                estimated_reuse_prefix_len=4.0,
                business_complete=False,
                biz_type="default",
            ),
            ScenarioEvent(
                "cold_b",
                [401, 402, 403, 404],
                hot_bucket_score=0.0,
                time_window_score=0.0,
                estimated_reload_cost=1.0,
                estimated_reuse_prefix_len=4.0,
                business_complete=False,
                biz_type="default",
            ),
        ],
        replay_events=[
            ScenarioEvent(
                "recent_low_value",
                [201, 202, 203, 204],
                hot_bucket_score=0.0,
                time_window_score=0.0,
                estimated_reload_cost=1.0,
                estimated_reuse_prefix_len=4.0,
                business_complete=True,
                biz_type="best_effort_chat",
                sla_class="best_effort",
            ),
            ScenarioEvent(
                "pressure_insert",
                [501, 502, 503, 504],
                hot_bucket_score=0.0,
                time_window_score=0.0,
                estimated_reload_cost=1.0,
                estimated_reuse_prefix_len=4.0,
                business_complete=False,
                biz_type="default",
            ),
        ],
        post_evict_probes=[
            ScenarioEvent(
                "valuable_old_probe",
                [101, 102, 103, 104],
                hot_bucket_score=20000.0,
                time_window_score=2000.0,
                estimated_reload_cost=20000.0,
                estimated_reuse_prefix_len=20000.0,
                business_complete=False,
                biz_type="premium_rag",
                sla_class="premium",
            ),
            ScenarioEvent(
                "recent_low_value_probe",
                [201, 202, 203, 204],
                hot_bucket_score=0.0,
                time_window_score=0.0,
                estimated_reload_cost=1.0,
                estimated_reuse_prefix_len=4.0,
                business_complete=True,
                biz_type="best_effort_chat",
                sla_class="best_effort",
            ),
            ScenarioEvent(
                "pressure_insert_probe",
                [501, 502, 503, 504],
                hot_bucket_score=0.0,
                time_window_score=0.0,
                estimated_reload_cost=1.0,
                estimated_reuse_prefix_len=4.0,
                business_complete=False,
                biz_type="default",
            ),
        ],
    ),
    "hotset_shift": ScenarioSpec(
        warm_events=[
            ScenarioEvent("old_hot_a", [601, 602, 603, 604], 8.0, 0.0, 6.0, 16.0, biz_type="legacy_hot"),
            ScenarioEvent("old_hot_b", [611, 612, 613, 614], 8.0, 0.0, 6.0, 16.0, biz_type="legacy_hot"),
            ScenarioEvent("new_hot_a", [621, 622, 623, 624], 1.0, 5.0, 6.0, 16.0, biz_type="new_hot"),
            ScenarioEvent("cold_tail", [631, 632, 633, 634], 0.0, 0.0, 1.0, 4.0),
        ],
        replay_events=[
            ScenarioEvent("new_hot_a", [621, 622, 623, 624], 1.0, 5.0, 6.0, 16.0, biz_type="new_hot"),
            ScenarioEvent("new_hot_b", [641, 642, 643, 644], 1.0, 5.0, 6.0, 16.0, biz_type="new_hot"),
        ],
        post_evict_probes=[
            ScenarioEvent("new_hot_a_probe", [621, 622, 623, 624], 1.0, 5.0, 6.0, 16.0, biz_type="new_hot"),
            ScenarioEvent("new_hot_b_probe", [641, 642, 643, 644], 1.0, 5.0, 6.0, 16.0, biz_type="new_hot"),
            ScenarioEvent("old_hot_a_probe", [601, 602, 603, 604], 8.0, 0.0, 6.0, 16.0, biz_type="legacy_hot"),
        ],
    ),
    "short_burst": ScenarioSpec(
        warm_events=[
            ScenarioEvent("stable_valuable", [701, 702, 703, 704], 2.0, 0.0, 20.0, 24.0, biz_type="stable"),
            ScenarioEvent("burst_1", [711, 712, 713, 714], 0.0, 0.0, 1.0, 4.0, biz_type="burst"),
            ScenarioEvent("burst_2", [721, 722, 723, 724], 0.0, 0.0, 1.0, 4.0, biz_type="burst"),
            ScenarioEvent("burst_3", [731, 732, 733, 734], 0.0, 0.0, 1.0, 4.0, biz_type="burst"),
        ],
        replay_events=[
            ScenarioEvent("burst_3", [731, 732, 733, 734], 0.0, 0.0, 1.0, 4.0, biz_type="burst"),
            ScenarioEvent("burst_4", [741, 742, 743, 744], 0.0, 0.0, 1.0, 4.0, biz_type="burst"),
        ],
        post_evict_probes=[
            ScenarioEvent("stable_valuable_probe", [701, 702, 703, 704], 2.0, 0.0, 20.0, 24.0, biz_type="stable"),
            ScenarioEvent("burst_4_probe", [741, 742, 743, 744], 0.0, 0.0, 1.0, 4.0, biz_type="burst"),
            ScenarioEvent("burst_3_probe", [731, 732, 733, 734], 0.0, 0.0, 1.0, 4.0, biz_type="burst"),
        ],
    ),
    "long_vs_short_prefix": ScenarioSpec(
        warm_events=[
            ScenarioEvent("long_prefix", [801, 802, 803, 804, 805, 806, 807, 808], 1.0, 0.0, 30.0, 64.0, biz_type="long_prefix"),
            ScenarioEvent("short_recent", [811, 812, 813, 814], 0.0, 0.0, 1.0, 4.0, business_complete=True, biz_type="short_prefix"),
            ScenarioEvent("cold_x", [821, 822, 823, 824], 0.0, 0.0, 1.0, 4.0),
            ScenarioEvent("cold_y", [831, 832, 833, 834], 0.0, 0.0, 1.0, 4.0),
        ],
        replay_events=[
            ScenarioEvent("short_recent", [811, 812, 813, 814], 0.0, 0.0, 1.0, 4.0, business_complete=True, biz_type="short_prefix"),
            ScenarioEvent("pressure_insert", [841, 842, 843, 844], 0.0, 0.0, 1.0, 4.0),
        ],
        post_evict_probes=[
            ScenarioEvent("long_prefix_probe", [801, 802, 803, 804, 805, 806, 807, 808], 1.0, 0.0, 30.0, 64.0, biz_type="long_prefix"),
            ScenarioEvent("short_recent_probe", [811, 812, 813, 814], 0.0, 0.0, 1.0, 4.0, business_complete=True, biz_type="short_prefix"),
        ],
    ),
    "tenant_fairness": ScenarioSpec(
        warm_events=[
            ScenarioEvent("tenant_a_valuable", [901, 902, 903, 904], 3.0, 0.0, 12.0, 16.0, biz_type="tenant_a", tenant="tenant_a"),
            ScenarioEvent("tenant_b_valuable", [911, 912, 913, 914], 3.0, 0.0, 12.0, 16.0, biz_type="tenant_b", tenant="tenant_b"),
            ScenarioEvent("tenant_a_cold", [921, 922, 923, 924], 0.0, 0.0, 1.0, 4.0, biz_type="tenant_a", tenant="tenant_a"),
            ScenarioEvent("tenant_b_cold", [931, 932, 933, 934], 0.0, 0.0, 1.0, 4.0, biz_type="tenant_b", tenant="tenant_b"),
        ],
        replay_events=[
            ScenarioEvent("tenant_b_valuable", [911, 912, 913, 914], 3.0, 0.0, 12.0, 16.0, biz_type="tenant_b", tenant="tenant_b"),
            ScenarioEvent("pressure_insert", [941, 942, 943, 944], 0.0, 0.0, 1.0, 4.0, biz_type="shared", tenant="shared"),
        ],
        post_evict_probes=[
            ScenarioEvent("tenant_a_valuable_probe", [901, 902, 903, 904], 3.0, 0.0, 12.0, 16.0, biz_type="tenant_a", tenant="tenant_a"),
            ScenarioEvent("tenant_b_valuable_probe", [911, 912, 913, 914], 3.0, 0.0, 12.0, 16.0, biz_type="tenant_b", tenant="tenant_b"),
        ],
    ),
}



class RecordingAllocator(unittest.mock.Mock):
    """最小 mock allocator。

    RadixCache simulated mode 只要求：
    - `.device`
    - `.free(tensor)`

    这里额外记录被 free 的 token index，便于结果解释。
    """

    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")
        self.freed_sequences: List[List[int]] = []

    def free(self, free_index: torch.Tensor):  # type: ignore[override]
        if free_index is None:
            self.freed_sequences.append([])
            return
        self.freed_sequences.append(free_index.detach().cpu().tolist())


def insert_event(cache: RadixCache, event: ScenarioEvent):
    key = RadixKey(event.token_ids)
    value = torch.tensor(event.token_ids, dtype=torch.int64)
    cache.insert(InsertParams(key=key, value=value))

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


def touch_event(cache: RadixCache, event: ScenarioEvent) -> dict[str, Any]:
    key = RadixKey(event.token_ids)
    result = cache.match_prefix(MatchPrefixParams(key=key))
    matched = len(result.device_indices) == len(event.token_ids)
    explanation = None
    if matched:
        explanation = cache.get_business_metadata_explanation(result.last_device_node)
    return {
        "key": event.key,
        "matched": matched,
        "matched_len": len(result.device_indices),
        "explanation": explanation,
    }


def replay_accesses(cache: RadixCache, replay_events: List[ScenarioEvent]) -> List[dict[str, Any]]:
    replay_logs = []
    for event in replay_events:
        pre_match = cache.match_prefix(MatchPrefixParams(key=RadixKey(event.token_ids)))
        is_full_match = len(pre_match.device_indices) == len(event.token_ids)
        if is_full_match:
            replay_logs.append({"type": "touch", **touch_event(cache, event)})
        else:
            inserted_node = insert_event(cache, event)
            replay_logs.append(
                {
                    "type": "insert",
                    "key": event.key,
                    "node_id": inserted_node.id,
                    "explanation": cache.get_business_metadata_explanation(inserted_node),
                }
            )
    return replay_logs


def collect_leaf_state(cache: RadixCache) -> List[dict[str, Any]]:
    leaves = []
    stack = [cache.root_node]
    while stack:
        node = stack.pop()
        if node is not cache.root_node and len(node.children) == 0 and not node.evicted:
            raw_metadata = cache.business_metadata_store.get_for_node(node.id)
            item = {
                "node_id": node.id,
                "tokens": node.key.token_ids,
                "value": node.value.detach().cpu().tolist() if node.value is not None else None,
                "lock_ref": node.lock_ref,
                "hit_count": node.hit_count,
                "last_access_time": node.last_access_time,
                "explanation": cache.get_business_metadata_explanation(node),
                "raw_metadata": {
                    "hot_bucket_score": raw_metadata.hot_bucket_score,
                    "time_window_score": raw_metadata.time_window_score,
                    "estimated_reload_cost": raw_metadata.estimated_reload_cost,
                    "estimated_reuse_prefix_len": raw_metadata.estimated_reuse_prefix_len,
                    "business_complete": raw_metadata.business_complete,
                    "biz_type": raw_metadata.biz_type,
                    "sla_class": raw_metadata.sla_class,
                    "priority": raw_metadata.priority,
                    "tenant": raw_metadata.tenant,
                } if raw_metadata is not None else None,
            }
            leaves.append(item)
        for child in node.children.values():
            stack.append(child)
    leaves.sort(key=lambda x: (x["tokens"], x["node_id"]))
    return leaves


def match_status(cache: RadixCache, events: List[ScenarioEvent]) -> List[dict[str, Any]]:
    statuses = []
    for event in events:
        result = cache.match_prefix(MatchPrefixParams(key=RadixKey(event.token_ids)))
        statuses.append(
            {
                "key": event.key,
                "matched": len(result.device_indices) == len(event.token_ids),
                "matched_len": len(result.device_indices),
            }
        )
    return statuses


def compute_probe_summary(cache: RadixCache, probe_events: List[ScenarioEvent]) -> dict[str, Any]:
    statuses = match_status(cache, probe_events)
    total = len(statuses)
    miss_count = sum(1 for item in statuses if not item["matched"])
    hit_count = total - miss_count
    return {
        "total": total,
        "hit_count": hit_count,
        "miss_count": miss_count,
        "miss_rate": (miss_count / total) if total else 0.0,
        "details": statuses,
    }


def _leaf_token_key(tokens: List[int]) -> tuple:
    return tuple(tokens)


def compute_evicted_set(
    pre_evict_leaves: List[dict[str, Any]],
    post_evict_leaves: List[dict[str, Any]],
) -> List[dict[str, Any]]:
    """Diff pre/post eviction leaf sets to identify evicted nodes."""
    surviving_keys = {_leaf_token_key(l["tokens"]) for l in post_evict_leaves}
    evicted = [
        leaf
        for leaf in pre_evict_leaves
        if _leaf_token_key(leaf["tokens"]) not in surviving_keys
    ]
    return evicted


def compute_regret_and_cost(
    evicted_leaves: List[dict[str, Any]],
    probe_events: List[ScenarioEvent],
    probe_statuses: List[dict[str, Any]],
) -> dict[str, Any]:
    """Compute eviction regret and extra prefill cost.

    regret: an evicted node whose token_ids match a post-eviction probe that
            missed.  This means we evicted something we immediately needed.
    extra_prefill_cost: sum of estimated_reload_cost for regret-evicted nodes,
            i.e. the recompute burden caused by premature eviction.
    """
    evicted_by_tokens = {}
    for leaf in evicted_leaves:
        evicted_by_tokens[_leaf_token_key(leaf["tokens"])] = leaf

    regrets: List[dict[str, Any]] = []
    total_extra_cost = 0.0

    for event, status in zip(probe_events, probe_statuses):
        if status["matched"]:
            continue
        token_key = tuple(event.token_ids)
        evicted_leaf = evicted_by_tokens.get(token_key)
        if evicted_leaf is None:
            continue
        raw_metadata = evicted_leaf.get("raw_metadata") or {}
        reload_cost = raw_metadata.get("estimated_reload_cost", 0.0)
        total_extra_cost += reload_cost
        regrets.append(
            {
                "probe_key": event.key,
                "biz_type": event.biz_type,
                "evicted_node_id": evicted_leaf["node_id"],
                "estimated_reload_cost": reload_cost,
            }
        )

    return {
        "regret_count": len(regrets),
        "regret_details": regrets,
        "total_extra_prefill_cost": total_extra_cost,
    }


def compute_bucket_hit_loss(
    probe_events: List[ScenarioEvent],
    probe_statuses: List[dict[str, Any]],
) -> dict[str, Any]:
    """Group probe hit/miss by biz_type to show per-bucket impact."""
    buckets: Dict[str, dict[str, Any]] = {}
    for event, status in zip(probe_events, probe_statuses):
        bucket = event.biz_type
        if bucket not in buckets:
            buckets[bucket] = {"total": 0, "hit": 0, "miss": 0}
        buckets[bucket]["total"] += 1
        if status["matched"]:
            buckets[bucket]["hit"] += 1
        else:
            buckets[bucket]["miss"] += 1
    return buckets



def compute_recomputed_tokens(
    evicted_leaves: List[dict[str, Any]],
    probe_events: List[ScenarioEvent],
    probe_statuses: List[dict[str, Any]],
) -> int:
    evicted_token_keys = {_leaf_token_key(leaf["tokens"]) for leaf in evicted_leaves}
    recomputed = 0
    for event, status in zip(probe_events, probe_statuses):
        if status["matched"]:
            continue
        if tuple(event.token_ids) in evicted_token_keys:
            recomputed += len(event.token_ids)
    return recomputed


def estimate_metadata_memory_overhead(leaves: List[dict[str, Any]]) -> int:
    total = 0
    for leaf in leaves:
        raw = leaf.get("raw_metadata")
        if raw is None:
            continue
        total += len(json.dumps(raw, sort_keys=True, ensure_ascii=True).encode("utf-8"))
    return total


def compute_tenant_hit_loss(
    probe_events: List[ScenarioEvent],
    probe_statuses: List[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    tenants: Dict[str, dict[str, int]] = {}
    for event, status in zip(probe_events, probe_statuses):
        tenant = event.tenant
        if tenant not in tenants:
            tenants[tenant] = {"total": 0, "hit": 0, "miss": 0}
        tenants[tenant]["total"] += 1
        tenants[tenant]["hit" if status["matched"] else "miss"] += 1
    return tenants

def run_policy(policy: str, scenario_name: str, evict_tokens: int = 4) -> dict[str, Any]:
    TreeNode.counter = 0
    allocator = RecordingAllocator()
    cache = RadixCache.create_simulated(mock_allocator=allocator, eviction_policy=policy)
    scenario = SCENARIOS[scenario_name]

    warm_nodes = []
    for event in scenario.warm_events:
        node = insert_event(cache, event)
        warm_nodes.append((event, node))

    before_replay = [
        {
            "key": event.key,
            "node_id": node.id,
            "explanation": cache.get_business_metadata_explanation(node),
        }
        for event, node in warm_nodes
    ]

    replay_logs = replay_accesses(cache, scenario.replay_events)
    pre_evict_leaves = collect_leaf_state(cache)

    eviction_start = time.perf_counter()
    evict_result = cache.evict(EvictParams(num_tokens=evict_tokens))
    eviction_latency_ms = (time.perf_counter() - eviction_start) * 1000.0
    post_evict_leaves = collect_leaf_state(cache)
    probe_summary = compute_probe_summary(cache, scenario.post_evict_probes)

    evicted_leaves = compute_evicted_set(pre_evict_leaves, post_evict_leaves)
    regret_metrics = compute_regret_and_cost(
        evicted_leaves,
        scenario.post_evict_probes,
        probe_summary["details"],
    )
    bucket_metrics = compute_bucket_hit_loss(
        scenario.post_evict_probes,
        probe_summary["details"],
    )
    tenant_metrics = compute_tenant_hit_loss(
        scenario.post_evict_probes,
        probe_summary["details"],
    )
    recomputed_tokens = compute_recomputed_tokens(
        evicted_leaves,
        scenario.post_evict_probes,
        probe_summary["details"],
    )
    metadata_memory_overhead_bytes = estimate_metadata_memory_overhead(pre_evict_leaves)

    return {
        "policy": policy,
        "scenario": scenario_name,
        "evict_tokens_target": evict_tokens,
        "before_replay": before_replay,
        "replay_logs": replay_logs,
        "pre_evict_leaves": pre_evict_leaves,
        "evicted_tokens": evict_result.num_tokens_evicted,
        "allocator_freed_sequences": allocator.freed_sequences,
        "post_evict_match_status": match_status(
            cache,
            scenario.warm_events + scenario.replay_events,
        ),
        "post_evict_probe_summary": probe_summary,
        "post_evict_leaves": post_evict_leaves,
        "evicted_leaves": evicted_leaves,
        "regret_metrics": regret_metrics,
        "bucket_hit_loss": bucket_metrics,
        "tenant_hit_loss": tenant_metrics,
        "recomputed_tokens": recomputed_tokens,
        "eviction_decision_latency_ms": eviction_latency_ms,
        "metadata_memory_overhead_bytes": metadata_memory_overhead_bytes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=sorted(SCENARIOS.keys()), default="hotspot")
    parser.add_argument("--evict-tokens", type=int, default=4)
    args = parser.parse_args()

    results = [
        run_policy(policy, args.scenario, evict_tokens=args.evict_tokens)
        for policy in ["lru", "slru", "business_aware"]
    ]

    # Compact comparison summary
    print("=" * 72)
    print(f"Scenario: {args.scenario}  (evict_tokens={args.evict_tokens})")
    print("=" * 72)
    for r in results:
        regret = r["regret_metrics"]
        print(f"\n[{r['policy']}]")
        print(f"  evicted_tokens:     {r['evicted_tokens']}")
        print(f"  freed_sequences:    {r['allocator_freed_sequences']}")
        print(f"  probe hit/miss:     {r['post_evict_probe_summary']['hit_count']}/{r['post_evict_probe_summary']['miss_count']}")
        print(f"  regret_count:       {regret['regret_count']}")
        print(f"  extra_prefill_cost: {regret['total_extra_prefill_cost']:.1f}")
        print(f"  recomputed_tokens:  {r['recomputed_tokens']}")
        print(f"  evict_latency_ms:   {r['eviction_decision_latency_ms']:.3f}")
        print(f"  metadata_overhead:  {r['metadata_memory_overhead_bytes']} bytes")
        print(f"  bucket_hit_loss:    {r['bucket_hit_loss']}")
        print(f"  tenant_hit_loss:    {r['tenant_hit_loss']}")
    print("\n" + "=" * 72)
    print("Full JSON below:\n")

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
