"""Measured restore policy using the existing offline RestoreCostGate interface.

Calibration observations are request TTFT, not kernel time. Only the executed
path is observed. No 4B/business-profile coefficients are used by the runtime.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class CostModelConfig:
    restore_fixed_ms: float = 5.0
    restore_per_kb_ms: float = 0.0001
    recompute_fixed_ms: float = 20.0
    recompute_per_token_ms: float = 0.15
    risk_margin_ms: float = 50.0
    min_samples: int = 10
    p95_factor: float = 2.5


@dataclass(frozen=True)
class RestoreCostInput:
    missing_tokens: int
    kv_bytes: int
    queue_ms: float = 0.0
    install_ms: float = 0.0
    sync_ms: float = 0.0
    transfer_ms: float | None = None


@dataclass(frozen=True)
class CostGateDecision:
    should_restore: bool
    restore_estimate_ms: float
    recompute_estimate_ms: float
    risk_margin_ms: float
    reason: str


@dataclass
class CostObservation:
    restore_ms: float
    recompute_ms: float


class RestoreCostGate:
    """Offline-first restore-vs-recompute decision gate."""

    def __init__(
        self,
        config: CostModelConfig | None = None,
        window_size: int = 100,
    ) -> None:
        self.config = config or CostModelConfig()
        self.observations: deque[CostObservation] = deque(maxlen=window_size)

    def record_observation(self, restore_ms: float, recompute_ms: float) -> None:
        """Calibration pairs from separate requests, never a per-request counterfactual."""
        self.observations.append(CostObservation(restore_ms, recompute_ms))

    def _restore_estimate(self, data: RestoreCostInput) -> float:
        if data.transfer_ms is not None:
            transfer_ms = data.transfer_ms
        else:
            transfer_ms = data.kv_bytes / 1024 * self.config.restore_per_kb_ms
        return (
            self.config.restore_fixed_ms
            + transfer_ms
            + data.queue_ms
            + data.install_ms
            + data.sync_ms
        )

    def _recompute_estimate(self, data: RestoreCostInput) -> float:
        return self.config.recompute_fixed_ms + (
            data.missing_tokens * self.config.recompute_per_token_ms
        )

    def _p95_restore_ms(self) -> float | None:
        if len(self.observations) < self.config.min_samples:
            return None
        values = sorted(obs.restore_ms for obs in self.observations)
        index = min(len(values) - 1, math.ceil(0.95 * len(values)) - 1)
        return values[index]

    def decide(self, data: RestoreCostInput) -> CostGateDecision:
        restore_ms = self._restore_estimate(data)
        recompute_ms = self._recompute_estimate(data)
        p95_restore_ms = self._p95_restore_ms()
        if p95_restore_ms is not None:
            restore_ms = max(restore_ms, p95_restore_ms * self.config.p95_factor)

        should_restore = restore_ms + self.config.risk_margin_ms < recompute_ms
        if should_restore:
            reason = "restore_estimate_below_recompute_with_margin"
        else:
            reason = "recompute_is_cheaper_or_margin_exceeded"

        return CostGateDecision(
            should_restore=should_restore,
            restore_estimate_ms=restore_ms,
            recompute_estimate_ms=recompute_ms,
            risk_margin_ms=self.config.risk_margin_ms,
            reason=reason,
        )

    def metrics(self) -> dict[str, float]:
        if not self.observations:
            return {
                "restore_win_rate": 0.0,
                "restore_regret_rate": 0.0,
                "observation_count": 0,
            }
        wins = sum(1 for obs in self.observations if obs.restore_ms < obs.recompute_ms)
        regrets = len(self.observations) - wins
        return {
            "restore_win_rate": wins / len(self.observations),
            "restore_regret_rate": regrets / len(self.observations),
            "observation_count": len(self.observations),
        }


class RuntimeRestoreCostGate:
    """Reload measured windows; unknown keys preserve native restore behavior.

    A profile is calibrated separately from acceptance traffic. TTFT samples
    include overlap directly; phase percentiles are never added together.
    Atomic profile replacement permits off/shadow/enforce on one deployment.
    """

    def __init__(self, filename: str, model: str, tp: int, bytes_per_rank: int):
        from pathlib import Path

        self.filename = Path(filename)
        self.model = model
        self.tp = tp
        self.bytes_per_rank = bytes_per_rank
        self._stamp = None
        self._profile = None
        self._error = "missing_profile"
        self._signature = 0

    def _reload(self):
        import hashlib
        import json

        try:
            # Some network filesystems coarsen mtime; compare content, not
            # timestamps, so a quick mode/profile replacement cannot be missed.
            raw = self.filename.read_bytes()
            stamp = hashlib.sha256(raw).hexdigest()
            if stamp == self._stamp:
                return
            profile = json.loads(raw)
            assert profile["schema_version"] == 1
            assert profile["mode"] in ("off", "shadow", "enforce")
            assert profile["model"] == self.model
            assert profile["tp"] == self.tp
            assert profile["kv_bytes_per_token_per_rank"] == self.bytes_per_rank
            assert profile["cost_scope"] == "request_ttft_ms"
            assert isinstance(profile["buckets"], list)
            for row in profile["buckets"]:
                assert isinstance(row, dict)
                assert (
                    isinstance(row["prefix_tokens"], int) and row["prefix_tokens"] > 0
                )
                assert isinstance(row.get("restore_ttft_ms", []), list)
                assert isinstance(row.get("recompute_ttft_ms", []), list)
            assert 3 <= profile.get("min_samples", 3) <= 100
            assert profile.get("window_size", 100) >= profile.get("min_samples", 3)
            margin = profile.get("risk_margin_ms", 0.0)
            assert math.isfinite(margin) and margin >= 0
            self._profile = profile
            self._stamp = stamp
            self._signature = int.from_bytes(
                hashlib.sha256(raw).digest()[:8], "little"
            ) & ((1 << 62) - 1)
            self._error = None
        except (OSError, ValueError, KeyError, TypeError, AssertionError):
            self._stamp = None
            self._profile = None
            self._signature = 0
            self._error = "missing_invalid_or_incompatible_profile"

    def evaluate(self, tokens: int, source_tier: str, load_bucket: str) -> dict:
        self._reload()
        record = {
            "mode": (self._profile or {}).get("mode", "off"),
            "profile_signature": self._signature,
            "source_tier": source_tier,
            "reusable_prefix_tokens": tokens,
            "load_bucket": load_bucket,
            "kv_payload_bytes_per_rank": tokens * self.bytes_per_rank,
            "kv_payload_bytes_all_ranks": tokens * self.bytes_per_rank * self.tp,
            "valid": False,
            "should_restore": True,
            "predicted_restore_ms": None,
            "predicted_recompute_ms": None,
            "cost_scope": "request_ttft_ms",
            "fallback_reason": self._error or "missing_bucket_or_samples",
        }
        if self._profile is None:
            return record
        p = self._profile
        window = p.get("window_size", 100)
        minimum = p.get("min_samples", 3)
        for row in p["buckets"]:
            bucket = row.get("prefix_tokens", 0)
            if (
                row.get("source_tier") != source_tier
                or row.get("load_bucket") != load_bucket
                or abs(tokens - bucket) > max(16, bucket // 8)
            ):
                continue
            restore = row.get("restore_ttft_ms", [])[-window:]
            recompute = row.get("recompute_ttft_ms", [])[-window:]
            samples = restore + recompute
            if (
                len(restore) < minimum
                or len(recompute) < minimum
                or not all(
                    isinstance(v, (int, float)) and math.isfinite(v) and v > 0
                    for v in samples
                )
            ):
                continue
            restore = sorted(restore)
            recompute = sorted(recompute)
            restore_ms = restore[math.ceil(0.95 * len(restore)) - 1]
            # Conservative recompute estimate: median, not its slowest tail.
            import statistics

            recompute_ms = statistics.median(recompute)
            model = RestoreCostGate(
                CostModelConfig(
                    restore_fixed_ms=0,
                    restore_per_kb_ms=0,
                    recompute_fixed_ms=recompute_ms,
                    recompute_per_token_ms=0,
                    risk_margin_ms=p.get("risk_margin_ms", 0.0),
                )
            )
            decision = model.decide(
                RestoreCostInput(
                    tokens, record["kv_payload_bytes_all_ranks"], transfer_ms=restore_ms
                )
            )
            record.update(
                valid=True,
                prefix_bucket=bucket,
                should_restore=decision.should_restore,
                predicted_restore_ms=decision.restore_estimate_ms,
                predicted_recompute_ms=decision.recompute_estimate_ms,
                risk_margin_ms=decision.risk_margin_ms,
                restore_sample_count=len(restore),
                recompute_sample_count=len(recompute),
                reason=decision.reason,
                fallback_reason=None,
            )
            return record
        return record


def consensus_gate_action(vote) -> tuple[bool, str | None]:
    """MIN-reduced [valid_enforce, permit_restore, signature, -signature]."""
    if int(vote[2]) != -int(vote[3]):
        return True, "rank_profile_mismatch"
    if not int(vote[0]):
        return True, "off_shadow_or_missing_rank_samples"
    return bool(int(vote[1])), None
