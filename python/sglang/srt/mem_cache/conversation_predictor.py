"""Trace-driven conversation round predictor for Business-Aware eviction.

Inspired by Alibaba USENIX ATC'25: business profile-based KV cache eviction.
Uses historical conversation round counts to predict remaining rounds,
feeding predictions into BusinessMetadata for eviction scoring.

Statistical method (median), not ML — by design:
- No training step, no model artifacts
- Graceful degradation: user -> tenant -> global default
- Thread-safe, O(1) per prediction after warm-up
"""

from __future__ import annotations

import statistics
import threading
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Tuple


DEFAULT_GLOBAL_ROUNDS: int = 5
DEFAULT_TOKENS_PER_ROUND: float = 512.0
MIN_SAMPLES: int = 3


@dataclass(frozen=True)
class ConversationPrediction:
    """Result of a single conversation-round prediction.

    Attributes:
        predicted_remaining_rounds: estimated rounds left in the conversation.
        confidence: prediction confidence in [0.0, 1.0], derived from sample count.
        signal_source: which history level was used — "user_level",
                       "tenant_level", or "default".
    """

    predicted_remaining_rounds: int
    confidence: float
    signal_source: str


class ConversationRoundPredictor:
    """Statistical predictor for remaining conversation rounds.

    Maintains a bounded history of completed-conversation round counts per
    ``(tenant, user_id)`` pair and per ``tenant``.  At query time the median
    of the most specific available history is used as the expected total
    round count; the remaining rounds are ``expected_total - current_round``.

    Degradation ladder (fail-closed):
        1. user-level history (>= MIN_SAMPLES entries)
        2. tenant-level history (>= MIN_SAMPLES entries)
        3. global default (DEFAULT_GLOBAL_ROUNDS)
    """

    def __init__(
        self,
        max_history: int = 1000,
        default_global_rounds: int = DEFAULT_GLOBAL_ROUNDS,
        tokens_per_round: float = DEFAULT_TOKENS_PER_ROUND,
    ) -> None:
        self._max_history = max_history
        self._default_global_rounds = default_global_rounds
        self._tokens_per_round = tokens_per_round
        self._lock = threading.Lock()
        self._user_history: Dict[Tuple[str, str], Deque[int]] = {}
        self._tenant_history: Dict[str, Deque[int]] = {}

    # ------------------------------------------------------------------ #
    # Recording
    # ------------------------------------------------------------------ #

    def record_round(
        self, tenant: str, user_id: str, round_count: int
    ) -> None:
        """Record the total round count of a completed conversation."""
        if round_count < 0:
            round_count = 0
        with self._lock:
            self._append_user(tenant, user_id, round_count)
            self._append_tenant(tenant, round_count)

    def _append_user(
        self, tenant: str, user_id: str, round_count: int
    ) -> None:
        key = (tenant, user_id)
        dq = self._user_history.get(key)
        if dq is None:
            dq = deque(maxlen=self._max_history)
            self._user_history[key] = dq
        dq.append(round_count)

    def _append_tenant(self, tenant: str, round_count: int) -> None:
        dq = self._tenant_history.get(tenant)
        if dq is None:
            dq = deque(maxlen=self._max_history)
            self._tenant_history[tenant] = dq
        dq.append(round_count)

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #

    def predict(
        self, tenant: str, user_id: str, current_round: int
    ) -> ConversationPrediction:
        """Predict remaining rounds for an in-progress conversation."""
        with self._lock:
            user_samples = list(self._user_history.get((tenant, user_id), ()))
            tenant_samples = list(self._tenant_history.get(tenant, ()))

        if len(user_samples) >= MIN_SAMPLES:
            median_total = statistics.median(user_samples)
            confidence = min(1.0, len(user_samples) / 20.0)
            signal_source = "user_level"
        elif len(tenant_samples) >= MIN_SAMPLES:
            median_total = statistics.median(tenant_samples)
            confidence = min(0.8, len(tenant_samples) / 25.0)
            signal_source = "tenant_level"
        else:
            median_total = self._default_global_rounds
            confidence = 0.0
            signal_source = "default"

        remaining = int(median_total) - current_round
        if remaining < 0:
            remaining = 0

        return ConversationPrediction(
            predicted_remaining_rounds=remaining,
            confidence=round(confidence, 4),
            signal_source=signal_source,
        )

    def predict_to_metadata(
        self,
        tenant: str,
        user_id: str,
        current_round: int,
        reload_cost: float = 0.0,
    ) -> dict:
        """Predict and map the result into a BusinessMetadata context dict.

        The returned dict can be passed directly to
        ``cache.set_business_metadata_from_context(node, context)``.
        """
        prediction = self.predict(tenant, user_id, current_round)
        remaining = prediction.predicted_remaining_rounds

        # high confidence + many remaining rounds => high hot-bucket score
        hot_bucket_score = prediction.confidence * remaining

        return {
            "estimated_reuse_prefix_len": remaining * self._tokens_per_round,
            "business_complete": remaining <= 0,
            "hot_bucket_score": hot_bucket_score,
            "time_window_score": 0.0,
            "estimated_reload_cost": reload_cost,
            "tenant": tenant,
        }

    # ------------------------------------------------------------------ #
    # Introspection helpers (read-only, for testing)
    # ------------------------------------------------------------------ #

    def user_sample_count(self, tenant: str, user_id: str) -> int:
        with self._lock:
            return len(self._user_history.get((tenant, user_id), ()))

    def tenant_sample_count(self, tenant: str) -> int:
        with self._lock:
            return len(self._tenant_history.get(tenant, ()))


__all__ = [
    "ConversationPrediction",
    "ConversationRoundPredictor",
    "DEFAULT_GLOBAL_ROUNDS",
    "DEFAULT_TOKENS_PER_ROUND",
    "MIN_SAMPLES",
]
