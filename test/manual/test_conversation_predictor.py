#!/usr/bin/env python3
"""Tests for the trace-driven conversation round predictor.

The predictor itself only depends on the standard library, so these tests
do not require torch.  We replicate the minimal sglang package-stub logic
from test_business_aware_eviction_replay.py to make ``import sglang...``
resolve to the local source tree without pulling in the full serving stack.

Run (from sglang repo root):
    python3 -m pytest test/manual/test_conversation_predictor.py -v
"""

from __future__ import annotations

import sys
import threading
import types
from pathlib import Path

# --------------------------------------------------------------------------- #
# Minimal local sglang import bootstrap (copied from replay harness — no torch).
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


_bootstrap_local_sglang_import()

from sglang.srt.mem_cache.conversation_predictor import (
    ConversationPrediction,
    ConversationRoundPredictor,
    DEFAULT_GLOBAL_ROUNDS,
    DEFAULT_TOKENS_PER_ROUND,
    MIN_SAMPLES,
)

import pytest


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


class TestUserLevelPrediction:
    """Sufficient user-level history should use the user median."""

    def test_user_level_prediction(self):
        predictor = ConversationRoundPredictor()
        # Record 5 conversations for user "u1" in tenant "t1"
        # Rounds: 4, 6, 8, 10, 12 -> median = 8
        for rounds in [4, 6, 8, 10, 12]:
            predictor.record_round("t1", "u1", rounds)

        # Current round = 3, expected total = 8, remaining = 5
        pred = predictor.predict("t1", "u1", current_round=3)
        assert pred.signal_source == "user_level"
        assert pred.predicted_remaining_rounds == 5  # 8 - 3
        assert 0.0 < pred.confidence <= 1.0

    def test_user_level_odd_sample_count(self):
        """Median of odd-count samples is the middle value."""
        predictor = ConversationRoundPredictor()
        # 3 samples: 3, 7, 5 -> sorted: 3, 5, 7 -> median = 5
        for rounds in [3, 7, 5]:
            predictor.record_round("t1", "u2", rounds)

        pred = predictor.predict("t1", "u2", current_round=2)
        assert pred.signal_source == "user_level"
        assert pred.predicted_remaining_rounds == 3  # 5 - 2


class TestTenantLevelFallback:
    """User history < MIN_SAMPLES should fall back to tenant median."""

    def test_tenant_level_fallback(self):
        predictor = ConversationRoundPredictor()
        # Give user "u1" only 1 sample (below MIN_SAMPLES=3)
        predictor.record_round("t1", "u1", 4)

        # Give other users in tenant "t1" enough samples
        # Rounds across tenant: 4, 6, 8, 10, 12 -> median = 8
        for rounds in [6, 8, 10, 12]:
            predictor.record_round("t1", "u_other", rounds)

        # User u1 has 1 sample < 3, so fall back to tenant level
        pred = predictor.predict("t1", "u1", current_round=3)
        assert pred.signal_source == "tenant_level"
        # Tenant median of [4, 6, 8, 10, 12] = 8, remaining = 8 - 3 = 5
        assert pred.predicted_remaining_rounds == 5
        assert 0.0 < pred.confidence <= 0.8


class TestGlobalDefaultFallback:
    """Both user and tenant history insufficient should use global default."""

    def test_global_default_fallback(self):
        predictor = ConversationRoundPredictor()
        # No history at all
        pred = predictor.predict("t1", "u1", current_round=2)
        assert pred.signal_source == "default"
        # DEFAULT_GLOBAL_ROUNDS (5) - 2 = 3
        assert pred.predicted_remaining_rounds == DEFAULT_GLOBAL_ROUNDS - 2
        assert pred.confidence == 0.0

    def test_insufficient_samples_still_uses_default(self):
        """Two samples (< MIN_SAMPLES=3) for both user and tenant."""
        predictor = ConversationRoundPredictor()
        predictor.record_round("t1", "u1", 10)
        predictor.record_round("t1", "u1", 20)
        # Only 2 user samples and 2 tenant samples, both < 3
        pred = predictor.predict("t1", "u1", current_round=1)
        assert pred.signal_source == "default"
        assert pred.predicted_remaining_rounds == DEFAULT_GLOBAL_ROUNDS - 1


class TestPredictToMetadataMapping:
    """predict_to_metadata should map prediction to BusinessMetadata fields."""

    def test_predict_to_metadata_mapping(self):
        predictor = ConversationRoundPredictor(tokens_per_round=100.0)
        # Record enough history for user-level prediction
        # Rounds: 3, 7, 5 -> sorted: 3, 5, 7 -> median = 5
        for rounds in [3, 7, 5]:
            predictor.record_round("t1", "u1", rounds)

        # current_round=2, remaining=3, confidence=min(1.0, 3/20)=0.15
        metadata = predictor.predict_to_metadata(
            "t1", "u1", current_round=2, reload_cost=42.0
        )

        assert metadata["estimated_reuse_prefix_len"] == 3 * 100.0  # 300.0
        assert metadata["business_complete"] is False
        assert metadata["estimated_reload_cost"] == 42.0
        assert metadata["time_window_score"] == 0.0
        assert metadata["tenant"] == "t1"
        # hot_bucket_score = confidence * remaining
        pred = predictor.predict("t1", "u1", current_round=2)
        assert metadata["hot_bucket_score"] == pytest.approx(
            pred.confidence * 3
        )

    def test_predict_to_metadata_business_complete(self):
        """When remaining rounds <= 0, business_complete should be True."""
        predictor = ConversationRoundPredictor()
        for rounds in [3, 5, 7]:
            predictor.record_round("t1", "u1", rounds)
        # median = 5, current_round = 5, remaining = 0
        metadata = predictor.predict_to_metadata(
            "t1", "u1", current_round=5, reload_cost=0.0
        )
        assert metadata["business_complete"] is True
        assert metadata["estimated_reuse_prefix_len"] == 0.0


class TestThreadSafety:
    """Concurrent record + predict should not crash or corrupt state."""

    def test_thread_safety(self):
        predictor = ConversationRoundPredictor(max_history=500)
        errors = []

        def writer():
            try:
                for i in range(200):
                    predictor.record_round("t_concurrent", f"u{i % 10}", i % 15 + 1)
            except Exception as exc:
                errors.append(exc)

        def reader():
            try:
                for i in range(200):
                    predictor.predict("t_concurrent", f"u{i % 10}", i % 5)
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=writer) for _ in range(4)
        ] + [
            threading.Thread(target=reader) for _ in range(4)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Concurrent access produced errors: {errors}"

        # After all writers finish, we should have predictable sample counts
        # 4 writer threads * 200 records / 10 users = 80 per user
        # But capped at max_history=500, so 80 < 500 -> all stored
        assert predictor.user_sample_count("t_concurrent", "u0") == 80
        # Tenant level: 4 * 200 = 800 records, capped at 500
        assert predictor.tenant_sample_count("t_concurrent") == 500


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
