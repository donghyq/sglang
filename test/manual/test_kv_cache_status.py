#!/usr/bin/env python3
"""Tests for the KV cache status query interface.

These tests exercise the fail-closed behavior of KVCacheStatusReporter:
hit scenarios, miss scenarios, and exception/degraded-input handling.

The modules under test only depend on the standard library + the local
sglang mem_cache source, so no torch is required.

Run (from sglang repo root):
    python3 -m pytest test/manual/test_kv_cache_status.py -v
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

# --------------------------------------------------------------------------- #
# Minimal local sglang import bootstrap (no torch needed).
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

from sglang.srt.mem_cache.kv_cache_status import (
    KVCacheStatus,
    KVCacheStatusReporter,
)
from sglang.srt.mem_cache.retrieval_cache_planner import (
    RetrievalConditionedKVPlanner,
    RetrievedChunkKVStore,
    StoredRetrievalChunk,
)

import pytest


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_stored_chunk(
    chunk_id: str = "chunk_1",
    content_hash: str = "hash_abc",
    namespace: str = "ns_test",
    token_count: int = 128,
) -> StoredRetrievalChunk:
    """Create a fully-specified StoredRetrievalChunk for testing."""
    return StoredRetrievalChunk(
        namespace=namespace,
        chunk_id=chunk_id,
        content_hash=content_hash,
        model_name="test-model",
        model_fingerprint="fp_model_v1",
        tokenizer_rev="tok_v1",
        tokenizer_fingerprint="fp_tok_v1",
        template_rev="tmpl_v1",
        render_rev="render_v1",
        special_token_config="stc_v1",
        schema_version="schema_v1",
        kv_blob_uri="blob://test/chunk_1",
        token_count=token_count,
        size_bytes=4096,
    )


def _make_payload(
    chunk_id: str = "chunk_1",
    content_hash: str = "hash_abc",
    namespace: str = "ns_test",
) -> dict:
    """Create a retrieval payload that matches a stored chunk."""
    return {
        "namespace": namespace,
        "model_name": "test-model",
        "model_fingerprint": "fp_model_v1",
        "tokenizer_rev": "tok_v1",
        "tokenizer_fingerprint": "fp_tok_v1",
        "template_rev": "tmpl_v1",
        "render_rev": "render_v1",
        "special_token_config": "stc_v1",
        "schema_version": "schema_v1",
        "order_sensitive": True,
        "chunks": [
            {"id": chunk_id, "content_hash": content_hash},
        ],
    }


def _make_reporter_with_store(
    stored_chunks=None,
) -> KVCacheStatusReporter:
    """Create a reporter backed by a store with optional pre-stored chunks."""
    store = RetrievedChunkKVStore()
    if stored_chunks:
        for chunk in stored_chunks:
            store.put(chunk)
    planner = RetrievalConditionedKVPlanner(store=store)
    return KVCacheStatusReporter(planner=planner)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


class TestHitScenario:
    """A matching stored chunk should produce has_cache=True."""

    def test_hit_scenario(self):
        chunk = _make_stored_chunk(token_count=256)
        reporter = _make_reporter_with_store([chunk])

        payload = _make_payload()
        status = reporter.query(payload)

        assert status.has_cache is True
        assert status.hit_chunk_count == 1
        assert status.miss_chunk_count == 0
        assert status.reusable_token_count == 256
        assert status.fallback_required is False
        assert status.miss_breakdown == {}
        assert status.plan_latency_ms >= 0.0

    def test_hit_to_dict(self):
        """to_dict should produce a JSON-friendly dict."""
        chunk = _make_stored_chunk(token_count=100)
        reporter = _make_reporter_with_store([chunk])

        status = reporter.query(_make_payload())
        d = status.to_dict()

        assert d["has_cache"] is True
        assert d["hit_chunk_count"] == 1
        assert d["miss_chunk_count"] == 0
        assert d["reusable_token_count"] == 100
        assert d["fallback_required"] is False
        assert d["miss_breakdown"] == {}
        assert isinstance(d["plan_latency_ms"], float)

    def test_partial_hit(self):
        """One hit + one miss should report has_cache=True, fallback=True."""
        hit_chunk = _make_stored_chunk(
            chunk_id="chunk_hit", content_hash="hash_hit"
        )
        reporter = _make_reporter_with_store([hit_chunk])

        payload = {
            "namespace": "ns_test",
            "model_name": "test-model",
            "model_fingerprint": "fp_model_v1",
            "tokenizer_rev": "tok_v1",
            "tokenizer_fingerprint": "fp_tok_v1",
            "template_rev": "tmpl_v1",
            "render_rev": "render_v1",
            "special_token_config": "stc_v1",
            "schema_version": "schema_v1",
            "order_sensitive": True,
            "chunks": [
                {"id": "chunk_hit", "content_hash": "hash_hit"},
                {"id": "chunk_miss", "content_hash": "hash_miss"},
            ],
        }
        status = reporter.query(payload)

        assert status.has_cache is True
        assert status.hit_chunk_count == 1
        assert status.miss_chunk_count == 1
        assert status.fallback_required is True
        assert "blob_missing" in status.miss_breakdown
        assert status.miss_breakdown["blob_missing"] == 1


class TestMissScenario:
    """No matching stored chunk should produce has_cache=False."""

    def test_miss_scenario(self):
        """Store is empty, so query should report a miss."""
        reporter = _make_reporter_with_store()  # empty store

        payload = _make_payload()
        status = reporter.query(payload)

        assert status.has_cache is False
        assert status.hit_chunk_count == 0
        assert status.miss_chunk_count == 1
        assert status.reusable_token_count == 0
        assert status.fallback_required is True
        assert "blob_missing" in status.miss_breakdown

    def test_miss_content_hash_mismatch(self):
        """Different content_hash should produce a miss."""
        chunk = _make_stored_chunk(content_hash="hash_original")
        reporter = _make_reporter_with_store([chunk])

        payload = _make_payload(content_hash="hash_different")
        status = reporter.query(payload)

        assert status.has_cache is False
        assert status.hit_chunk_count == 0
        assert status.miss_chunk_count == 1
        assert status.fallback_required is True


class TestExceptionFailClosed:
    """Any exception or invalid input should produce has_cache=False."""

    def test_none_payload(self):
        """None payload should be fail-closed."""
        reporter = _make_reporter_with_store()
        status = reporter.query(None)

        assert status.has_cache is False
        assert status.fallback_required is True
        assert status.hit_chunk_count == 0

    def test_empty_payload(self):
        """Empty dict payload should be fail-closed."""
        reporter = _make_reporter_with_store()
        status = reporter.query({})

        assert status.has_cache is False
        assert status.fallback_required is True

    def test_chunk_missing_id(self):
        """A chunk without 'id' triggers ValueError inside the adapter,
        which is caught by plan_from_payload -> returns None -> fail-closed."""
        reporter = _make_reporter_with_store()
        payload = {
            "namespace": "ns_test",
            "model_name": "test-model",
            "model_fingerprint": "fp_model_v1",
            "tokenizer_rev": "tok_v1",
            "tokenizer_fingerprint": "fp_tok_v1",
            "template_rev": "tmpl_v1",
            "render_rev": "render_v1",
            "special_token_config": "stc_v1",
            "schema_version": "schema_v1",
            "chunks": [{"content_hash": "hash_but_no_id"}],
        }
        status = reporter.query(payload)

        assert status.has_cache is False
        assert status.fallback_required is True

    def test_no_chunks_in_payload(self):
        """Payload with empty chunks list should be fail-closed."""
        reporter = _make_reporter_with_store()
        payload = {
            "namespace": "ns_test",
            "model_name": "test-model",
            "chunks": [],
        }
        status = reporter.query(payload)

        assert status.has_cache is False
        assert status.fallback_required is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
