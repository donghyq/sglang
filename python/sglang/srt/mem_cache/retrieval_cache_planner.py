from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, List, Optional, Tuple


class RetrievalCacheMissType(str, Enum):
    BLOB_MISSING = "blob_missing"
    MISSING_IDENTITY = "missing_identity"
    NAMESPACE_MISMATCH = "namespace_mismatch"
    CONTENT_HASH_MISMATCH = "content_hash_mismatch"
    MODEL_MISMATCH = "model_mismatch"
    TOKENIZER_MISMATCH = "tokenizer_mismatch"
    TEMPLATE_MISMATCH = "template_mismatch"
    RENDER_MISMATCH = "render_mismatch"
    FORMAT_MISMATCH = "format_mismatch"


@dataclass(frozen=True)
class RetrievalChunkRef:
    chunk_id: str
    content_hash: Optional[str] = None


@dataclass(frozen=True)
class RetrievalRenderKey:
    namespace: Optional[str] = None
    model_name: Optional[str] = None
    model_fingerprint: Optional[str] = None
    tokenizer_rev: Optional[str] = None
    tokenizer_fingerprint: Optional[str] = None
    template_rev: Optional[str] = None
    render_rev: Optional[str] = None
    special_token_config: Optional[str] = None
    schema_version: Optional[str] = None
    order_sensitive: bool = True


@dataclass(frozen=True)
class StoredRetrievalChunk:
    namespace: str
    chunk_id: str
    content_hash: str
    model_name: str
    model_fingerprint: str
    tokenizer_rev: str
    tokenizer_fingerprint: str
    template_rev: str
    render_rev: str
    special_token_config: str
    schema_version: str
    kv_format: str = "radix_prefix_v1"
    kv_blob_uri: Optional[str] = None
    token_count: int = 0
    size_bytes: int = 0


@dataclass(frozen=True)
class RetrievalChunkDecision:
    chunk: RetrievalChunkRef
    hit: bool
    kv_blob_uri: Optional[str] = None
    token_count: int = 0
    size_bytes: int = 0
    miss_type: Optional[RetrievalCacheMissType] = None


@dataclass(frozen=True)
class RetrievalCachePlan:
    decisions: Tuple[RetrievalChunkDecision, ...]
    reusable_token_count: int
    fallback_required: bool
    plan_latency_ms: float

    @property
    def hit_count(self) -> int:
        return sum(1 for decision in self.decisions if decision.hit)

    @property
    def miss_count(self) -> int:
        return len(self.decisions) - self.hit_count


class RetrievedChunkKVStore:
    """Exact-match store for retrieval-conditioned KV metadata.

    Store is keyed by the full identity boundary, not just `chunk_id`.
    This avoids accidental hits across namespace/model/template/tokenizer changes.
    """

    def __init__(self):
        self._entries: Dict[Tuple[str, str, str], StoredRetrievalChunk] = {}

    @staticmethod
    def _make_key(namespace: str, chunk_id: str, content_hash: str) -> Tuple[str, str, str]:
        return (namespace, chunk_id, content_hash)

    def put(self, entry: StoredRetrievalChunk) -> None:
        self._entries[self._make_key(entry.namespace, entry.chunk_id, entry.content_hash)] = entry

    def get(
        self,
        namespace: Optional[str],
        chunk_id: str,
        content_hash: Optional[str],
    ) -> Optional[StoredRetrievalChunk]:
        if namespace is None or content_hash is None:
            return None
        return self._entries.get(self._make_key(namespace, chunk_id, content_hash))


class RetrievalConditionedKVPlanner:
    """Plan exact retrieval KV reuse and record explicit fallback reasons.

    The planner is intentionally fail-closed:
    - missing request identity => miss
    - missing stored identity => miss
    - any incompatibility => miss
    - exceptions are expected to be caught by adapter/caller and trigger full prefill
    """

    def __init__(self, store: RetrievedChunkKVStore):
        self.store = store

    def plan(
        self,
        render_key: RetrievalRenderKey,
        retrieved_chunks: Iterable[RetrievalChunkRef],
    ) -> RetrievalCachePlan:
        start = time.perf_counter()
        decisions: List[RetrievalChunkDecision] = []
        reusable_token_count = 0
        fallback_required = False

        for chunk in retrieved_chunks:
            decision = self._plan_one(render_key, chunk)
            decisions.append(decision)
            if decision.hit:
                reusable_token_count += decision.token_count
            else:
                fallback_required = True

        return RetrievalCachePlan(
            decisions=tuple(decisions),
            reusable_token_count=reusable_token_count,
            fallback_required=fallback_required,
            plan_latency_ms=(time.perf_counter() - start) * 1000.0,
        )

    def _plan_one(
        self,
        render_key: RetrievalRenderKey,
        chunk: RetrievalChunkRef,
    ) -> RetrievalChunkDecision:
        if not render_key.namespace or not chunk.content_hash:
            return RetrievalChunkDecision(
                chunk=chunk,
                hit=False,
                miss_type=RetrievalCacheMissType.MISSING_IDENTITY,
            )

        stored = self.store.get(render_key.namespace, chunk.chunk_id, chunk.content_hash)
        if stored is None:
            return RetrievalChunkDecision(
                chunk=chunk,
                hit=False,
                miss_type=RetrievalCacheMissType.BLOB_MISSING,
            )

        if not stored.kv_blob_uri:
            return RetrievalChunkDecision(
                chunk=chunk,
                hit=False,
                miss_type=RetrievalCacheMissType.BLOB_MISSING,
            )

        # Stored side must also have all required identity fields.
        required_stored_fields = [
            stored.namespace,
            stored.chunk_id,
            stored.content_hash,
            stored.model_name,
            stored.model_fingerprint,
            stored.tokenizer_rev,
            stored.tokenizer_fingerprint,
            stored.template_rev,
            stored.render_rev,
            stored.special_token_config,
            stored.schema_version,
            stored.kv_format,
        ]
        if any(field in (None, "") for field in required_stored_fields):
            return RetrievalChunkDecision(
                chunk=chunk,
                hit=False,
                miss_type=RetrievalCacheMissType.MISSING_IDENTITY,
            )

        if stored.namespace != render_key.namespace:
            return RetrievalChunkDecision(
                chunk=chunk,
                hit=False,
                miss_type=RetrievalCacheMissType.NAMESPACE_MISMATCH,
            )

        if stored.content_hash != chunk.content_hash:
            return RetrievalChunkDecision(
                chunk=chunk,
                hit=False,
                miss_type=RetrievalCacheMissType.CONTENT_HASH_MISMATCH,
            )

        compatibility_checks = (
            (render_key.model_name, stored.model_name, RetrievalCacheMissType.MODEL_MISMATCH),
            (
                render_key.model_fingerprint,
                stored.model_fingerprint,
                RetrievalCacheMissType.MODEL_MISMATCH,
            ),
            (
                render_key.tokenizer_rev,
                stored.tokenizer_rev,
                RetrievalCacheMissType.TOKENIZER_MISMATCH,
            ),
            (
                render_key.tokenizer_fingerprint,
                stored.tokenizer_fingerprint,
                RetrievalCacheMissType.TOKENIZER_MISMATCH,
            ),
            (
                render_key.template_rev,
                stored.template_rev,
                RetrievalCacheMissType.TEMPLATE_MISMATCH,
            ),
            (
                render_key.render_rev,
                stored.render_rev,
                RetrievalCacheMissType.RENDER_MISMATCH,
            ),
            (
                render_key.special_token_config,
                stored.special_token_config,
                RetrievalCacheMissType.FORMAT_MISMATCH,
            ),
            (
                render_key.schema_version,
                stored.schema_version,
                RetrievalCacheMissType.FORMAT_MISMATCH,
            ),
        )
        for expected, actual, miss_type in compatibility_checks:
            if expected is None:
                return RetrievalChunkDecision(
                    chunk=chunk,
                    hit=False,
                    miss_type=RetrievalCacheMissType.MISSING_IDENTITY,
                )
            if actual != expected:
                return RetrievalChunkDecision(
                    chunk=chunk,
                    hit=False,
                    miss_type=miss_type,
                )

        return RetrievalChunkDecision(
            chunk=chunk,
            hit=True,
            kv_blob_uri=stored.kv_blob_uri,
            token_count=stored.token_count,
            size_bytes=stored.size_bytes,
        )
