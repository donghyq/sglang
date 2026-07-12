from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Optional


def compute_retrieval_extra_key(payload: Optional[Dict[str, Any]]) -> Optional[str]:
    """Compute a stable retrieval-conditioned namespace key.

    This helper is intentionally lightweight and has no OpenAI/FastAPI/serving
    dependencies, so it can be unit-tested in a minimal CPU environment.
    """
    if not payload:
        return None

    normalized = {k: v for k, v in payload.items() if v is not None}
    chunks = normalized.get("chunks", [])
    if not normalized.get("order_sensitive", True):
        chunks = sorted(
            chunks,
            key=lambda chunk: json.dumps(
                chunk, ensure_ascii=True, separators=(",", ":"), sort_keys=True
            ),
        )
    normalized["chunks"] = chunks

    if normalized == {"chunks": [], "order_sensitive": True}:
        return None

    encoded = json.dumps(
        {"version": 1, **normalized},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:32]
    return f"retrieval:v1:{digest}"


def compose_prefix_cache_extra_key(
    cache_salt: Optional[str],
    retrieval_cache_payload: Optional[Dict[str, Any]],
    extra_key: Optional[str],
) -> Optional[str]:
    parts = []
    if cache_salt:
        parts.append(("cache_salt", cache_salt))

    retrieval_extra_key = compute_retrieval_extra_key(retrieval_cache_payload)
    if retrieval_extra_key is not None:
        parts.append(("retrieval_cache", retrieval_extra_key))

    if extra_key:
        parts.append(("extra_key", extra_key))

    if not parts:
        return None

    return "|".join(f"{key}={value}" for key, value in parts)
