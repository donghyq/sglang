from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Union


def _normalize_runtime_payload(
    payload: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not payload:
        return None

    normalized = {k: v for k, v in payload.items() if v is not None}
    chunks = normalized.get("chunks") or []
    if not chunks:
        return None

    runtime_chunks = []
    for chunk in chunks:
        chunk_id = chunk.get("id")
        content_hash = chunk.get("content_hash")
        text = chunk.get("text")
        if not chunk_id or not content_hash or not isinstance(text, str):
            return None
        runtime_chunks.append(
            {
                "id": chunk_id,
                "content_hash": content_hash,
                "text": text,
            }
        )

    if not normalized.get("order_sensitive", True):
        runtime_chunks = sorted(
            runtime_chunks,
            key=lambda chunk: json.dumps(
                chunk, ensure_ascii=True, separators=(",", ":"), sort_keys=True
            ),
        )

    normalized["chunks"] = runtime_chunks
    return normalized


def render_retrieval_runtime_prefix_text(
    payload: Optional[Dict[str, Any]],
) -> Optional[str]:
    """Render a stable retrieval prefix for exact runtime prefix reuse.

    This is the P1.5 prototype path: we do not compose external KV blobs.
    Instead, we deterministically render the retrieved chunks into a concrete
    text prefix so the existing RadixCache can reuse it via exact token-prefix
    matching.

    Fail-closed behavior:
    - no payload / empty chunks => None
    - any chunk missing id/content_hash/text => None
    - caller should fall back to the normal full prompt path when None
    """
    normalized = _normalize_runtime_payload(payload)
    if normalized is None:
        return None

    canonical = json.dumps(
        {"version": 1, **normalized},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    lines = [
        f"{digest}",
        "<<retrieval-prefix>>",
        f"namespace={normalized.get('namespace', '')}",
        f"template_rev={normalized.get('template_rev', '')}",
        f"render_rev={normalized.get('render_rev', '')}",
        f"schema_version={normalized.get('schema_version', '')}",
        f"order_sensitive={str(normalized.get('order_sensitive', True)).lower()}",
    ]

    for chunk in normalized["chunks"]:
        lines.extend(
            [
                f"<<chunk id={chunk['id']} content_hash={chunk['content_hash']}>>",
                chunk["text"],
                "<</chunk>>",
            ]
        )

    lines.append("<<query-suffix>>")
    return "\n".join(lines) + "\n"


def prepend_retrieval_runtime_prefix(
    prompt: Union[str, list[str], Any],
    payload: Optional[Dict[str, Any]],
) -> Union[str, list[str], Any]:
    """Prepend rendered retrieval prefix to text prompts only.

    For token-id prompts we intentionally do nothing. Re-tokenizing and then
    concatenating separately encoded prefixes could violate exact token truth at
    the boundary, so the runtime prototype only enables the text path.
    """
    prefix = render_retrieval_runtime_prefix_text(payload)
    if prefix is None:
        return prompt

    if isinstance(prompt, str):
        return prefix + prompt

    if isinstance(prompt, list) and (
        len(prompt) == 0 or all(isinstance(item, str) for item in prompt)
    ):
        return [prefix + item for item in prompt]

    return prompt


def prepend_retrieval_prefix_to_chat_messages(
    messages: List[Dict[str, Any]],
    payload: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Prepend retrieval runtime prefix to the last user turn before tokenization.

    This keeps chat token truth owned by the existing template/tokenizer path,
    instead of post-hoc re-encoding a rendered prompt string.

    Supported shapes:
    - last user message content is a string
    - last user message content is a list of content parts; we insert a text part
      at the beginning

    For all other shapes, fail closed and return the original messages.
    """
    prefix = render_retrieval_runtime_prefix_text(payload)
    if prefix is None or not messages:
        return messages

    copied = [dict(message) for message in messages]
    last_user_index = None
    for index in range(len(copied) - 1, -1, -1):
        if copied[index].get("role") == "user":
            last_user_index = index
            break

    if last_user_index is None:
        return messages

    target = dict(copied[last_user_index])
    content = target.get("content")
    if isinstance(content, str):
        target["content"] = prefix + content
    elif isinstance(content, list):
        target["content"] = [{"type": "text", "text": prefix}, *content]
    else:
        return messages

    copied[last_user_index] = target
    return copied
