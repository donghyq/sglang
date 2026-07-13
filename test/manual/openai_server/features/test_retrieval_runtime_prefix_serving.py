"""Manual serving validation for Retrieval P1.5 exact rendered-prefix reuse.

This script is intentionally environment-dependent. It is not part of the
lightweight CPU unit-test path because it requires a real SGLang serving
environment with OpenAI-compatible endpoints, cache reporting enabled, and a
model that can actually start.

Validated scenario:
1. same retrieval payload + different query suffix -> second request should
   report more cached prompt tokens than the first request
2. changed retrieval content/template -> cached prompt tokens should drop
   relative to the warmed exact-hit request

Usage examples:

  # Reuse an already running server
  python test/manual/openai_server/features/test_retrieval_runtime_prefix_serving.py \
    --base-url http://127.0.0.1:30000 \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --mode both

  # Launch a temporary local server
  python test/manual/openai_server/features/test_retrieval_runtime_prefix_serving.py \
    --launch-server \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --mode completion
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from typing import Any, Dict, Optional

import requests


def _headers(api_key: Optional[str]) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _post_json(
    base_url: str,
    path: str,
    payload: Dict[str, Any],
    api_key: Optional[str],
    timeout: int = 120,
) -> Dict[str, Any]:
    response = requests.post(
        f"{base_url}{path}",
        headers=_headers(api_key),
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def _flush_cache(base_url: str, api_key: Optional[str]) -> None:
    response = requests.post(
        f"{base_url}/flush_cache",
        headers=_headers(api_key),
        timeout=60,
    )
    response.raise_for_status()


def _extract_usage(response_json: Dict[str, Any]) -> Dict[str, int]:
    usage = response_json.get("usage") or {}
    prompt_details = usage.get("prompt_tokens_details") or {}
    cached_tokens = int(prompt_details.get("cached_tokens", 0))
    return {
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
        "cached_tokens": cached_tokens,
    }


def _retrieval_payload(
    *,
    namespace: str = "waimai-poi",
    template_rev: str = "tpl-v1",
    render_rev: str = "render-v1",
    schema_version: str = "schema-v1",
    content_hash: str = "hash-a",
    text: str = "门店营业时间：09:00-22:00",
) -> Dict[str, Any]:
    return {
        "namespace": namespace,
        "template_rev": template_rev,
        "render_rev": render_rev,
        "schema_version": schema_version,
        "order_sensitive": True,
        "chunks": [
            {
                "id": "poi:1001",
                "content_hash": content_hash,
                "text": text,
            }
        ],
    }


def _print_result(label: str, result: Dict[str, Any]) -> None:
    print(f"[{label}] {json.dumps(result, ensure_ascii=False, sort_keys=True)}")


def _assert_has_cache_report(label: str, result: Dict[str, Any]) -> None:
    if "cached_tokens" not in result:
        raise AssertionError(f"{label}: response missing cached_tokens")
    if result["prompt_tokens"] <= 0:
        raise AssertionError(f"{label}: prompt_tokens not reported correctly: {result}")


def run_completion_scenario(
    *,
    base_url: str,
    model: str,
    api_key: Optional[str],
) -> Dict[str, Dict[str, int]]:
    cache_salt = f"retrieval-p15-completion-{uuid.uuid4()}"
    payload_hit = _retrieval_payload()
    payload_miss = _retrieval_payload(
        content_hash="hash-a2",
        text="门店营业时间：10:00-23:00",
    )

    _flush_cache(base_url, api_key)

    first = _extract_usage(
        _post_json(
            base_url,
            "/v1/completions",
            {
                "model": model,
                "prompt": "A问题：这家店营业到几点？",
                "max_tokens": 1,
                "temperature": 0,
                "cache_salt": cache_salt,
                "retrieval_cache": payload_hit,
            },
            api_key,
        )
    )
    second = _extract_usage(
        _post_json(
            base_url,
            "/v1/completions",
            {
                "model": model,
                "prompt": "B问题：这家店支持夜宵吗？",
                "max_tokens": 1,
                "temperature": 0,
                "cache_salt": cache_salt,
                "retrieval_cache": payload_hit,
            },
            api_key,
        )
    )
    changed = _extract_usage(
        _post_json(
            base_url,
            "/v1/completions",
            {
                "model": model,
                "prompt": "A问题：这家店营业到几点？",
                "max_tokens": 1,
                "temperature": 0,
                "cache_salt": cache_salt,
                "retrieval_cache": payload_miss,
            },
            api_key,
        )
    )

    _assert_has_cache_report("completion:first", first)
    _assert_has_cache_report("completion:second", second)
    _assert_has_cache_report("completion:changed", changed)

    _print_result("completion:first", first)
    _print_result("completion:second", second)
    _print_result("completion:changed", changed)

    if second["cached_tokens"] <= first["cached_tokens"]:
        raise AssertionError(
            "completion scenario failed: exact rendered-prefix hit did not increase cached prompt tokens"
        )
    if changed["cached_tokens"] >= second["cached_tokens"]:
        raise AssertionError(
            "completion scenario failed: changed retrieval payload should reduce cached prompt tokens"
        )
    if changed["cached_tokens"] > first["cached_tokens"]:
        raise AssertionError(
            "completion scenario failed: changed retrieval payload should fail closed relative to the cold request"
        )

    return {"first": first, "second": second, "changed": changed}


def run_chat_scenario(
    *,
    base_url: str,
    model: str,
    api_key: Optional[str],
) -> Dict[str, Dict[str, int]]:
    cache_salt = f"retrieval-p15-chat-{uuid.uuid4()}"
    payload_hit = _retrieval_payload(namespace="waimai-chat", template_rev="tpl-chat-v1")
    payload_miss = _retrieval_payload(
        namespace="waimai-chat",
        template_rev="tpl-chat-v1",
        content_hash="hash-a2",
        text="门店营业时间：10:00-23:00",
    )

    _flush_cache(base_url, api_key)

    shared_messages = [
        {"role": "system", "content": "你是外卖门店助手。"},
    ]

    first = _extract_usage(
        _post_json(
            base_url,
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [
                    *shared_messages,
                    {"role": "user", "content": "A问题：这家店营业到几点？"},
                ],
                "max_tokens": 1,
                "temperature": 0,
                "cache_salt": cache_salt,
                "retrieval_cache": payload_hit,
            },
            api_key,
        )
    )
    second = _extract_usage(
        _post_json(
            base_url,
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [
                    *shared_messages,
                    {"role": "user", "content": "B问题：这家店支持夜宵吗？"},
                ],
                "max_tokens": 1,
                "temperature": 0,
                "cache_salt": cache_salt,
                "retrieval_cache": payload_hit,
            },
            api_key,
        )
    )
    changed = _extract_usage(
        _post_json(
            base_url,
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [
                    *shared_messages,
                    {"role": "user", "content": "A问题：这家店营业到几点？"},
                ],
                "max_tokens": 1,
                "temperature": 0,
                "cache_salt": cache_salt,
                "retrieval_cache": payload_miss,
            },
            api_key,
        )
    )

    _assert_has_cache_report("chat:first", first)
    _assert_has_cache_report("chat:second", second)
    _assert_has_cache_report("chat:changed", changed)

    _print_result("chat:first", first)
    _print_result("chat:second", second)
    _print_result("chat:changed", changed)

    if second["cached_tokens"] <= first["cached_tokens"]:
        raise AssertionError(
            "chat scenario failed: exact rendered-prefix hit did not increase cached prompt tokens"
        )
    if changed["cached_tokens"] >= second["cached_tokens"]:
        raise AssertionError(
            "chat scenario failed: changed retrieval payload should reduce cached prompt tokens"
        )

    return {"first": first, "second": second, "changed": changed}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:30000",
        help="SGLang server base URL without /v1 suffix.",
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Model name to send in OpenAI requests.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Optional bearer token for the target server.",
    )
    parser.add_argument(
        "--mode",
        choices=["completion", "chat", "both"],
        default="both",
        help="Which OpenAI path to validate.",
    )
    parser.add_argument(
        "--launch-server",
        action="store_true",
        help="Launch a temporary local SGLang server instead of reusing an existing one.",
    )
    parser.add_argument(
        "--startup-timeout",
        type=int,
        default=600,
        help="Server startup timeout in seconds when --launch-server is used.",
    )
    parser.add_argument(
        "--server-arg",
        action="append",
        default=[],
        help="Extra raw arguments forwarded to `sglang serve` when --launch-server is used.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    process = None

    try:
        if args.launch_server:
            from sglang.srt.utils import kill_process_tree
            from sglang.test.test_utils import popen_launch_server

            launch_args = ["--enable-cache-report", *args.server_arg]
            process = popen_launch_server(
                args.model,
                args.base_url,
                timeout=args.startup_timeout,
                api_key=args.api_key,
                other_args=launch_args,
            )
            # Give the server a brief settle window so the first request is not
            # racing with startup logs or lazy initialization.
            time.sleep(2)
        else:
            kill_process_tree = None

        results: Dict[str, Any] = {}
        if args.mode in ("completion", "both"):
            results["completion"] = run_completion_scenario(
                base_url=args.base_url,
                model=args.model,
                api_key=args.api_key,
            )
        if args.mode in ("chat", "both"):
            results["chat"] = run_chat_scenario(
                base_url=args.base_url,
                model=args.model,
                api_key=args.api_key,
            )

        print("[summary] validation passed")
        print(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    finally:
        if process is not None:
            from sglang.srt.utils import kill_process_tree

            kill_process_tree(process.pid)


if __name__ == "__main__":
    raise SystemExit(main())
