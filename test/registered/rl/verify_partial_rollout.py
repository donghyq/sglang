#!/usr/bin/env python3
"""Partial Rollout 基础功能验证脚本

验证 pause/resume 机制可用，包括全局 pause(preserve_kv) 和 per-request pause_request。

用法（需先启动 SGLang server）：
    python -m sglang.launch_server \\
        --model-path Qwen/Qwen2.5-0.5B-Instruct \\
        --port 30000 \\
        --enable-hierarchical-cache \\
        --hicache-ratio 2.0 \\
        --context-length 4096 \\
        --mem-fraction-static 0.7

    python test/registered/rl/verify_partial_rollout.py --port 30000

依赖：pip install requests
"""

import argparse
import json
import sys
import threading
import time

import requests


def generate(server, prompt, max_tokens=100, rid=None, stream=False):
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": stream,
    }
    if rid:
        payload["rid"] = rid
    return payload


def test_global_pause_resume(base_url):
    """测试 1: 全局 pause_generation(preserve_kv) + continue_generation"""
    print("=== 测试 1: 全局 pause_generation(preserve_kv) ===")

    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json=generate(base_url, "请写一篇500字的文章介绍人工智能", max_tokens=500),
    )
    assert resp.status_code == 200, f"生成失败: {resp.status_code}"
    print("  生成完成")

    print("  暂停 (preserve_kv)...")
    resp = requests.post(f"{base_url}/pause_generation", json={"mode": "preserve_kv"})
    assert resp.status_code == 200, f"暂停失败: {resp.status_code}"
    print(f"  {resp.json()}")

    print("  恢复...")
    resp = requests.post(f"{base_url}/continue_generation", json={})
    assert resp.status_code == 200, f"恢复失败: {resp.status_code}"

    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json=generate(base_url, "1+1等于几？", max_tokens=10),
    )
    assert resp.status_code == 200
    answer = resp.json()["choices"][0]["message"]["content"]
    print(f"  恢复后正常: {answer}")
    print("  测试 1 通过\n")


def test_per_request_pause_resume(base_url):
    """测试 2: per-request pause_request / resume_request"""
    print("=== 测试 2: per-request pause_request ===")

    results = {"text": "", "done": False}

    def stream_generate():
        payload = generate(
            base_url, "请详细解释Transformer架构", max_tokens=200, rid="test-req-001", stream=True
        )
        resp = requests.post(f"{base_url}/v1/chat/completions", json=payload, stream=True)
        for line in resp.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0].get("delta", {})
                        if "content" in delta:
                            results["text"] += delta["content"]
                    except Exception:
                        pass
        results["done"] = True

    t = threading.Thread(target=stream_generate)
    t.start()
    time.sleep(1)

    print("  暂停请求 test-req-001...")
    resp = requests.post(
        f"{base_url}/pause_request", json={"rid": "test-req-001", "pause_all": False}
    )
    print(f"  {resp.json()}")
    time.sleep(1)

    print("  恢复请求...")
    resp = requests.post(f"{base_url}/resume_request", json={"resume_all": True})
    print(f"  {resp.json()}")

    t.join(timeout=30)
    print(f"  生成内容长度: {len(results['text'])} 字符")
    print("  测试 2 完成\n")


def test_pause_all_requests(base_url):
    """测试 3: pause_all 暂停所有请求"""
    print("=== 测试 3: pause_all ===")

    resp = requests.post(
        f"{base_url}/pause_request", json={"rid": "", "pause_all": True}
    )
    assert resp.status_code == 200, f"pause_all 失败: {resp.status_code}"
    print(f"  pause_all: {resp.json()}")

    resp = requests.post(f"{base_url}/resume_request", json={"resume_all": True})
    assert resp.status_code == 200
    print(f"  resume_all: {resp.json()}")

    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json=generate(base_url, "hello", max_tokens=5),
    )
    assert resp.status_code == 200
    print("  恢复后正常")
    print("  测试 3 通过\n")


def main():
    ap = argparse.ArgumentParser(description="Partial Rollout 基础功能验证")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=30000)
    args = ap.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    print(f"Server: {base_url}\n")

    try:
        test_global_pause_resume(base_url)
        test_per_request_pause_resume(base_url)
        test_pause_all_requests(base_url)
        print("=== 所有基础测试通过 ===")
    except Exception as e:
        print(f"=== 测试失败: {e} ===")
        sys.exit(1)


if __name__ == "__main__":
    main()
