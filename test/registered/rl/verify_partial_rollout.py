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

    results = {"text": "", "done": False, "chunks": 0, "error": None}
    started = threading.Event()

    def stream_generate():
        try:
            payload = generate(
                base_url,
                "请从1开始逐项解释1000个机器学习概念，每项都给出详细定义和例子",
                max_tokens=2000,
                rid="test-req-001",
                stream=True,
            )
            resp = requests.post(
                f"{base_url}/v1/chat/completions", json=payload, stream=True, timeout=120
            )
            resp.raise_for_status()
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
                            if delta.get("content"):
                                results["text"] += delta["content"]
                                results["chunks"] += 1
                                if results["chunks"] >= 5:
                                    started.set()
                        except (KeyError, json.JSONDecodeError):
                            pass
        except Exception as exc:
            results["error"] = exc
            started.set()
        finally:
            results["done"] = True

    t = threading.Thread(target=stream_generate)
    t.start()
    assert started.wait(timeout=30), "流式请求未在 30 秒内开始生成"
    assert results["error"] is None, f"流式请求失败: {results['error']}"
    assert not results["done"], "请求在 pause 前已完成，未覆盖运行中请求"

    print("  暂停请求 test-req-001...")
    resp = requests.post(
        f"{base_url}/pause_request", json={"rid": "test-req-001", "pause_all": False}
    )
    assert resp.status_code == 200, f"pause_request 失败: {resp.status_code} {resp.text}"
    paused_chunks = results["chunks"]
    time.sleep(1)
    assert results["chunks"] == paused_chunks, (
        f"pause 后仍输出 token: {paused_chunks} -> {results['chunks']}"
    )
    assert not results["done"], "请求在 paused 状态意外结束"
    print(f"  pause 生效，1 秒内 chunks 保持 {paused_chunks}")

    print("  恢复请求...")
    resp = requests.post(f"{base_url}/resume_request", json={"resume_all": True})
    assert resp.status_code == 200, f"resume_request 失败: {resp.status_code} {resp.text}"

    t.join(timeout=120)
    assert not t.is_alive(), "resume 后请求未在 120 秒内完成"
    assert results["error"] is None, f"恢复后的流式请求失败: {results['error']}"
    assert results["done"]
    assert results["chunks"] > paused_chunks, "resume 后没有继续输出 token"
    print(f"  生成内容长度: {len(results['text'])} 字符")
    print(f"  chunks: pause={paused_chunks}, complete={results['chunks']}")
    print("  测试 2 通过\n")


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
