#!/usr/bin/env python3
"""Partial Rollout KV 复用验证脚本

核心验证：pause(preserve_kv) 后重新发送相同 prompt，TTFT 是否低于首次。
如果 TTFT 明显下降，说明 KV cache 被保留在 Host 层，resume 时命中而非重新 prefill。

用法（需同时启动两个 server，一个带 HiCache，一个不带）：
    # Server 1: 带 HiCache（KV-aware 验证）
    python -m sglang.launch_server \\
        --model-path Qwen/Qwen2.5-0.5B-Instruct \\
        --port 30000 \\
        --enable-hierarchical-cache \\
        --hicache-ratio 2.0 \\
        --hicache-write-policy write_through \\
        --context-length 4096 \\
        --mem-fraction-static 0.7

    # Server 2: 不带 HiCache（token-only 对照）
    python -m sglang.launch_server \\
        --model-path Qwen/Qwen2.5-0.5B-Instruct \\
        --port 30001 \\
        --context-length 4096 \\
        --mem-fraction-static 0.7

    python test/registered/rl/verify_kv_reuse.py

依赖：pip install requests
"""

import argparse
import json
import sys
import time

import requests


def generate_stream(server, prompt, max_tokens=300):
    """流式生成，返回 (首token时间, 总时间, token数)"""
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    start = time.perf_counter()
    resp = requests.post(f"{server}/v1/chat/completions", json=payload, stream=True)
    first_token_time = None
    token_count = 0
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
                    if "content" in delta and delta["content"]:
                        if first_token_time is None:
                            first_token_time = time.perf_counter() - start
                        token_count += 1
                except Exception:
                    pass
    total_time = time.perf_counter() - start
    return first_token_time, total_time, token_count


def test_prefix_cache_hit(server, label):
    """步骤 1-2: 相同 prompt 二次发送，验证前缀缓存命中"""
    print(f"=== {label}: 前缀缓存命中验证 ===")
    prompt = "请写一篇详细的文章介绍深度学习的发展历史，从感知机开始到现代大语言模型，至少1000字。"

    ttft1, total1, n1 = generate_stream(server, prompt, max_tokens=300)
    print(f"  首次  TTFT: {ttft1:.3f}s, 总时间: {total1:.3f}s, tokens: {n1}")

    ttft2, total2, n2 = generate_stream(server, prompt, max_tokens=300)
    print(f"  二次  TTFT: {ttft2:.3f}s, 总时间: {total2:.3f}s, tokens: {n2}")
    print(f"  TTFT 加速比: {ttft1/ttft2:.2f}x")
    return ttft1, ttft2


def test_preserve_kv_resume(server, label):
    """步骤 3: pause(preserve_kv) 后重新发送，验证 KV 复用"""
    print(f"\n=== {label}: preserve_kv 暂停后复用验证 ===")
    prompt = "请解释量子计算的基本原理"

    ttft1, _, _ = generate_stream(server, prompt, max_tokens=100)
    print(f"  首次 TTFT: {ttft1:.3f}s")

    print("  暂停 (preserve_kv)...")
    resp = requests.post(f"{server}/pause_generation", json={"mode": "preserve_kv"})
    if resp.status_code != 200:
        print(f"  ⚠️ 暂停失败 (HTTP {resp.status_code})，跳过")
        return None, None
    time.sleep(0.5)

    print("  恢复...")
    requests.post(f"{server}/continue_generation", json={})
    time.sleep(0.5)

    ttft2, _, _ = generate_stream(server, prompt, max_tokens=100)
    print(f"  恢复后 TTFT: {ttft2:.3f}s")
    ratio = ttft2 / ttft1 if ttft1 > 0 else 0
    print(f"  TTFT 比值: {ratio:.2f}x ({'命中 ✅' if ratio < 0.8 else '未命中 ❌'})")
    return ttft1, ttft2


def test_token_only_resume(server, label):
    """步骤 4: 不带 HiCache 的 server 做 token-only 对照"""
    print(f"\n=== {label}: token-only 暂停后对照 ===")
    prompt = "请解释量子计算的基本原理"

    ttft1, _, _ = generate_stream(server, prompt, max_tokens=100)
    print(f"  首次 TTFT: {ttft1:.3f}s")

    print("  暂停 (retract)...")
    requests.post(f"{server}/pause_generation", json={"mode": "retract"})
    time.sleep(0.5)

    print("  恢复...")
    requests.post(f"{server}/continue_generation", json={})
    time.sleep(0.5)

    ttft2, _, _ = generate_stream(server, prompt, max_tokens=100)
    print(f"  恢复后 TTFT: {ttft2:.3f}s")
    ratio = ttft2 / ttft1 if ttft1 > 0 else 0
    print(f"  TTFT 比值: {ratio:.2f}x ({'命中 ✅' if ratio < 0.8 else '未命中 ❌'})")
    return ttft1, ttft2


def main():
    ap = argparse.ArgumentParser(description="Partial Rollout KV 复用验证")
    ap.add_argument("--hicache-url", default="http://localhost:30000", help="带 HiCache 的 server")
    ap.add_argument("--nocache-url", default="http://localhost:30001", help="不带 HiCache 的 server")
    ap.add_argument("--skip-nocache", action="store_true", help="跳过无 HiCache 对照")
    args = ap.parse_args()

    print("=" * 60)
    print("Partial Rollout KV Cache 复用验证")
    print("=" * 60)

    results = {}

    # 带 HiCache 的 server
    ttft1, ttft2 = test_prefix_cache_hit(args.hicache_url, "HiCache")
    results["hicache_prefix"] = (ttft1, ttft2)

    ttft1, ttft2 = test_preserve_kv_resume(args.hicache_url, "HiCache")
    if ttft1 and ttft2:
        results["hicache_preserve_kv"] = (ttft1, ttft2)

    # 不带 HiCache 的 server（对照）
    if not args.skip_nocache:
        try:
            ttft1, ttft2 = test_prefix_cache_hit(args.nocache_url, "NoCache")
            results["nocache_prefix"] = (ttft1, ttft2)

            ttft1, ttft2 = test_token_only_resume(args.nocache_url, "NoCache")
            if ttft1 and ttft2:
                results["nocache_token_only"] = (ttft1, ttft2)
        except requests.ConnectionError:
            print(f"\n⚠️ 无 HiCache server ({args.nocache_url}) 未启动，跳过对照")

    # 汇总
    print("\n" + "=" * 60)
    print("验证汇总")
    print("=" * 60)

    for name, (t1, t2) in results.items():
        ratio = t2 / t1 if t1 > 0 else 0
        hit = "✅ 命中" if ratio < 0.8 else "❌ 未命中"
        print(f"  {name:30s}  首次={t1:.3f}s  恢复后={t2:.3f}s  比值={ratio:.2f}x  {hit}")

    print("\n关键判据：")
    print("  preserve_kv 恢复后 TTFT 应明显低于首次 → 说明 KV cache 被保留在 Host 层")
    print("  对照组（无 HiCache / retract 模式）TTFT 应接近首次 → 说明重新 prefill")


if __name__ == "__main__":
    main()
