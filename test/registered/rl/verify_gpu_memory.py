#!/usr/bin/env python3
"""Partial Rollout 显存验证脚本

验证 pause(preserve_kv) 后 GPU 显存是否释放，continue 后是否回升。

用法（需先启动 SGLang server）：
    python -m sglang.launch_server \\
        --model-path Qwen/Qwen2.5-0.5B-Instruct \\
        --port 30000 \\
        --enable-hierarchical-cache \\
        --hicache-ratio 2.0 \\
        --context-length 4096 \\
        --mem-fraction-static 0.7

    python test/registered/rl/verify_gpu_memory.py --port 30000

依赖：pip install requests
"""

import argparse
import subprocess
import sys
import time

import requests


def get_gpu_memory():
    """获取 GPU 显存使用量 (MB)"""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    )
    lines = result.stdout.strip().split("\n")
    if len(lines) == 1:
        return int(lines[0])
    # 多 GPU 取总和
    return sum(int(x) for x in lines)


def main():
    ap = argparse.ArgumentParser(description="Partial Rollout 显存验证")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=30000)
    args = ap.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    print(f"Server: {base_url}")
    print(f"GPU 显存监控: nvidia-smi\n")

    # 基线
    mem_idle = get_gpu_memory()
    print(f"基线显存 (server 启动后):  {mem_idle} MB")

    # 发送请求，占用 KV cache
    print("\n发送生成请求（占用 KV cache）...")
    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": "default",
            "messages": [{"role": "user", "content": "写一篇长文章" * 50}],
            "max_tokens": 500,
        },
    )
    if resp.status_code != 200:
        print(f"❌ 生成失败: {resp.status_code}")
        sys.exit(1)

    mem_after_gen = get_gpu_memory()
    print(f"生成后显存:               {mem_after_gen} MB (↑{mem_after_gen - mem_idle} MB)")

    # pause with preserve_kv
    print("\n暂停 (preserve_kv)...")
    resp = requests.post(f"{base_url}/pause_generation", json={"mode": "preserve_kv"})
    if resp.status_code != 200:
        print(f"❌ 暂停失败: {resp.status_code}")
        sys.exit(1)
    time.sleep(1)

    mem_after_pause = get_gpu_memory()
    released = mem_after_gen - mem_after_pause
    print(f"暂停后显存:               {mem_after_pause} MB (↓{released} MB)")
    print(f"  KV cache 释放: {'✅' if released > 50 else '⚠️ 释放不明显'}")

    # continue
    print("\n恢复...")
    requests.post(f"{base_url}/continue_generation", json={})
    time.sleep(1)

    mem_after_resume = get_gpu_memory()
    recovered = mem_after_resume - mem_after_pause
    print(f"恢复后显存:               {mem_after_resume} MB (↑{recovered} MB)")
    print(f"  KV cache 回升: {'✅' if recovered > 0 else '⚠️ 回升不明显'}")

    # 汇总
    print(f"\n{'=' * 50}")
    print("显存变化汇总:")
    print(f"  基线:        {mem_idle} MB")
    print(f"  生成后:      {mem_after_gen} MB (+{mem_after_gen - mem_idle} MB)")
    print(f"  pause 后:    {mem_after_pause} MB ({'-' if released >= 0 else '+'}{abs(released)} MB)")
    print(f"  continue 后: {mem_after_resume} MB (+{recovered} MB)")
    print(f"{'=' * 50}")

    if released > 50 and recovered > 0:
        print("\n✅ 验证通过：preserve_kv 模式正确释放和恢复了 GPU 显存")
    else:
        print("\n⚠️ 显存变化不明显，可能原因：")
        print("  1. 请求生成的 token 太少，KV cache 占用小")
        print("  2. HiCache 未正确启用（检查 --enable-hierarchical-cache）")
        print("  3. mem-fraction-static 预分配了大量显存")


if __name__ == "__main__":
    main()
