# Partial Rollout KV Cache Resume — 验证指南

## 目标

验证 SGLang `preserve_kv` pause/resume 机制可用，收集关键性能数据决定方案是否继续推进。

## 前提条件

### 硬件

| 项目 | 最低要求 | 推荐 |
|---|---|---|
| GPU 显存 | 8GB | 12GB+ |
| 模型 | Qwen2.5-0.5B-Instruct (~1GB) | Qwen2.5-1.5B-Instruct (~3GB) |
| Python | 3.10+ | 3.12 |
| 依赖 | requests | requests |

### 安装

```bash
# SGLang（改动分支）
cd ~/Projects/sglang
git checkout feat/partial-rollout-kv-resume
pip install -e "python[all]"
pip install requests

# VeRL（E2E 验证用）
cd ~/Projects/verl
git checkout feat/partial-rollout
# 不需要完整安装，E2E 脚本直接加载模块

# 模型
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir ~/models/Qwen2.5-0.5B-Instruct
```

---

## 验证步骤

### 第一步：启动 SGLang Server

```bash
cd ~/Projects/sglang

# 带 HiCache（KV-aware 验证）
python -m sglang.launch_server \
    --model-path ~/models/Qwen2.5-0.5B-Instruct \
    --port 30000 \
    --enable-hierarchical-cache \
    --hicache-ratio 2.0 \
    --hicache-write-policy write_through \
    --context-length 4096 \
    --mem-fraction-static 0.7
```

等待日志出现 `The server is fired up and ready to roll!`。

### 第二步：确认 HiCache 已启用

在 server 日志中搜索：

```bash
# 方法 1：启动时检查日志输出
# 应看到类似: "Enable hierarchical cache" 或 "HiCache" 或 "hierarchical_cache"

# 方法 2：调用 API 确认
curl -s http://localhost:30000/get_server_info | python3 -c "
import sys, json
info = json.load(sys.stdin)
# 搜索 hierarchical / hicache 相关字段
for k, v in info.items():
    if 'cache' in k.lower() or 'hi' in k.lower():
        print(f'{k}: {v}')
"
```

**需要的结果 #1**：贴出包含 `hierarchical` 或 `hicache` 的日志行（1-2 行即可）。

### 第三步：SGLang 侧基础功能验证

```bash
cd ~/Projects/sglang
python test/registered/rl/verify_partial_rollout.py --port 30000
```

预期输出：
```
=== 测试 1: 全局 pause_generation(preserve_kv) ===
  生成完成
  暂停 (preserve_kv)...
  {'message': 'Generation paused successfully.', 'status': 'ok'}
  恢复...
  恢复后正常: <model answer>
  测试 1 通过

=== 测试 2: per-request pause_request ===
  ...
  测试 2 完成

=== 测试 3: pause_all ===
  ...
  测试 3 通过

=== 所有基础测试通过 ===
```

**需要的结果 #2**：贴出完整输出。如果失败，贴出报错信息。

### 第四步：显存验证

```bash
python test/registered/rl/verify_gpu_memory.py --port 30000
```

预期输出：
```
Server: http://localhost:30000

基线显存 (server 启动后):  XXXX MB
发送生成请求（占用 KV cache）...
生成后显存:               XXXX MB (↑YYY MB)

暂停 (preserve_kv)...
暂停后显存:               XXXX MB (↓ZZZ MB)
  KV cache 释放: ✅

恢复...
恢复后显存:               XXXX MB (↑WWW MB)
  KV cache 回升: ✅
```

**需要的结果 #3**：贴出完整输出，特别是三个数字：
- 生成后显存
- 暂停后显存（应该比生成后低）
- 恢复后显存（应该回升）

如果暂停后显存没有下降，说明 `preserve_kv` 模式没有正确释放 KV cache。

### 第五步：KV 复用 TTFT 对比（关键）

```bash
python test/registered/rl/verify_kv_reuse.py --skip-nocache
```

预期输出：
```
=== HiCache: 前缀缓存命中验证 ===
  首次  TTFT: 0.XXXs, ...
  二次  TTFT: 0.XXXs, ...
  TTFT 加速比: X.XXx

=== HiCache: preserve_kv 暂停后复用验证 ===
  首次 TTFT: 0.XXXs
  暂停 (preserve_kv)...
  恢复...
  恢复后 TTFT: 0.XXXs
  TTFT 比值: 0.XXx (命中 ✅ / 未命中 ❌)
```

**需要的结果 #4**（最关键）：贴出两个 TTFT 数字和比值：
- preserve_kv 恢复后 TTFT
- 首次 TTFT
- 比值 = 恢复后 / 首次

判据：
- 比值 < 0.8 → KV 复用生效，方案继续推进 ✅
- 比值 >= 0.8 → KV 没有被保留或 match_prefix 没命中，需要排查 ❌

### 第六步：VeRL 端到端验证

```bash
cd ~/Projects/verl
python tests/experimental/partial_rollout/verify_e2e_real_sglang.py --port 30000
```

预期输出：
```
SGLang server: http://localhost:30000
Server healthy

=== Test 1: Basic pause/save/resume/complete (KV-aware) ===
  ...
  Recomputed: 0 (KV-aware OK)
  Test 1 PASSED

=== Test 2: Token-only resume (release KV) ===
  ...
  Recomputed: 5 (token-only OK)
  Test 2 PASSED

=== Test 3: Weight version mismatch ===
  ...
  Correctly rejected: ...
  Test 3 PASSED

=== Test 4: TTFT comparison (preserve_kv) ===
  First TTFT:  0.XXXs
  Resume TTFT: 0.XXXs
  Ratio: 0.XXx (HIT/MISS)
  Test 4 done

=== Test 5: Concurrent paused rollouts ===
  ...
  Test 5 PASSED

E2E Verification Summary
  basic               PASS
  token_only          PASS
  weight_mismatch     PASS
  ttft                0.XXXs -> 0.XXXs (0.XXx)
  concurrent          PASS
```

**需要的结果 #5**：贴出 Verification Summary 部分。

### 第七步：检查 server 日志中的关键路径

```bash
# 在 server 的日志中搜索这些关键词
grep -i "preserve_kv\|write_backup\|pause_request\|pause_generation" /path/to/sglang/server.log
```

**需要的结果 #6**：贴出包含以下关键词的日志行：
- `pause_generation:preserve_kv` — 确认 preserve_kv 分支被执行
- `write_backup` — 确认 KV 被写入 Host（如果有日志的话）
- `pause_request` — 确认 per-request 暂停被执行

---

## 需要收集的结果汇总

跑完上述步骤后，把以下 6 项结果贴回来：

| 编号 | 结果 | 来源 | 判据 |
|---|---|---|---|
| #1 | HiCache 启用确认 | 第二步 | 日志中有 hierarchical cache |
| #2 | 基础功能测试输出 | 第三步 | 所有测试通过 |
| #3 | 显存变化数字 | 第四步 | 暂停后显存下降 50MB+ |
| #4 | TTFT 对比数字 | 第五步 | 比值 < 0.8 为通过 |
| #5 | E2E Verification Summary | 第六步 | 所有 functional 测试 PASS |
| #6 | server 日志关键词 | 第七步 | 有 preserve_kv 和 write_backup |

**最关键的是 #4（TTFT 比值）和 #3（显存变化）**。这两个数据点决定方案是否继续推进。

---

## 结果模板

复制以下模板填写：

```
## 验证环境
- GPU: [型号, 显存]
- 模型: Qwen2.5-0.5B-Instruct
- SGLang branch: feat/partial-rollout-kv-resume
- VeRL branch: feat/partial-rollout
- HiCache: [启用/未启用]

## 结果 #1: HiCache 确认
[贴日志行]

## 结果 #2: 基础功能测试
[贴输出]

## 结果 #3: 显存变化
- 生成后: XXX MB
- 暂停后: XXX MB (下降 XXX MB)
- 恢复后: XXX MB (回升 XXX MB)

## 结果 #4: TTFT 对比
- 首次 TTFT: 0.XXX s
- 恢复后 TTFT: 0.XXX s
- 比值: 0.XXx
- 判定: [命中/未命中]

## 结果 #5: E2E Summary
[贴 Verification Summary]

## 结果 #6: Server 日志
[贴含 preserve_kv / write_backup 的行]
```

---

## 故障排查

### server 启动失败

```bash
# 检查 CUDA
python -c "import torch; print(torch.cuda.is_available())"

# 减少显存占用
--mem-fraction-static 0.5

# 关闭 HiCache 先验证基础功能
# 去掉 --enable-hierarchical-cache
```

### HiCache 未启用

确认 `--enable-hierarchical-cache` 参数已传入。server 日志应包含 hierarchical cache 相关输出。

### preserve_kv 报错

如果 `pause_generation(mode="preserve_kv")` 返回 422 或 500：
1. 确认 SGLang 分支正确：`git branch --show-current` 应为 `feat/partial-rollout-kv-resume`
2. 确认 io_struct.py 包含 preserve_kv：`grep preserve_kv python/sglang/srt/managers/io_struct.py`

### TTFT 没有下降

1. 检查 server 日志是否有 `pause_generation:preserve_kv` 输出
2. 增大 HiCache 比例：`--hicache-ratio 4.0`
3. 确认 write-policy：`--hicache-write-policy write_through`
4. 检查 `write_backup` 是否被调用（搜索日志）

### VeRL E2E 脚本报 ImportError

```bash
# 确保在 verl 仓库根目录运行
cd ~/Projects/verl
python tests/experimental/partial_rollout/verify_e2e_real_sglang.py --port 30000

# 如果缺少 requests
pip install requests
```

### 端口冲突

修改 `--port` 参数，验证脚本用 `--port` 指定。

---

## 脚本说明

| 仓库 | 脚本 | 用途 |
|---|---|---|
| sglang | `test/registered/rl/verify_partial_rollout.py` | 基础功能（pause/resume 不报错） |
| sglang | `test/registered/rl/verify_gpu_memory.py` | 显存验证（GPU 内存变化） |
| sglang | `test/registered/rl/verify_kv_reuse.py` | TTFT 对比（KV 复用验证） |
| verl | `tests/experimental/partial_rollout/verify_e2e_real_sglang.py` | 端到端（VeRL manager + 真实 SGLang） |

所有脚本只需 `pip install requests`，不需要 pytest，不需要 GPU 训练环境。
