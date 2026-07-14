# Partial Rollout KV Cache Resume 验证脚本

本目录包含验证 SGLang `preserve_kv` pause/resume 机制的脚本。

## 前提条件

### 硬件

| 项目 | 最低要求 | 推荐 |
|---|---|---|
| GPU 显存 | 8GB | 12GB+ |
| 模型 | Qwen2.5-0.5B-Instruct | Qwen2.5-1.5B-Instruct |
| Python | 3.10+ | 3.12 |

### 安装

```bash
cd ~/Projects/sglang
git checkout feat/partial-rollout-kv-resume
pip install -e "python[all]"
pip install requests

# 下载模型
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir ~/models/Qwen2.5-0.5B-Instruct
```

## 启动 Server

### 带 HiCache（KV-aware 验证）

```bash
python -m sglang.launch_server \
    --model-path ~/models/Qwen2.5-0.5B-Instruct \
    --port 30000 \
    --enable-hierarchical-cache \
    --hicache-ratio 2.0 \
    --hicache-write-policy write_through \
    --context-length 4096 \
    --mem-fraction-static 0.7
```

### 不带 HiCache（Token-only 对照）

```bash
python -m sglang.launch_server \
    --model-path ~/models/Qwen2.5-0.5B-Instruct \
    --port 30001 \
    --context-length 4096 \
    --mem-fraction-static 0.7
```

## 验证步骤

### 步骤 1：基础功能验证

```bash
python test/registered/rl/verify_partial_rollout.py --port 30000
```

验证内容：
- 全局 `pause_generation(preserve_kv)` + `continue_generation`
- per-request `pause_request` / `resume_request`
- `pause_all` 暂停所有请求

### 步骤 2：KV 复用验证

```bash
# 需要同时启动两个 server（端口 30000 和 30001）
python test/registered/rl/verify_kv_reuse.py

# 如果只有一个 server，跳过对照
python test/registered/rl/verify_kv_reuse.py --skip-nocache
```

验证内容：
- 相同 prompt 二次发送的 TTFT（前缀缓存命中）
- `preserve_kv` 暂停后重新发送的 TTFT（Host 层 KV 复用）
- 对照：无 HiCache server 的 token-only resume

预期结果：preserve_kv 恢复后 TTFT 应明显低于首次。

### 步骤 3：显存验证

```bash
python test/registered/rl/verify_gpu_memory.py --port 30000
```

验证内容：
- pause 后 GPU 显存下降（KV cache 释放到 Host）
- continue 后 GPU 显存回升（KV cache 从 Host 加载回来）

## 预期结果

| 测试项 | 预期 | 说明 |
|---|---|---|
| pause(preserve_kv) | 不报错 | server 正常暂停 |
| continue | 正常恢复 | 可继续处理请求 |
| pause_request | 指定请求暂停 | 其他请求不受影响 |
| 恢复后 TTFT | < 首次 TTFT × 0.8 | 命中 Host 层 KV |
| pause 后显存 | 下降 50MB+ | KV cache 释放 |
| continue 后显存 | 回升 | KV cache 加载回来 |

## 故障排查

### server 启动失败

```bash
# 检查 CUDA
python -c "import torch; print(torch.cuda.is_available())"

# 减少显存占用
--mem-fraction-static 0.5
```

### HiCache 未启用

server 日志应出现 `Enable hierarchical cache`。如果没有，检查 `--enable-hierarchical-cache`。

### write_backup 报错

如果日志出现 `write_backup failed`，确保 `--enable-hierarchical-cache` 已设置且 `--hicache-ratio` > 0。

### TTFT 没有下降

1. 检查 server 日志是否有 `pause_generation:preserve_kv` 输出
2. 增大 `--hicache-ratio`（如 4.0）
3. 确认 `--hicache-write-policy write_through`

### 端口冲突

修改 `--port` 参数，脚本用 `--port` 指定。

## 脚本说明

| 脚本 | 用途 | 依赖 |
|---|---|---|
| `verify_partial_rollout.py` | 基础功能验证（pause/resume 不报错） | requests |
| `verify_kv_reuse.py` | KV 复用验证（TTFT 对比） | requests |
| `verify_gpu_memory.py` | 显存验证（GPU 内存变化） | requests, nvidia-smi |

所有脚本都是独立的 Python 文件，只需 `pip install requests` 即可运行。
不需要 pytest，不需要 GPU 训练环境。
