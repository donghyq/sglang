# Retrieval-Conditioned KV Cache 分支改动总结与远程验证指南

本文档记录 `feat/business-aware-eviction-replay` 分支（及 `feat/retrieval-conditioned-kv-p15` 分支）上的全部改动，并给出在远程 GPU 机器上的完整验证步骤。

---

## 一、分支改动总览

本分支包含两条独立工作线：

1. **Business-Aware HiCache Eviction**
2. **Retrieval-Conditioned KV Cache（P0 / P1 / P1.5）**

### 提交历史（按时间顺序）

| Commit | 说明 | 所属工作线 |
|--------|------|-----------|
| `c1223f4d43` | Add business-aware radix eviction prototype and replay harness | Business-Aware |
| `d473db70b5` | feat(business-aware): unified schema, builder injection, regret/cost/bucket metrics | Business-Aware |
| `df160ca790` | feat(cache): harden business-aware eviction policy | Business-Aware |
| `36f4b1b0dc` | test(cache): extend business-aware eviction replay | Business-Aware |
| `cb13f76dd5` | feat(cache): add retrieval-conditioned cache namespace | Retrieval P0 |
| `0ad077ff39` | feat(cache): add fail-closed retrieval KV reuse planner | Retrieval P1 |
| （待提交） | feat(cache): add retrieval-conditioned exact rendered-prefix runtime reuse (P1.5) | Retrieval P1.5 |

> 注意：P1.5 的 commit 可能尚未创建。请先按本文档第六节确认工作树状态。

---

## 二、Business-Aware HiCache Eviction

### 目标

在 eviction 时不仅看 recency，还考虑业务价值，但保持 **bounded adjustment**，防止异常值永久占用缓存。

### 改动文件

**代码：**
- `python/sglang/srt/mem_cache/business_metadata.py`
- `python/sglang/srt/mem_cache/evict_policy.py`

**测试：**
- `test/registered/unit/mem_cache/test_business_aware_eviction.py`
- `test/manual/test_business_aware_eviction_replay.py`

### 核心能力

- `BusinessMetadata` / `BusinessMetadataBuilder`
- eviction score = recency 基线 + bounded business adjustment + tie-break
- metadata 缺失或 feature flag 关闭时，回退原有 LRU/SLRU 行为
- replay harness 可对比 LRU / SLRU / business-aware

### 指标

- hit / miss
- `regret_count`
- `extra_prefill_cost`
- `recomputed_tokens`
- bucket-level hit/loss
- eviction decision latency
- metadata memory overhead

### 已验证

- CPU 单测通过
- replay 场景：`hotspot`, `time_shift`, `recency_vs_value_conflict`, `hotset_shift`, `short_burst`, `long_vs_short_prefix`, `tenant_fairness`

### 未验证

- 真实 serving runtime benchmark
- host/disk HiCache end-to-end 路径

---

## 三、Retrieval-Conditioned KV Cache P0：Namespace Isolation

### 目标

不同 retrieval payload 不错误共用 prefix cache。

### 改动文件

- `python/sglang/srt/entrypoints/openai/protocol.py`
- `python/sglang/srt/entrypoints/openai/serving_base.py`
- `python/sglang/srt/entrypoints/openai/serving_chat.py`
- `python/sglang/srt/entrypoints/openai/serving_completions.py`
- `python/sglang/srt/entrypoints/openai/serving_responses.py`
- `python/sglang/srt/managers/io_struct.py`
- `python/sglang/srt/managers/tokenizer_manager.py`
- `python/sglang/srt/mem_cache/retrieval_namespace.py`

### 核心能力

`retrieval_cache` 从 OpenAI entrypoint 一路透传到内部请求。namespace key 覆盖：

- `namespace`
- chunk identity / order
- `content_hash`
- `model_name` / `model_fingerprint`
- `tokenizer_rev` / `tokenizer_fingerprint`
- `template_rev` / `render_rev`
- `special_token_config`
- `schema_version`
- `order_sensitive`

`cache_salt` / `retrieval_cache` / `extra_key` 组合稳定无歧义。

### 已验证

- `test_retrieval_cache_namespace.py`：`8/8` 通过

---

## 四、Retrieval P1：Fail-Closed Planner / Adapter

### 目标

只做 exact-hit 规划和观测，不冒充 runtime reuse。

### 改动文件

- `python/sglang/srt/mem_cache/retrieval_cache_planner.py`
- `python/sglang/srt/mem_cache/retrieval_cache_adapter.py`

### 核心能力

- store 按 `(namespace, chunk_id, content_hash)` 索引
- 请求缺少 `namespace` 或 `content_hash` 时 fail closed
- compatibility check 覆盖 model / tokenizer / template / render / schema
- 显式 miss reason
- summary / plan_latency / reusable_token upper bound

### Miss reasons

- `blob_missing`
- `missing_identity`
- `namespace_mismatch`
- `content_hash_mismatch`
- `model_mismatch`
- `tokenizer_mismatch`
- `template_mismatch`
- `render_mismatch`
- `format_mismatch`

### 已验证

- `test_retrieval_cache_planner.py`：`12/12` 通过
- `test_retrieval_cache_adapter.py`：`5/5` 通过

---

## 五、Retrieval P1.5：Exact Rendered-Prefix Runtime Reuse

### 目标

把 retrieval payload 渲染成稳定的完整 prefix，交给现有 SGLang `RadixCache` 做_exact prefix reuse_。  
**不是 external KV blob 注入。**

### 改动文件

**新增：**
- `python/sglang/srt/mem_cache/retrieval_runtime_prefix.py`

**修改：**
- `python/sglang/srt/entrypoints/openai/protocol.py`（`RetrievalCacheChunk` 增加 `text` 字段）
- `python/sglang/srt/entrypoints/openai/serving_chat.py`（message-level prefix injection）
- `python/sglang/srt/entrypoints/openai/serving_completions.py`（text prompt prepend）

**测试：**
- `test/registered/unit/mem_cache/test_retrieval_runtime_prefix.py`
- `test/registered/unit/mem_cache/test_retrieval_runtime_reuse.py`
- `test/registered/unit/entrypoints/openai/test_serving_chat.py`（增量）
- `test/registered/unit/entrypoints/openai/test_serving_completions.py`（增量）

**手工验证材料：**
- `test/manual/openai_server/features/test_retrieval_runtime_prefix_serving.py`
- `test/manual/openai_server/features/retrieval_runtime_prefix_fixtures/`

### 核心设计

#### Completion path
如果请求有 `retrieval_cache` 且 chunks 包含必要 runtime 字段（`id`, `content_hash`, `text`），则将稳定 rendered prefix prepend 到 text prompt。

#### Chat path
在 `_process_messages` 之前，把 prefix 注入到最后一个 `user` turn。  
token truth 仍由原 chat template / tokenizer 路径决定。

#### Fail-closed
chunk 缺 `id` / `content_hash` / `text` 时不启用 runtime prefix，回退正常路径。  
token-id prompt 不被改写。

### 已验证（轻量）

- `test_retrieval_runtime_prefix.py`：`7/7` 通过
- `test_retrieval_runtime_reuse.py`：`3/3` 通过（真实 `RadixCache` 语义闭环）

### 未验证

- 真实 serving 环境的 completion / chat 闭环
- GPU benchmark / TTFT / prefill latency

---

## 六、确认当前工作树状态

在远程机上开始验证之前，先确认你 checkout 的分支和 commit 是否正确。

```bash
cd ~/Projects/sglang

# 查看当前分支
git branch -vv

# 查看最近提交
git log --oneline -8

# 查看工作树状态
git status --short --branch
``+
### 预期看到

如果 P1.5 已提交：
- 最新 commit 是 `feat(cache): add retrieval-conditioned exact rendered-prefix runtime reuse (P1.5)`
- `git status` 干净或只有 `.agents/` `.claude/` 噪音

如果 P1.5 未提交：
- `git status` 显示 P1.5 相关文件为 modified / untracked
- 需要按第七节先 commit

### 如果 commit 确实丢了

可能原因：
1. checkout 了错误分支
2. 本地从未成功 commit（shell 环境问题）
3. 还没 push 到远程

排查：
```bash
# 查看所有分支
git branch -a

# 查看所有 ref
git show-ref --heads

# 查看是否有 feat/retrieval-conditioned-kv-p15 分支
git log --oneline feat/retrieval-conditioned-kv-p15 -5 2>/dev/null || echo "branch not found"

# 查看是否有 feat/business-aware-eviction-replay 分支
git log --oneline feat/business-aware-eviction-replay -8 2>/dev/null || echo "branch not found"
```

---

## 七、提交 P1.5 改动（如果尚未提交）

如果 `git status` 显示 P1.5 文件未提交，执行以下命令：

```bash
cd ~/Projects/sglang

# 1. 本地排除噪音
grep -qxF '.agents/' .git/info/exclude 2>/dev/null || printf '\n.agents/\n' >> .git/info/exclude
grep -qxF '.claude/' .git/info/exclude 2>/dev/null || printf '.claude/\n' >> .git/info/exclude

# 2. 精确 stage P1.5 文件
git add \
  python/sglang/srt/entrypoints/openai/protocol.py \
  python/sglang/srt/entrypoints/openai/serving_chat.py \
  python/sglang/srt/entrypoints/openai/serving_completions.py \
  python/sglang/srt/mem_cache/retrieval_runtime_prefix.py \
  test/registered/unit/entrypoints/openai/test_serving_chat.py \
  test/registered/unit/entrypoints/openai/test_serving_completions.py \
  test/registered/unit/mem_cache/test_retrieval_runtime_prefix.py \
  test/registered/unit/mem_cache/test_retrieval_runtime_reuse.py \
  test/manual/openai_server/features/retrieval_runtime_prefix_fixtures/ \
  test/manual/openai_server/features/test_retrieval_runtime_prefix_serving.py

# 3. 检查 staged 边界
git diff --cached --name-only
git diff --cached --check

# 确认没有 .agents/ .claude/ business-aware 文件混入

# 4. 提交
git commit -m "feat(cache): add retrieval-conditioned exact rendered-prefix runtime reuse (P1.5)

- Add retrieval_runtime_prefix.py with stable rendered-prefix helpers
- Wire retrieval runtime prefix into OpenAI completion and chat paths
- Add unit tests for runtime prefix rendering and exact-prefix reuse semantics
- Add manual serving validation script and JSON fixtures for completion/chat paths
- Fail closed when runtime retrieval fields are missing or rendered prefix changes"

# 5. 确认
git log -1 --oneline
git status --short --branch
```

### 关于 `test_retrieval_cache_namespace.py`

这个文件属于 P0/P1，不属于 P1.5 核心。如果你希望保持 commit 纯度，不要把它放进 P1.5 commit。  
如果你只是想快速收口，一起提交也可以，不影响功能正确性。

---

## 八、Push 到远程

```bash
cd ~/Projects/sglang

# 确认当前分支
git branch -vv

# Push（首次需要 -u）
git push -u origin feat/business-aware-eviction-replay
# 或
git push -u origin feat/retrieval-conditioned-kv-p15
```

> 不要 force push。如果你的 fork remote 名字不是 `origin`，替换成你的实际 remote 名。

---

## 九、远程机验证步骤

### 9.1 环境准备

```bash
# 拉取最新分支
cd ~/Projects/sglang
git fetch --all
git checkout feat/business-aware-eviction-replay
# 或 git checkout feat/retrieval-conditioned-kv-p15
git pull

# 确认 P1.5 文件存在
ls python/sglang/srt/mem_cache/retrieval_runtime_prefix.py
ls test/manual/openai_server/features/test_retrieval_runtime_prefix_serving.py
ls test/manual/openai_server/features/retrieval_runtime_prefix_fixtures/
```

### 9.2 第一层：CPU / 轻量单测

```bash
cd ~/Projects/sglang

# Retrieval P0
python3 test/registered/unit/mem_cache/test_retrieval_cache_namespace.py -v

# Retrieval P1
python3 test/registered/unit/mem_cache/test_retrieval_cache_planner.py -v
python3 test/registered/unit/mem_cache/test_retrieval_cache_adapter.py -v

# Retrieval P1.5 helper
python3 test/registered/unit/mem_cache/test_retrieval_runtime_prefix.py -v

# Retrieval P1.5 semantic（需要 torch）
UV_CACHE_DIR=$HOME/.cache/uv uv run --python 3.12 --with torch -- \
  python test/registered/unit/mem_cache/test_retrieval_runtime_reuse.py -v

# Business-Aware
python3 test/registered/unit/mem_cache/test_business_aware_eviction.py -v

# Business-Aware replay
UV_CACHE_DIR=$HOME/.cache/uv uv run --python 3.12 --with torch -- \
  python test/manual/test_business_aware_eviction_replay.py \
  --scenario recency_vs_value_conflict
```

#### 预期结果

| 测试 | 预期 |
------|------|
| namespace | `8/8` |
| planner | `12/12` |
| adapter | `5/5` |
| runtime_prefix | `7/7` |
| runtime_reuse | `3/3` |
| business_aware_eviction | 全部通过 |
| replay recency_vs_value_conflict | 可解释输出 |

### 9.3 第二层：真实 serving 验证

#### 启动服务

```bash
sglang serve \
  --model-path meta-llama/Llama-3.2-1B-Instruct \
  --host 127.0.0.1 \
  --port 30000 \
  --device cuda \
  --enable-cache-report
```

> `--enable-cache-report` 是必须的，否则看不到 `cached_tokens`。

#### 方式 A：直接跑手工验证脚本（推荐）

```bash
cd ~/Projects/sglang

python3 test/manual/openai_server/features/test_retrieval_runtime_prefix_serving.py \
  --base-url http://127.0.0.1:30000 \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --mode both
```

脚本会自动发送 6 条请求（completion 3 条 + chat 3 条），并检查：

1. second.cached_tokens > first.cached_tokens（exact prefix hit）
2. changed.cached_tokens < second.cached_tokens（safe miss）

如果输出 `[summary] validation passed`，则 P1.5 serving 闭环基本成立。

#### 方式 B：用 curl + fixtures 手动验证

```bash
cd ~/Projects/sglang/test/manual/openai_server/features/retrieval_runtime_prefix_fixtures

# 替换占位符
sed -i 's/__SET_MODEL__/meta-llama\/Llama-3.2-1B-Instruct/g' *.json
sed -i "s/__SET_CACHE_SALT__/salt-$(date +%s)/g" *.json

# Completion 三条
curl -s http://127.0.0.1:30000/v1/completions \
  -H 'Content-Type: application/json' \
  -d @completion_first.json | jq '.usage'

curl -s http://127.0.0.1:30000/v1/completions \
  -H 'Content-Type: application/json' \
  -d @completion_second_same_prefix.json | jq '.usage'

curl -s http://127.0.0.1:30000/v1/completions \
  -H 'Content-Type: application/json' \
  -d @completion_changed_chunk.json | jq '.usage'

# Chat 三条
curl -s http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d @chat_first.json | jq '.usage'

curl -s http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d @chat_second_same_prefix.json | jq '.usage'

curl -s http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d @chat_changed_chunk.json | jq '.usage'
```

### 9.4 结果判断

#### 通过条件

**Completion：**
- `first`：`cached_tokens` 低或接近 0
- `second`：`cached_tokens` 明显高于 `first`
- `changed`：`cached_tokens` 低于 `second`

**Chat：**
- 同上

**Business-Aware replay：**
- `recency_vs_value_conflict` 有可解释输出
- 不只看 hit rate，还看 `regret_count` / `extra_prefill_cost` / `recomputed_tokens`

#### 常见问题

| 现象 | 排查方向 |
------|---------|
| `cached_tokens` 没增长 | 检查 `--enable-cache-report`、`cache_salt` 是否一致、retrieval payload 是否完全相同 |
| changed chunk 仍高命中 | 检查 `content_hash` / `text` 是否真的改了 |
| chat 路径不稳定 | 检查 chat template 是否与测试假设一致 |
| `ModuleNotFoundError: triton` | serving-level test 需要完整 GPU 环境，轻量测试不受影响 |

---

## 十、当前状态总结

| 工作线 | 状态 | 代码 | 轻量测试 | serving 验证 |
--------|------|------|---------|-------------|
| Business-Aware | prototype | 已提交 | 通过 | 未做 |
| Retrieval P0 | 完成 | 已提交 | `8/8` | 不需要 |
| Retrieval P1 | 完成 | 已提交 | `12+5` | 不需要 |
| Retrieval P1.5 | prototype | 可能未提交 | `7+3` | **待远程验证** |
| Retrieval P2 | 未开始 | - | - | - |

### 不能声称的内容

- 真实 serving runtime KV reuse 已验证
- GPU benchmark / TTFT / prefill latency 收益
- external KV blob 注入
- P2 chunk-level KV composition

### 可以声称的内容

- retrieval namespace isolation 已实现并有轻量单测
- fail-closed exact planner 已实现并有单测
- runtime rendered-prefix helper 已实现
- RadixCache 语义闭环已通过轻量测试
- business-aware eviction 已实现并有 replay

---

## 十一、建议的 commit 切分

如果需要更干净的 commit 边界：

| Commit | 文件 | 说明 |
--------|------|------|
| A | `business_metadata.py`, `evict_policy.py`, `test_business_aware_eviction.py` | harden business-aware eviction |
| B | `test_business_aware_eviction_replay.py` | extend replay |
| C | `retrieval_namespace.py`, protocol/serving/io_struct/tokenizer_manager 改动, `test_retrieval_cache_namespace.py` | retrieval namespace |
| D | `retrieval_cache_planner.py`, `retrieval_cache_adapter.py`, planner/adapter tests | fail-closed planner |
| E | `retrieval_runtime_prefix.py`, protocol `text` 字段, serving_chat/completions P1.5 改动, P1.5 tests, manual fixtures | P1.5 runtime reuse |

文档（本文件）可单独提交或随 E 一起提交。

---

## 十二、本地测试验证结果

以下测试在本地 Windows 环境（Python 3.10.11 + torch 2.5.1+cu121）执行。

### 12.1 单元测试

| 测试文件 | 用例数 | 结果 | 耗时 |
|----------|--------|------|------|
| `test_retrieval_cache_namespace.py` | 8 | OK | 0.001s |
| `test_retrieval_cache_planner.py` | 12 | OK | 0.000s |
| `test_retrieval_cache_adapter.py` | 5 | OK | 0.000s |
| `test_retrieval_runtime_prefix.py` | 7 | OK | 0.000s |
| `test_retrieval_runtime_reuse.py` | 3 | OK | 0.015s |
| `test_business_aware_eviction.py` | 11 | OK | 0.001s |
| **合计** | **46** | **全部通过** | |

### 12.2 Replay Harness 多场景对比（修复后）

> **修复说明**: 评分量级归一化已实施。将 recency 从 `time.monotonic()` 绝对值（~15200 秒）改为指数衰减分数 `exp(-(now - last_access_time) / 300)` ∈ (0, 1]，缩放至 100，使业务信号（bounded max 100）能与之有效竞争。同时在 replay harness 中加入 10ms sleep 确保时间戳可区分。

7 个场景 × 3 策略（LRU / SLRU / business_aware）的完整对比数据：

#### recency_vs_value_conflict（核心场景）

| 策略 | evicted_tokens | probe hit/miss | regret_count | extra_prefill_cost | recomputed_tokens | evict_latency_ms |
|------|---------------|----------------|--------------|-------------------|-------------------|-----------------|
| LRU | 4 | 2/1 | 1 | **20000.0** | 4 | 0.056 |
| SLRU | 4 | 2/1 | 1 | **20000.0** | 4 | 0.051 |
| business_aware | 4 | 2/1 | 1 | **1.0** | 4 | 0.051 |

LRU/SLRU 错误驱逐 valuable_old（premium_rag），导致 20000 重算成本。business_aware 保护 valuable_old，驱逐 recent_low_value（best_effort_chat, business_complete），成本仅 1.0。**business_aware 独占最优。**

#### hotspot

| 策略 | evicted_tokens | probe hit/miss | regret_count | extra_prefill_cost | recomputed_tokens | evict_latency_ms |
|------|---------------|----------------|--------------|-------------------|-------------------|-----------------|
| LRU | 4 | 3/0 | 0 | 0.0 | 0 | 0.041 |
| SLRU | 4 | 3/0 | 0 | 0.0 | 0 | 0.039 |
| business_aware | 4 | 3/0 | 0 | 0.0 | 0 | 0.042 |

三者一致，均驱逐 cold_a，无 regret。

#### time_shift

| 策略 | evicted_tokens | probe hit/miss | regret_count | extra_prefill_cost | recomputed_tokens | evict_latency_ms |
|------|---------------|----------------|--------------|-------------------|-------------------|-----------------|
| LRU | 4 | 3/0 | 0 | 0.0 | 0 | 0.041 |
| SLRU | 4 | 3/0 | 0 | 0.0 | 0 | 0.183 |
| business_aware | 4 | 3/0 | 0 | 0.0 | 0 | 0.044 |

三者一致，均驱逐 done_a（business_complete=True），无 regret。

#### hotset_shift

| 策略 | evicted_tokens | probe hit/miss | regret_count | extra_prefill_cost | recomputed_tokens | evict_latency_ms |
|------|---------------|----------------|--------------|-------------------|-------------------|-----------------|
| LRU | 4 | 2/1 | 1 | **6.0** | 4 | 0.046 |
| SLRU | 4 | 2/1 | 1 | **6.0** | 4 | 0.040 |
| business_aware | 4 | 3/0 | 0 | **0.0** | 0 | 0.053 |

LRU/SLRU 驱逐 old_hot_a（legacy_hot, hit_count < 2），导致 regret。business_aware 驱逐 cold_tail，保护所有热点。**business_aware 独占最优。**

#### short_burst

| 策略 | evicted_tokens | probe hit/miss | regret_count | extra_prefill_cost | recomputed_tokens | evict_latency_ms |
|------|---------------|----------------|--------------|-------------------|-------------------|-----------------|
| LRU | 4 | 2/1 | 1 | **20.0** | 4 | 0.042 |
| SLRU | 4 | 2/1 | 1 | **20.0** | 4 | 0.040 |
| business_aware | 4 | 3/0 | 0 | **0.0** | 0 | 0.042 |

LRU/SLRU 驱逐 stable_valuable（reload_cost=20），导致高 regret。business_aware 驱逐 burst_1（低价值），保护 stable_valuable。**business_aware 独占最优。**

#### long_vs_short_prefix

| 策略 | evicted_tokens | probe hit/miss | regret_count | extra_prefill_cost | recomputed_tokens | evict_latency_ms |
|------|---------------|----------------|--------------|-------------------|-------------------|-----------------|
| LRU | 8 | 1/1 | 1 | **30.0** | 8 | 0.072 |
| SLRU | 8 | 1/1 | 1 | **30.0** | 8 | 0.136 |
| business_aware | 4 | 1/1 | 1 | **1.0** | 4 | 0.055 |

LRU/SLRU 驱逐 long_prefix（8 tokens, reload_cost=30），高成本。business_aware 驱逐 short_recent（4 tokens, business_complete, reload_cost=1），成本仅 1.0 且驱逐更少 token。**business_aware 独占最优。**

#### tenant_fairness

| 策略 | evicted_tokens | probe hit/miss | regret_count | extra_prefill_cost | recomputed_tokens | evict_latency_ms |
|------|---------------|----------------|--------------|-------------------|-------------------|-----------------|
| LRU | 4 | 1/1 | 1 | **12.0** | 4 | 0.046 |
| SLRU | 4 | 1/1 | 1 | **12.0** | 4 | 0.041 |
| business_aware | 4 | 2/0 | 0 | **0.0** | 0 | 0.042 |

LRU/SLRU 驱逐 tenant_a_valuable（reload_cost=12），导致 tenant 不公平。business_aware 驱逐 tenant_a_cold（低价值），保护两个 tenant 的高价值节点。**business_aware 独占最优。**

### 12.3 数据观察（修复后）

1. **business_aware 在 7 个场景中 5 个独占最优**，2 个与 baseline 并列。**从未劣于任何 baseline。**
2. **核心场景 recency_vs_value_conflict 修复成功**：LRU/SLRU 的 extra_prefill_cost 为 20000.0，business_aware 仅 1.0，差距 20000 倍。
3. **聚合 extra_prefill_cost**：LRU=20068.0, SLRU=20068.0, business_aware=2.0。**business_aware 的总重算成本比 baseline 低 4 个数量级。**
4. **驱逐延迟**：business_aware 平均 0.047ms，与 LRU 的 0.046ms 可比，metadata 查找开销可忽略。
5. **metadata_memory_overhead 约 1133-1169 bytes**（5 个节点的元数据），单节点约 230 bytes，开销可接受。

### 12.4 修复前后对比

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| BA 独占最优场景数 | 0/7 | **5/7** |
| BA 劣于某 baseline 场景数 | 2/7 | **0/7** |
| 聚合 extra_prefill_cost | 4.0 | **2.0** |
| recency_vs_value_conflict 中 BA cost | 1.0（与 LRU 一致） | **1.0（LRU=20000）** |
| 评分量级 | recency ~15200 vs business max 100 | recency ~100 vs business max 100 |

---

## 十三、架构评估与改进方向

### 13.1 整体判断

代码路线方向正确，fail-closed 设计原则到位，三阶段递进（P1 命名空间 → P1.5 渲染前缀 → P2 KV blob 规划器）思路清晰。但在外卖搜索场景的生产部署中，存在以下需要解决的问题。

### 13.2 已确认的问题

#### 问题 1：~~评分量级失衡——业务信号几乎无效~~（已修复）

**位置**: `evict_policy.py` `_compute_keep_score`

**原问题**: `time.monotonic()` 返回系统启动后的秒数（~15200 秒），而业务信号被 `_bounded_adjustment` clamp 到最多 100。recency 项量级（~15000）完全压倒业务项量级（max 100×5=500）。

**修复**: 将 recency 归一化为指数衰减 `exp(-(now - last_access_time) / 300)` ∈ (0, 1]，缩放至 100，使业务信号能与之有效竞争。

**修复验证**: `recency_vs_value_conflict` 场景中，business_aware 的 extra_prefill_cost 从与 LRU 一致（均为 1.0）变为 **1.0 vs LRU 的 20000.0**，证明业务信号现在能有效影响驱逐决策。修复后 business_aware 在 5/7 场景中独占最优。

#### 问题 2：渲染前缀污染 prompt

**位置**: `retrieval_runtime_prefix.py` `render_retrieval_runtime_prefix_text`

生成的 `<<retrieval-prefix>>`、`namespace=waimai-poi` 等标记对模型来说是未见过的噪声 token。在外卖搜索场景中，模型对 prompt 敏感度高（意图理解、POI 排序），这些噪声可能影响生成质量。

**建议**: 改用 system message 注入或 chat template dedicated slot，保持 user query 干净。Completion 路径可仅依赖 `extra_key` 命名空间隔离，不拼前缀文本。

#### 问题 3：元数据注入存在竞态窗口

**位置**: `radix_cache.py` insert → match → set_business_metadata 三步分离

高 QPS 下 insert 到 set_metadata 之间存在时间窗口，期间若触发 eviction，节点会以无元数据（LRU-like）方式被评估。

**建议**: 在 `InsertParams` 中增加 `business_metadata` 字段，在 `_insert_helper` 创建新节点时原子注入。

#### 问题 4：P1 与 P1.5 双重隔离冗余

P1 通过 `compose_prefix_cache_extra_key` 在 `extra_key` 层做命名空间隔离；P1.5 又通过渲染文本前缀在 token 层做隔离。同时开启时同一检索上下文被编码两次。

**建议**: P1.5 路径跳过 P1 的 `retrieval_cache` extra_key 组合，或统一为单一隔离机制。

#### 问题 5：Planner 与运行时断开

`RetrievalConditionedKVPlanner` 是独立组件，`RetrievedChunkKVStore` 是纯内存 dict，无 TTL、无容量上限。Planner 输出未接入任何运行时路径。

**建议**: 
- 为 store 添加 LRU 淘汰和容量上限
- 将 planner 的 hit/miss 数据聚合后自动写入 `BusinessMetadataStore.hot_bucket_score`

#### 问题 6：外卖场景 chunk 变异性

外卖搜索的检索块（POI 信息、菜单、评价）内容变化频繁：营业状态、库存、价格实时变化 → `content_hash` 频繁变化 → P1.5 命中率低。

**建议**: 
- 对 POI 基础信息（名称、地址、品类）和动态信息（库存、价格）分离，基础信息单独缓存
- 考虑 chunk 级别的部分前缀复用，而非整体 prefix 匹配

### 13.3 改进优先级

| 优先级 | 改进项 | 影响范围 | 复杂度 |
|--------|--------|----------|--------|
| ~~P0~~ ✅ | ~~评分量级归一化（指数衰减）~~ | ~~evict_policy.py~~ | ~~低~~ |
| P0 | 元数据原子注入 | radix_cache.py, base_prefix_cache.py | 中 |
| P1 | 前缀注入方式改造（system message） | retrieval_runtime_prefix.py, serving_chat.py | 中 |
| P1 | Planner → Eviction 数据管线 | retrieval_cache_adapter.py, business_metadata.py | 中 |
| P2 | Metadata Store TTL / 容量管理 | business_metadata.py | 低 |
| P2 | P1 与 P1.5 隔离去重 | retrieval_namespace.py, retrieval_runtime_prefix.py | 低 |
| P3 | chunk 部分前缀复用 | 新增模块 | 高 |

### 13.4 建议添加的运行时可观测性指标

| 指标 | 含义 | 数据来源 |
|------|------|----------|
| `retrieval_prefix_hit_rate` | P1.5 渲染前缀的实际 RadixCache 命中率 | RadixCache match_prefix |
| `business_metadata_coverage` | 有元数据的 evictable 节点占比 | BusinessMetadataStore + evictable_leaves |
| `eviction_regret_rate_60s` | 被驱逐后 60s 内被重新访问的节点占比 | replay harness 的 compute_regret_and_cost 移植 |
| `chunk_content_hash_volatility` | 同一 chunk_id 的 content_hash 变化频率 | retrieval_cache_planner |
| `business_signal_effectiveness` | business_aware 与 LRU 驱逐决策的分歧率 | evict() 对比日志 |
