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
