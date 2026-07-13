# Retrieval P1.5 Mock Data

这组 mock 数据用于手工验证 **Retrieval P1.5: exact rendered-prefix runtime reuse**。

目标闭环：

1. **相同 retrieval payload + 不同 query suffix**
   - 第一条请求：cold prefill
   - 第二条请求：应出现更高的 `cached_tokens`
2. **retrieval content/template 变化**
   - 应安全 miss / fallback
   - `cached_tokens` 应低于 exact-hit warmed 请求

## 使用前需要替换的占位符

- `__SET_MODEL__`：替换成实际启动的模型名
- `__SET_CACHE_SALT__`：每次验证建议换一个新值，避免复用旧缓存

## 建议验证顺序

### Completion

1. `completion_first.json`
2. `completion_second_same_prefix.json`
3. `completion_changed_chunk.json`

### Chat

1. `chat_first.json`
2. `chat_second_same_prefix.json`
3. `chat_changed_chunk.json`

## 建议服务启动参数

至少需要：

```bash
sglang serve \
  --model-path <your-model> \
  --host 127.0.0.1 \
  --port 30000 \
  --enable-cache-report
```

## curl 示例

```bash
curl -s http://127.0.0.1:30000/v1/completions \
  -H 'Content-Type: application/json' \
  -d @completion_first.json | jq '.usage'
```

```bash
curl -s http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d @chat_first.json | jq '.usage'
```

## 期望观察

### Completion

- `completion_first.json`：`cached_tokens` 较低或接近 0
- `completion_second_same_prefix.json`：`cached_tokens` 高于第一条
- `completion_changed_chunk.json`：`cached_tokens` 低于第二条

### Chat

- `chat_first.json`：`cached_tokens` 较低或接近 0
- `chat_second_same_prefix.json`：`cached_tokens` 高于第一条
- `chat_changed_chunk.json`：`cached_tokens` 低于第二条

## 正确性边界

这些 mock 数据验证的是：

- rendered retrieval prefix 是否稳定
- 相同 prefix 是否能转化为 SGLang 现有 prefix cache 命中
- retrieval 内容变化时是否 safe miss

它们**不证明**：

- external KV blob 注入
- chunk-level KV composition
- RoPE 修正
- GPU benchmark 收益
