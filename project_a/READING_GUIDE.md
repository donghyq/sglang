# Project A 源码阅读顺序

建议按下列顺序读。每一节都先看函数入口，再沿调用链追状态和回退路径。

## 1. 请求与 Session 管理

先读控制面，不先读策略：

- [`TokenizerManager._handle_session_lifecycle_req_output()`](../python/sglang/srt/managers/tokenizer_manager.py#L3509)：把 lifecycle 控制响应归还给等待的 future。
- [`Scheduler.update_session_lifecycle()`](../python/sglang/srt/managers/scheduler.py#L5865)：管理事件的统一入口；异常时保留原生请求路径。
- [`SessionController.update_hint()`](../python/sglang/srt/session/session_controller.py#L451)：把 lifecycle hint 转为 demote / prefetch 或 native no-op。
- [`UnifiedSessionRefTracker.set_session_lifecycle()`](../python/sglang/srt/mem_cache/unified_cache/session_ref_tracker.py#L114)：保存会话状态，但不改变 KV 正确性。

阅读方法：先看 `update_session_lifecycle()` 的异常处理，再看 `update_hint()` 里每个 `return None`。这些 `None` 就是“没有足够证据时保持原生”的边界。

## 2. 原生 P/D 与 Decode prefix 复用

这部分是社区原生能力，只确认 Project A 挂接点：

- [`UnifiedRadixCache.prefetch_session()`](../python/sglang/srt/mem_cache/unified_radix_cache.py#L4634)：复用原生 `match_prefix()` 和 `prefetch_from_storage()`，不另建预取路径。
- [`UnifiedRadixCache.loading_check()`](../python/sglang/srt/mem_cache/unified_radix_cache.py#L3743)：原生加载完成轮询和 rank 同步边界。
- [`disaggregation/decode.py`](../python/sglang/srt/disaggregation/decode.py#L2474)：在 `DecodeTransferQueue` 提交前检查 `decode_prefix_len`，并记录 P/D 历史路径观测。

阅读方法：从 `prefetch_session()` 进入，先确认 synthetic handle 只用于触发原生查找；再看 `loading_check()` 和 Decode 提交点如何判断恢复是否完整。

## 3. 分组写入 commit-last manifest

这是首次写入的逐 Rank 原型，不是 CAS 或全跨 Rank 原子提交：

- [`MooncakeStore._manifest_key()`](../python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py#L774)：manifest key 包含 local rank、PP rank、CP rank。
- [`MooncakeStore._read_manifest()`](../python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py#L785)：校验 schema、logical key、generation 和去重后的 members。
- [`MooncakeStore._write_manifest()`](../python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py#L811)：最后写入 manifest；写失败抛错。
- [`MooncakeStore.batch_set_v1()`](../python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py#L1262)：首次写成员，成功后才写 manifest；已有 manifest 时复用原 generation。

阅读方法：按 `batch_set_v1()` → `_read_manifest()` → `_generation_key()` → `_batch_set_v1_impl()` → `_write_manifest()` 读。注意 `_manifest_key()` 的 rank 命名和 topology namespace 限制。

## 4. RuntimeRestoreCostGate 与 native 回退

先看 profile 校验，再看 rank vote：

- [`RuntimeRestoreCostGate.evaluate()`](../python/sglang/srt/mem_cache/restore_cost.py#L189)：校验 profile 并生成 restore / recompute 决策。
- [`RestoreCostGate.decide()`](../python/sglang/srt/mem_cache/restore_cost.py#L90)：保守比较 restore 估计、风险边际和 recompute 估计。
- [`UnifiedRadixCache._runtime_restore_cost_action()`](../python/sglang/srt/mem_cache/unified_radix_cache.py#L2990)：生成 rank vote 并决定恢复或重算。
- [`consensus_gate_action()`](../python/sglang/srt/mem_cache/restore_cost.py#L268)：profile 或 hit 签名不一致时回退恢复，避免 rank 分歧。

阅读方法：`off` / `shadow` 或样本不足都会走到 `permit=True` 的原生路径；只有所有 rank 的有效 `enforce` 决策一致时才可能拒绝恢复。再用 `_finish_restore_cost()` 看最终 outcome。

## 5. Lifecycle V1、Host 保留与 deadline admission

生命周期信号只影响调度，不授权复用：

- [`AgentLifecycle.remember_request()`](../python/sglang/srt/mem_cache/agent_lifecycle.py#L130)：只保存逻辑 token 数组和身份字段，不保存 request 对象或 KV handle。
- [`AgentLifecycle.admission()`](../python/sglang/srt/mem_cache/agent_lifecycle.py#L562)：检查 deadline、置信度、依赖收敛、容量、inflight 和恢复估计。
- [`AgentLifecycle.can_retain_host()`](../python/sglang/srt/mem_cache/agent_lifecycle.py#L621)：要求完整 Host 副本且写回不在途。
- [`UnifiedRadixCache.demote_agent_session()`](../python/sglang/srt/mem_cache/unified_radix_cache.py#L4320)：用原生 Host 引用保留副本；满足条件才释放设备层。

阅读方法：先读 `admission()` 的所有 reject reason，再读 `demote_agent_session()` 的 rank vote。重点确认 incomplete history、Host budget、peer mismatch 和无压力时都会保持原生路径。

## 6. FULL KV victim、DAG 消费者与 pending → ready

这一节是 Project A 的 DAG 原型：

- [`FullComponent._dag_stable_key()`](../python/sglang/srt/mem_cache/unified_cache/components/full.py#L219)：用逻辑前缀、extra key、cache salt 生成稳定 TP tie-break。
- [`FullComponent._session_ref_eviction_strategy()`](../python/sglang/srt/mem_cache/unified_cache/components/full.py#L358)：DAG 关闭时保留原生 LRU 顺序；开启时按未来使用距离排序。
- [`FullComponent._evict_device_start()`](../python/sglang/srt/mem_cache/unified_cache/components/full.py#L422)：建立有界保护预算，并记录 native/DAG 两个候选顺序。
- [`UnifiedRadixCache._agent_dag_victim_plan()`](../python/sglang/srt/mem_cache/unified_radix_cache.py#L4212)：对 EP prefetch 做无释放的容量试探。

阅读方法：从 `_evict_device_start()` 进入，看保护预算如何用稳定键排序；再看 `_evict_device_next_node()` 中 lock、load-back pending 和 evictable leaf 的检查。DAG 消费者的 `pending → ready` 表示业务依赖收敛；它不表示 KV 已回载。继续读 `AgentLifecycle.handle_dag_event()` 和 `admission()`，确认依赖归零后候选仍能接受准入评估。物理恢复完成另以服务端 GPU_READY 和实际请求消费为准。

## 7. 指标与测试

读机制时同步看可观测性和回归：

- [`MetricsCollector.observe_storage_prefetch_phase_duration()`](../python/sglang/srt/observability/metrics_collector.py#L2064)：原生 storage prefetch phase 指标入口。
- [`test_mooncake_group_semantics.py`](../test/registered/unit/mem_cache/test_mooncake_group_semantics.py)：generation / group 语义。
- [`test_runtime_restore_cost.py`](../test/registered/unit/mem_cache/test_runtime_restore_cost.py)：profile、rank vote、拒绝和清理。
- [`test_dag_kv_policy.py`](../test/registered/unit/mem_cache/test_dag_kv_policy.py)：FULL KV DAG 排序和调度边界。

补充测试：[`test_agent_lifecycle_v1.py`](../test/registered/unit/mem_cache/test_agent_lifecycle_v1.py) 覆盖 lifecycle V1；[`test_dag_agent_lifecycle_local.py`](../test/manual/mem_cache/test_dag_agent_lifecycle_local.py) 是 dependency-free 的本机检查。

## CPU 最小验证

```bash
python3 -m py_compile \
  python/sglang/srt/mem_cache/agent_lifecycle.py \
  python/sglang/srt/mem_cache/restore_cost.py \
  python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py \
  python/sglang/srt/mem_cache/unified_radix_cache.py \
  python/sglang/srt/mem_cache/unified_cache/components/full.py

python3 test/manual/mem_cache/test_dag_agent_lifecycle_local.py
```

这只覆盖语法和 standalone 逻辑，不是完整 SGLang runtime 验收，也不包含 GPU 路径。
