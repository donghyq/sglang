# Project A 架构阅读图

## 请求主路径

```text
Agent / OpenAI-compatible request
  -> Dynamo frontend / existing KV-aware router
  -> SGLang TokenizerManager
  -> Scheduler / SessionController
  -> UnifiedRadixCache prefix match
  -> native P/D + HiCache + Mooncake storage
  -> scheduler admission / decode

lifecycle management event
  -> TokenizerManager control response
  -> Scheduler.update_session_lifecycle()
  -> SessionController / AgentLifecycle
  -> cache admission, retention, prefetch or native fallback
```

生成请求仍走 SGLang 原生请求路径。Project A 的 lifecycle、cost gate、DAG 提示都挂在管理入口或缓存准入边界上，不新增第二套请求调度路径。

## 分层边界

| 层 | 原生能力 | Project A 个人增量 | 阅读入口 |
|---|---|---|---|
| Session | 原生 session radix cache 和引用管理 | lifecycle state、保守请求身份、DAG 事件入口 | [`python/sglang/srt/managers/scheduler.py`](../python/sglang/srt/managers/scheduler.py#L5865) |
| Prefix / P/D | 原生 prefix match、P/D 传输、Decode prefix 复用 | 只在观测和准入边界补齐一致性 | [`python/sglang/srt/mem_cache/unified_radix_cache.py`](../python/sglang/srt/mem_cache/unified_radix_cache.py#L4634) |
| Mooncake | 原生 Mooncake Store 与 shared L3 | generation manifest、逐 Rank key、commit-last 发布 | [`python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py`](../python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py#L774) |
| Restore choice | 原生按需恢复 / 重算 | measured profile 的 off/shadow/enforce 与 rank vote | [`python/sglang/srt/mem_cache/restore_cost.py`](../python/sglang/srt/mem_cache/restore_cost.py#L127) |
| Lifecycle | 原生 LRU、引用、Host/GPU 分层 | V1 deadline admission、Host 保留、受限预取 | [`python/sglang/srt/mem_cache/agent_lifecycle.py`](../python/sglang/srt/mem_cache/agent_lifecycle.py#L562) |
| FULL KV / DAG | 原生 tree eviction | DAG 消费者提示、稳定逻辑键、victim 排序 | [`python/sglang/srt/mem_cache/unified_cache/components/full.py`](../python/sglang/srt/mem_cache/unified_cache/components/full.py#L358) |

## 关键边界

1. **generation manifest 是首次写入的逐 Rank 原型。** `_manifest_key()` 按 local rank / PP / CP 维度命名，`_write_manifest()` 最后写入。它不是 CAS，也不是全跨 Rank 原子提交；布局和 topology namespace 限制仍然保留。
2. **cost gate 未知时回退原生。** `RuntimeRestoreCostGate.evaluate()` 未找到有效 profile、样本不足或 mode 不是 `enforce` 时返回原生恢复；rank profile 不一致也会被 `consensus_gate_action()` 拦下。
3. **lifecycle 不授权 KV 复用。** 生命周期状态只影响保留、降层、预取和释放优先级；真实复用仍由 token / page identity、KV compatibility、完整副本和容量决定。
4. **DAG 只改变排序和准入。** `FullComponent._dag_stable_key()` 用逻辑前缀做 TP tie-breaking，`_session_ref_eviction_strategy()` 只改变 FULL KV victim 顺序；锁、引用和原生容量检查仍生效。
5. **路由与事件目录职责明确。** Dynamo 是既定请求路由入口；其中 Agent-cache routing 是扩展原型，不能视为真实成本信号已完成接线。可选的是 Conductor 事件目录，它不成为第二个 Router。本 SGLang 快照不新增 Router 或 Store。

## 外部组件快照

精确版本见 [COMPONENTS.json](COMPONENTS.json)。三个 fork 均未声明已上游合并。

- Dynamo：包含 0eff 路由原型和被动 session identity 桥接；真实 runtime 成本信号路由接线尚未完整验收。
- Mooncake：Conductor 接 SGLang 事件桥接；它是可选目录 / 事件面，不是请求必经路由。
- SGLang：本目录所在仓库，保留完整原生源码。

## 证据边界

本文件描述的是当前提交的源码结构。已有受控回放和测试可以证明机制路径，但受样本、场景和配置限制，不能外推为生产收益或整体正收益结论。
