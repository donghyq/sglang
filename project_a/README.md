# Project A 学习入口

## 版本与范围

本目录用于在一个可学习的 SGLang fork 中阅读 Project A 的个人增量。当前快照固定为：

- 分支：`codex/project-a-study`
- 运行时代码修订：[`517f3acdfa2134199dcae28426b31728c6544c9e`](https://github.com/donghyq/sglang/commit/517f3acdfa2134199dcae28426b31728c6544c9e)。学习分支在其后增加整理提交；源码摘要见 [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json)。
- 公共上游基线：[`2f5c9ac43d76eef11e8561c89c55a79db7db715a`](https://github.com/sgl-project/sglang/commit/2f5c9ac43d76eef11e8561c89c55a79db7db715a)
- 相对基线：24 个个人提交，最终 28 个文件变更

仓库保留 SGLang 原生目录和 Python runtime。Project A 学习材料只放在 `project_a/`，不物理移动 runtime，也不改原测试。

## 阅读顺序

1. [ARCHITECTURE.md](ARCHITECTURE.md)：先看请求路径、原生能力和个人增量的边界。
2. [READING_GUIDE.md](READING_GUIDE.md)：按调用链读源码。
3. [COMPONENTS.json](COMPONENTS.json)：核对 SGLang、Dynamo、Mooncake 的精确修订。
4. [examples/restore-cost-profile.example.json](examples/restore-cost-profile.example.json)：只看配置格式；空样本表示保持原生路径。
5. [SOURCE_MAP.md](SOURCE_MAP.md)：核对原工作区的去向、已包含历史与单独保存的候选。

6. [replay/README.md](replay/README.md)：阅读从封存客户端提取的控制队列，在本机运行无网络、无模型的合成依赖示例。完整历史实验的私有部署资产不作为公开示例的依赖。

## 组件快照

| 仓库 | 分支 | 精确提交 | 上游基线 |
|---|---|---|---|
| [donghyq/sglang](https://github.com/donghyq/sglang) | `codex/project-a-study` | [`517f3acdfa2134199dcae28426b31728c6544c9e`](https://github.com/donghyq/sglang/commit/517f3acdfa2134199dcae28426b31728c6544c9e) | `2f5c9ac43d76eef11e8561c89c55a79db7db715a` |
| [donghyq/dynamo](https://github.com/donghyq/dynamo) | `codex/project-a-study` | [`1e68205353f69f5bca67b179c7bcc9f2c8cd61b8`](https://github.com/donghyq/dynamo/commit/1e68205353f69f5bca67b179c7bcc9f2c8cd61b8) | `ea89e8bdf` |
| [donghyq/Mooncake](https://github.com/donghyq/Mooncake) | `codex/project-a-study` | [`7b8330baf042de5e35ccdf025201de1b281e9c7b`](https://github.com/donghyq/Mooncake/commit/7b8330baf042de5e35ccdf025201de1b281e9c7b) | `3d88bf917ae8` |

## 个人增量边界

本分支的个人增量集中在：

- Session lifecycle 管理入口与请求身份的保守记忆；
- Mooncake generation manifest 的首次写入逐 Rank 原型；
- RuntimeRestoreCostGate 的 `off` / `shadow` / `enforce` 决策与原生回退；
- Lifecycle V1 的 Host 保留、deadline admission 和预取尝试；
- FULL KV 的 DAG 消费者、victim 排序和 `pending → ready` 边界；
- 相关指标和 CPU/单测覆盖。

社区原生的 P/D、HiCache、Mooncake Store、Dynamo KV-aware routing 不是自研能力。Project A 只在这些原生路径上加正确性、调度和生命周期边界。

## 不包含的内容

- 不包含新的最终架构方案，不新增 Router 或 Store。
- [candidates/](candidates/) 单独保存原工作区的 remote restore admission 与 Conductor 未集成补丁。它们没有应用到本分支的运行时代码。
- 旧版 V0、诊断和 SmartEvict 模拟器是历史材料，不在主学习路径。
- 本任务不发布模拟正收益。已有受控回放只说明存在有边界的结果，不能外推为生产收益。

## 本机最小验证

本机只做 CPU 级检查，不尝试启动完整 SGLang CUDA 服务：

```bash
python3 -m py_compile \
  python/sglang/srt/mem_cache/agent_lifecycle.py \
  python/sglang/srt/mem_cache/restore_cost.py \
  python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py \
  python/sglang/srt/mem_cache/unified_radix_cache.py \
  python/sglang/srt/mem_cache/unified_cache/components/full.py

python3 test/manual/mem_cache/test_dag_agent_lifecycle_local.py
python3 project_a/check_source.py
python3 -m project_a.replay.demo
python3 -m project_a.replay.demo --native
```

这些命令核对源码摘要、独立 DAG 逻辑和控制队列示例，不调用 GPU，也不验证完整 CUDA 推理环境。

## 一个本地窗口阅读三个组件

```bash
python3 project_a/checkout_components.py             # 只打印计划
python3 project_a/checkout_components.py --execute   # 从自己的 fork 取固定版本
```

然后打开 [project-a.code-workspace](project-a.code-workspace)。外部源码放在被 Git 忽略的 `.project-a-components/`，不会作为嵌套仓库提交进 SGLang。
