# 各工作区的代码去了哪里

## 以后从一个入口读

以本仓库 `codex/project-a-study` 为学习入口。运行时代码固定于 `517f3acdfa`，整理只增加导航、公开示例、源码校验和候选快照。上游基线为 `2f5c9ac43d`，可查看 [完整产品差异](https://github.com/donghyq/sglang/compare/2f5c9ac43d...517f3acdfa)。

## 原来的分支与本版的关系

| 原分支 | 代码版本 | 整理后的去向 |
|---|---|---|
| `codex/project-a-runtime-cost-gate` | `98ce3b109fa7` | 已包含在本版提交历史，直接读本版即可 |
| `codex/project-a-lifecycle-v1` | `30c5551954e4` | 已包含在本版提交历史，直接读本版即可 |
| `codex/projecta-lifecycle-admission-20261016` | `bed4ddabfbfc` | 已包含在本版提交历史，直接读本版即可 |
| `codex/projecta-ep-window-fix-20261018` | `517f3acdfa21` | 已包含在本版提交历史，直接读本版即可 |
| `codex/project-a-wait0-prefetch-diagnostic` | `d5b70cee1d88` | 历史独立分支，保留原工作区；未强行合并 |
| `codex/project-a-smart-evict-main` | `fc3a35f99ea6` | 历史独立分支，保留原工作区；未强行合并 |
| `validate/project-a-rebase-20260912` | `7d9cd24ed4b6` | 历史独立分支，保留原工作区；未强行合并 |
| `validate/project-a-gpu-prefix-20260912` | `623b4eb74843` | 历史独立分支，保留原工作区；未强行合并 |

旧诊断分支和早期策略比较的存在，不意味着它们是本学习版必须启用的功能。原工作区未删除，也没有覆盖其中未提交的内容。

## 不在 SGLang 仓库里复制第二份实现

| 组件 | 学习分支 | 版本 | 作用 |
|---|---|---|---|
| SGLang | 本仓库 `codex/project-a-study` | `517f3acdfa` runtime | 请求、缓存恢复、Cost Gate、Lifecycle、DAG victim |
| Dynamo | [donghyq/dynamo](https://github.com/donghyq/dynamo/tree/codex/project-a-study) | `1e68205353f6` | 路由扩展原型及 session identity 透传 |
| Mooncake | [donghyq/Mooncake](https://github.com/donghyq/Mooncake/tree/codex/project-a-study) | `7b8330baf042` | 可选的 Conductor 事件桥接 |

精确版本见 [COMPONENTS.json](COMPONENTS.json)。外部仓库仍保持自己的构建和依赖；Dynamo 的实测成本信号接线与 Conductor 候选的社区审阅状态，不因整理源码而变成已完成。

## 之前未提交的候选也有明确位置

[candidates/](candidates/) 保存 SGLang 请求级远端恢复准入，以及 Mooncake Conductor 工作区的未集成补丁；各自基线和文件摘要见清单。它们没有应用到本版 runtime。

[replay/](replay/) 保存控制队列的独立源码与标准库示例。原始完整回放依赖的部署脚本、内部输入、日志和资源清单继续保留在本地证据目录，不作为公开运行依赖。

## 检查整理是否改变了产品源码

```bash
python3 project_a/check_source.py
```

[SOURCE_MANIFEST.json](SOURCE_MANIFEST.json) 固定 28 个产品增量文件的内容摘要，并核对提取的控制队列函数。它验证源码保留，不替代功能与 GPU 性能验收。
