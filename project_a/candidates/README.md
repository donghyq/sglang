# 单独保存的未集成候选

这两个补丁保存整理时已有的未提交代码，方便阅读或继续审阅。它们**没有应用到学习主线**，也不属于 `517f3acdfa` 的实机验证版本。

| 补丁 | 用途 | 状态 |
|---|---|---|
| [sglang-remote-restore-admission.patch](sglang-remote-restore-admission.patch) | 请求级远端恢复准入及对应测试 | 待单独验证和社区审阅 |
| [mooncake-conductor-pending.patch](mooncake-conductor-pending.patch) | Conductor 工作区中尚未封存的改动 | 待单独验证和社区审阅 |

[MANIFEST.json](MANIFEST.json) 给出各自基线、文件摘要和补丁摘要。整理只检查了补丁能否应用到所列基线，没有据此确认功能、收益或可合并性。两个补丁属于不同仓库，不能直接应用到本学习分支。

需要研究候选时，在对应仓库的独立分支检出清单中的 `base_revision`，先运行 `git apply --check <patch>`，再审阅和验证。原有未提交工作区保留；本目录是快照，不会随那些工作区自动更新。
