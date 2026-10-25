# 按依赖顺序发送提示，让 Resume 独立发出

`control_dispatch.py` 提取封存客户端的 `workflow_control_queue()`，保留函数实现，移除了依赖个人目录的旧回放加载器。它是调用方接线，不是第二套 SGLang 恢复路径。

## 本机先运行

在仓库根目录执行：

```bash
python3 -m project_a.replay.demo
python3 -m project_a.replay.demo --native
```

示例只使用 Python 标准库和本地异步延迟，不调用模型、GPU、网络或实际服务。两个工具实际结束以后，依赖提示与 ACTIVE 由一个后台任务按顺序发送，示例 Resume 随即登记；COMPLETED 与 workflow_finish 等待队列收尾。输出中的耗时只说明本地示例的调用顺序，不是模型 TTFT 或预取收益。

## 再读真实实现

1. `workflow_control_queue()` 接收原调用方和两类已有控制方法，返回工具包装、生命周期包装、DAG 包装、统计和 drain。
2. 每个工作流只有一个队列发送任务。各依赖实际完成后先进入队列，ACTIVE 排在其后。
3. Resume 不等待这些性能提示；终态更新通过 drain 保持顺序。工具开始前的控制仍由原调用方同步处理。
4. `queued=true` 仅说明入队。预取是否完成，要看服务端逐 Rank 的搬运/安装与使用事件。
5. 取消 asyncio 任务不能停止已经运行的阻塞 I/O 线程。`uncertain_after_drain`、`io_thread_may_continue` 等字段必须保留，不能据任务取消宣称传输已停止。

`SyntheticRunner` 只展示适配器需要提供的接口。接入真实业务时沿用业务自己的请求、工具和控制客户端；不要把示例工具时长、提前量或伪造回包用于性能验收。历史请求数据、内部部署脚本与受控实验输出保留在本地证据目录，没有打包进公开学习仓库。
