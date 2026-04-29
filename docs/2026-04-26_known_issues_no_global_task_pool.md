# 已知问题：缺少全局任务池导致的多 client 与 failover 性能瓶颈 — 2026-04-26

本文档记录项目当前架构中**已识别但本次作业周期内未予修复**的一项核心性能缺陷，供后续迭代或下一阶段工作参考。该问题在功能测试中不会暴露（所有请求最终都能正确返回），但在多 client 并发或 worker 故障场景下会显著拖慢端到端时延。

## 1. 现象

1. **多 client 并发提交时无全局调度仲裁**：两个 client 同时提交请求，main_node 会为每个 client 起一条独立线程并各自走完 `dispatch → ThreadPoolExecutor.submit → wait` 的流程，dispatcher 是 stateless 的，因此可能把同一个 worker 的同一时间窗口同时分配给两个请求。没有任何排队、准入控制（admission control）或背压机制。
2. **failover 重派的 slice 必须在 survivor 跑完当前 slice 后才能开始执行**：单个请求内部，当某个 worker 失败时，重切的 retry slice 会立刻被 dispatcher 提交到 executor，但实际执行被 worker 端的连接锁阻塞，看起来就像"等所有人完成"。

## 2. 根因（重要：和直觉不一致）

两个现象其实是**同一个根因的两个面**——但不是"缺任务池"本身，而是缺一个**对所有正在排队/正在执行的 slice 有全局视图的调度器**。

具体到 failover 路径，[request_handler.py:369-372](../main_node/request_handler.py#L369-L372) 的 dispatcher 循环用的是 `concurrent.futures.wait(..., return_when=FIRST_COMPLETED)`，**第一个 future 完成（无论成功失败）就会立刻醒来**。失败 slice 在 [request_handler.py:413-430](../main_node/request_handler.py#L413-L430) 立刻被重切并 `executor.submit` 出去——dispatcher 层完全没有等所有人。

真正的阻塞在 worker 端的 `Connection.task_lock`。`_submit_retry` 自己的注释（[request_handler.py:328-347](../main_node/request_handler.py#L328-L347)）已经写明：

> When the survivor's `task_lock` is already held by an in-flight original slice, the retry will block inside `run_worker_task_slice` until that lock releases.

也就是说：每个 worker 同一时间只能跑一个 slice。retry 被按原始 gflops 比例重分给 N-1 个 survivor 之后，这 N-1 个全都正在跑各自的原始 slice，于是 retry 全员排队——**表象是"等所有人完成"，机制是 N-1 个独立的 per-worker 等待并发发生**。

多 client 场景同理：两个独立的 `ClientRequestHandler` 实例都在调用 dispatcher，谁也看不到谁正在占用哪个 worker。

## 3. 仅加任务池**解决不了**整个问题

这是本文档要强调的反直觉点。如果只在 main_node 加一个全局 FIFO 队列收 client 请求，那：

- ✅ 多 client 不再互相抢 worker（队列按序出，每次 dispatch 看到的是独占的 worker 集合）
- ❌ failover 仍然慢——retry slice 仍然只能被 push 到指定 survivor，仍然卡在 `task_lock` 上

要同时解决两个问题，**正确的形态是"全局可窃取队列 + worker 主动 pull"**：

| 改造点 | 作用 |
|--------|------|
| main_node 持有全局 slice 队列（含原始 slice 与 retry slice） | 多 client 自然分流；retry 不再需要预先选 survivor |
| worker 端从"被 push"改为"我闲了，给我下一个" | retry 自动落到当前最闲的 worker，无需 `_build_retry_assignments` 按 gflops 重分配 |
| `_run_assignments_with_failover` 退化为"提交 + 等结果聚合"，不再做 worker 选择 | dispatcher 大幅简化 |

副产物：`_build_retry_assignments` 中按 `original_weights` 加权切分的逻辑（[request_handler.py:230-280](../main_node/request_handler.py#L230-L280) 附近）可以删掉一大半。

## 4. 为什么本次未实施

- 改造涉及 worker 端协议从 push 模型改为 pull 模型，需要修改 wire 协议、worker 主循环、heartbeat 含义，超出本次作业的工程预算。
- 现有 push 模型已经能在功能正确性上覆盖所有评估场景；性能损失只在多 client 高频提交或 worker 故障率较高时才显著，与本次作业的评测重点（单 client 端到端正确性 + 异构后端调度）不重合。
- 真正治本还需要决定 worker 端是否允许并发执行多 slice（涉及 GPU/CPU 资源切分策略），这是独立的设计决定，不应与"加队列"绑定。

## 5. 相关代码位置

| 关注点 | 文件 | 行 |
|--------|------|---|
| Per-request executor，无跨请求视图 | [main_node/request_handler.py](../main_node/request_handler.py) | 282-438 |
| `FIRST_COMPLETED` 等待循环 | [main_node/request_handler.py](../main_node/request_handler.py) | 369-372 |
| Retry submit 后阻塞在 worker `task_lock` | [main_node/request_handler.py](../main_node/request_handler.py) | 328-347 |
| Retry slice 按原始 gflops 重分配 | [main_node/request_handler.py](../main_node/request_handler.py) | 230-280 |
| 多 client accept 循环（入口侧已是 multi-client clean） | [main_node/connection_service.py](../main_node/connection_service.py) | 208-250 |
| Per-client 服务线程 | [main_node/connection_service.py](../main_node/connection_service.py) | 163-170 |

## 6. 后续如要落地的最小步骤建议

1. 在 main_node 增加 `GlobalSliceQueue`，以 `(request_id, slice)` 入队，含优先级字段供 retry slice 插队
2. 增加 wire 协议消息 `WORKER_REQUEST_WORK` / `WORKER_NO_WORK_AVAILABLE`
3. worker 主循环改为：完成一个 slice → 发 `WORKER_REQUEST_WORK` → 阻塞等下一个 → 收到则执行
4. `_run_assignments_with_failover` 退化为：将 slice 全部 enqueue → 等所有 task_id 的结果回流 → 聚合
5. heartbeat 保留为 liveness 信号，不再承担"分配新任务"的隐式作用

如未来推进上述方向，建议同时引入端到端 benchmark：模拟 3 client × 持续提交，对比改造前后的尾延迟（p95/p99）与 worker 利用率曲线。
