# 单机 2×Metal + 1×CPU 共享 GPU 并发行为分析 — 2026-04-22

本次测试目的是在单台 M1 Pro Mac 上启动 1 个 main_node + 3 个 compute_node（2 个以 Metal 为后端、1 个以 CPU 为后端），观察两个 Metal worker 争抢同一块物理 GPU 时的行为，以及 CPU worker 并存时的调度、心跳、数据面表现。结论：**系统行为完全符合单机共享 GPU 的物理约束，没有发现 bug**。

## 测试配置

| 角色 | 日志 | backend | 自报算力 |
|------|------|---------|---------|
| main_node | `main-20260422-221819.txt` | — | — |
| worker-1 (peer-metal) | `worker-20260422-221854.txt` | Metal | gemv 80.021 GFLOPS / conv2d 955.035 GFLOPS |
| worker-2 (metal) | `worker-20260422-222556.txt` | Metal | gemv 80.021 GFLOPS / conv2d 955.035 GFLOPS |
| worker-3 (peer-cpu) | `worker-20260422-222558.txt` | CPU | gemv 52.488 GFLOPS / conv2d 202.933 GFLOPS |

四个进程全部跑在同一台 Mac（`192.168.1.136`，M1 Pro，10 核 CPU，16 GB 统一内存）上。数据面走 loopback，实测 recv 带宽 5–7 GB/s，与内核 memcpy 一致，可以排除任何"跨机"假象。

客户端运行窗口 2026-04-22 23:12:18 – 23:16:39，共发起 8 次任务（gemv × 4，conv2d × 4），均 `status=200`。

## 关键观察

### 1. 两个 Metal worker 在同一块 GPU 上被精确对半分（符合预期）

conv2d `size=large, iter=1` 的计时（`fixe-3 = conv2d-12`）：

| worker | 分到 oc | wall | compute | peripheral |
|--------|---------|------|---------|-----------|
| worker-1 (metal) | 0:116 | 27087 ms | **21972 ms** | 20 ms |
| worker-2 (metal) | 116:232 | 27073 ms | **22137 ms** | 21 ms |
| worker-3 (cpu) | 232:256 | 26281 ms | 7727 ms | 16543 ms |

两个 Metal worker 在：

- **compute 时间差 0.75%**（21972 vs 22137 ms）
- **finish 时间差 87 ms**（artifact publish 分别在 23:14:29.924 和 23:14:30.011）
- 不是 serial（若 serial 应当差 22 秒），也不是 2× parallel（若真并行各自应是 ~11 秒）

这是 Apple Silicon integrated GPU 在 **EU 级时间片轮转** 两个 Metal 进程命令队列的典型症状：两人同时推进，但每人只拿到约一半的 GPU 执行单元时间，因此各自感受到的 kernel 延迟正好翻倍。

### 2. GPU 并发吞吐不翻倍的根因

从 benchmark 源码侧可以直接印证：

- `gemv_metal_runner.mm` 每 chunk 都 `commit` + `waitUntilCompleted`（[L243-249](../compute_node/compute_methods/gemv/metal/gemv_metal_runner.mm#L243-L249)），chunk 间 GPU 会 idle。GFLOPS 计算时优先采用 `GPUStartTime`/`GPUEndTime` 硬件时间戳（[L348-352](../compute_node/compute_methods/gemv/metal/gemv_metal_runner.mm#L348-L352)），因此 80 GFLOPS 是纯 GPU kernel 时间。M1 Pro 内存带宽 ~200 GB/s，gemv 的算术强度 0.5 flop/byte → 带宽上限约 100 GFLOPS，**当前 80 GFLOPS 已经吃到带宽的 80%**，几乎无并发加速空间。
- `conv2d_metal_runner.mm` 用 MPSGraph 的同步 API `runWithMTLCommandQueue`（[L491-502](../compute_node/compute_methods/conv2d/metal/conv2d_metal_runner.mm#L491-L502)），计时用 `steady_clock` 包住整次 graph run，包括 CPU 侧 dispatch。955 GFLOPS ≈ 5.2 TFLOPS 理论峰值的 18%，对 MPSGraph 3×3 conv 这个档位属于合理的实测峰值。

**两个进程同构 workload × 2 不能互相填补 GPU idle**：因为两个 worker 的 chunk 节拍是同步的，会同时忙、同时在 CPU 侧 prep 下一个 chunk，GPU 的空闲窗口也是同时出现。真正能获得并发增益的场景是异构 workload（例如 gemv + conv2d 混跑）。

### 3. 调度器按自报 GFLOPS 分片，未考虑"同 backend 多实例共享硬件"

conv2d large 的切片比例是 116 : 116 : 24 = 45% : 45% : 9.4%，正好匹配 self-reported 955 : 955 : 203 GFLOPS 的比例。调度器把两个 Metal worker 当成两块独立的 955 GFLOPS 设备处理，导致 Metal 侧 compute 被打满 22 秒，而 CPU worker 7.7 秒就做完了自己那 24 channels，剩下 17 秒在 peripheral 里等 —— **整批任务的 wall time 由 Metal 这侧拖尾决定**。

这是**预期的局限**而非 bug：benchmark 是在"独占 backend"语境下测的 GFLOPS，调度器也按这个值决策。如果今后要让调度器感知共享硬件，一种做法是 main_node 在 REGISTER_WORKER 时按 `(hostname, backend)` 分组，组内把每实例的有效 GFLOPS 除以组大小。

### 4. HEARTBEAT 警告只是边缘 case，立即自恢复

[main-20260422-221819.txt:6728](../logs/main-20260422-221819.txt) 有唯一一条 WARNING：

```
23:14:29,901 HEARTBEAT failure for compute node-peer-metal at 192.168.1.136:59286
    attempt=1/4 after timed out waiting for heartbeat ack for 1776924868893
    active_task=conv2d-12
```

时刻精确落在：worker-1 正在 publish 1.9 GB artifact（23:14:29.924）+ main_node 准备 fetch 同一份 artifact（23:14:33.034）+ worker-2/worker-3 同时进入后处理阶段。四个进程 + 一次 4 GB 数据面传输把 CPU 和 IO 同时打满，heartbeat 线程被调度饿死 ~1 秒。

次轮 attempt（23:14:29.902）直接成功，没有升级到节点摘除。说明 1 秒的 heartbeat_ack 超时对"all-in-one 单机 + 大 artifact 高峰"略紧，但系统的重试机制处理得很干净。

### 5. iter=1 比 iter=100 慢是冷 page cache

| task | iter | 总 elapsed | worker-1/-2 compute | worker-1/-2 peripheral |
|------|------|-----------|---------------------|----------------------|
| gemv-6 | 1 | 4986 ms | 5–6 ms | **4806 ms** |
| gemv-7 | 100 | **4011 ms** ↓ | 889–912 ms | 2793 ms |
| gemv-8 | 1000 | 15100 ms | 9932–10284 ms | 4221 ms |

large gemv 矩阵 ~2 GB（16384×32768 f32），每个 worker 分到约 800 MB，首次请求是从磁盘冷读到 shared-memory MTLBuffer。后续请求命中 OS page cache，peripheral 降下来，compute 成为主导。**含义**：单次 iter=1 的基准数据具有误导性，性能回归对比至少要预热或取 iter ≥ 100。

### 6. worker-3 (CPU) 的 peripheral 吞掉大量时间

conv2d-13 stats-only（没 artifact 要上传）的 worker-3：`wall=25620ms, compute=7307ms, peripheral=17461ms`。peripheral 并非上传耗时，而是 CPU 路径的 sum / sum² / reshape / memcpy 在被 Metal 进程挤占的 CPU 上爬行；Metal 路径把这些后处理放在 GPU shader 里所以 peripheral 只有 20 ms。

这符合"同机异构 backend 抢 CPU"的预期。如果要诊断分布更清晰，后续可以把 `peripheral` 拆成 `artifact_stage / output_copy / stats_reduce` 三段。

## 结论

当前 run 的所有行为都能用"单机共享 GPU + 共享 CPU"的物理约束解释，没有发现需要修复的 bug：

- ✓ 两个 Metal worker 公平分走 GPU 时间（差 0.75%）
- ✓ 同步完成（差 87 ms）、无饿死
- ✓ heartbeat 在峰值 IO 下触发一次边缘 case 并立即恢复
- ✓ 数据面跑在 loopback，带宽内存级
- ✓ 8 次任务全部 `status=200`

如果后续要让这个拓扑的吞吐逼近"两块独立 GPU"，唯一真正能榨出增量的方向是**让两个 Metal worker 的 chunk 节拍错开**（不同 `output_channel_batch` 或引入微小起步抖动），让 GPU 的短 idle 窗口被对端 kernel 填补。在同构 workload × 2 的情况下，上限大概也就 5–10% aggregate gain，不值得为此牺牲代码简洁性。

## 潜在改进（非阻塞）

按优先级排序，都不影响当前设计目标：

1. **调度器感知同 backend 多实例**：注册时按 `(host, backend)` 分组，组内 GFLOPS 除以组大小。能避免"把单块 GPU 当两块派发"。
2. **放宽 heartbeat_ack 超时到 2–3 秒** 或在 artifact stage 路径里显式 `sched_yield`，消除 4 GB 峰值 IO 下的边缘 warning。
3. **拆 peripheral 细项**（stage / copy / reduce），精确看到 CPU 路径哪一块是瓶颈。
4. **benchmark 增加 warm-up + 多次采样**，让 iter=1 的首轮延迟不再主导用户看到的单次延迟。

## 已知平台差异：macOS peer 进程无可视窗口

`app/supervisor.py:215-242` 的 `_peer_popen_kwargs` 中，Windows 分支通过 `creationflags = CREATE_NEW_CONSOLE` 让 peer compute-node 子进程拥有独立的可视控制台窗口（docstring 明确说这是给 demo 看的）；POSIX 分支只返回空 kwargs，peer 直接作为后台子进程继承父进程 IO。结果：

- **功能无影响** —— peer 的 TRACE/INFO 仍写入各自的 `worker-*.txt`、`bootstrap-*.txt`，诊断信息不会丢
- **视觉差异** —— macOS 下用户无法直观看到"第二个 compute node 正在运行"，只有日志文件里能看到

要对齐 Windows 的 demo 观感，POSIX 分支需要走 `osascript -e 'tell app "Terminal" to do script ...'` 开新窗口，但代价：

- Terminal.app 会成为新进程组 parent，supervisor 对 `_peer_process` 的 PID 追踪和 `peer_watcher` / `peer_heartbeat_watcher` 的 cleanup 路径都要重写
- 不是所有 macOS 用户装了 Terminal.app（iTerm、Warp 等），需要先做应用探测
- 干净关闭 peer 时需要同时终结壳 shell 和 Python 进程，信号链更复杂

因此目前**有意保留这个差异**：demo 场景下用 `tail -f logs/worker-*.txt logs/main-*.txt` 或分别手开三个 Terminal 手动跑 `bootstrap.py --role ... --backend ...` 作为 workaround，功能完全等价，视觉上也满足"看到三个进程都活着"的需求。
