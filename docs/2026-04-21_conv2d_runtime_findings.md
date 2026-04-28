# Conv2d 运行时调试纪要 — 2026-04-21

这一轮针对 `compute_node/compute_methods/conv2d` 的调试，集中修了四个问题。每个都按「现象 / 根因 / 修复 / 验证」记录，方便后续复盘。

## 1. Native runner stderr 在 TASK_FAIL 日志中丢失

### 现象
CUDA runner 崩溃时，主进程日志里只能看到 `subprocess.CalledProcessError: Command '[...]' returned non-zero exit status 1.`，看不到底层 CUDA 报错信息。上一次调试甚至因为集群整体退出，连 stderr 原文都来不及留在屏幕上。

### 根因
两层叠加：

1. Conv2d 任务执行在 ProcessPoolExecutor 的子进程里。子进程默认没有配置任何 logging handler，`_LOGGER.error(...)` 在那里是 **死路** — 写进去也不会出现在主进程日志里。
2. `subprocess.CalledProcessError.__str__()` 只包含 command 和 exit code，不包含 `stderr`。即使把它原样 raise 出去、通过 pickle 传回主进程，stderr 字段能穿过来，但格式化后的 TASK_FAIL 消息里看不到。

### 修复
`executor.py` 新增 `RunnerProcessError(RuntimeError)` 异常类型，构造时把 stderr 尾段直接嵌进 message 字符串。`_run_native_runner` 捕获 `CalledProcessError` / `TimeoutExpired` 后重新 raise 成 `RunnerProcessError`，message 里包含：

- command
- exit code
- stderr 尾部 (`_tail_stream`, 2KB 上限)
- stdout 尾部

### 验证
新增回归测试 `test_runner_process_error_carries_stderr_through_pickle`，确认 pickle round-trip 后 stderr 仍然在 `str(exc)` 里。之后复现 CUDA 716 时，主日志 TASK_FAIL 行里可直接看到 `stderr_tail='CUDA error at ... code=716 "misaligned address"'`。

### 教训
**把诊断信息打包进异常的 str 里，不要依赖子进程的 logger。** 这条已写入 auto-memory `project_conv2d_runs_in_subprocess.md`。

---

## 2. CUDA `cudaErrorMisalignedAddress` (716)

### 现象
`c_out % 4 != 0` 的任务片会稳定触发 CUDA error 716。例如 `c_out=256` 切成 3 份，worker-1 拿到 `oc=0..121`、worker-2 拿到 `oc=121..242`，中间的 worker-2 起始 offset `121` 不是 4 的倍数，runner 崩在内核里。

### 根因
`accumulate_output_lanes` / `store_output_lanes` 中使用 `float4` 一次加载/写入 4 个 lane。`float4` 要求指针按 16B 对齐。权重张量布局是 `[kh][kw][ic][oc]`，最内层 stride 是 `c_out`。只有当 `c_out` 以及任务起始 `oc` 都整除 4，float4 的起始地址才必然对齐；否则 runner 会触发 716。

### 修复
内核在进入 `float4` 分支前加一道运行时对齐检查：

```cpp
if (valid_outputs == kOutputsPerThread &&
    (reinterpret_cast<uintptr_t>(weight_ptr) & 0xF) == 0) {
    // float4 快路径
} else {
    // 标量回退
}
```

对 `output_ptr` 做同样处理。这样不限制上层切分策略，runner 自己吸收 stride 不整除 4 的情况。

### 验证
重新跑 `c_out=256 / 3 worker` 划分，不再出现 716；runner 正常退出 0。

---

## 3. CUDA worker 比 CPU 慢 3.6–7.6× — autotune 每任务都跑

### 现象
第一次跑 3-worker 混合集群（2 CUDA + 1 CPU）时，CUDA 节点每通道 **比 CPU 还慢 3–7 倍**。直觉应该反过来。

### 根因
对比 Metal 和 CUDA 两个后端在 `executor.py` 里的命令行构造：Metal 分支从 `best_config` 里把 `block_size` / `tile_size` 传给 runner，CUDA 分支没有。

CUDA runner 在没有显式 `--block-sizes` / `--tile-sizes` 时会走内置 autotune —— 14 个 tile 候选，每个候选跑 1 warmup + 1 measurement，加起来是 **每个任务 28 次完整前向**。基准扫表早就跑过一次（结果在 `result.json` 里），但运行时没吃这个缓存，每个下发任务从头扫一遍。

### 修复
CUDA 分支也从 `best_config` 读 `block_size` / `tile_size`，有值就通过 `--block-sizes` / `--tile-sizes` pin 死给 runner：

```python
elif backend_name == "cuda":
    ...
    cuda_block_size = int(best_config.get("block_size") or 0)
    cuda_tile_size = int(best_config.get("tile_size") or 0)
    if cuda_block_size > 0 and cuda_tile_size > 0:
        cmd.extend([
            "--block-sizes", str(cuda_block_size),
            "--tile-sizes", str(cuda_tile_size),
        ])
```

`test_task_executor.py` 新增了正向 / 反向断言测试。

### 验证
大任务上每通道 compute 时间：**2918ms → 506ms**（≈5.8× 加速）。CUDA 单通道变成比 CPU 快 1.6–1.9×，和基准扫表得到的相对性能一致。

### 教训
**运行时应该吃基准的结果，而不是每次重新扫表。** 两个后端分支需要对齐命令行构造习惯，否则任何一个后端默默走 autotune 都很难发现。

---

## 4. 大任务 peripheral 时间异常 — 49 秒纯 Python 遍历

### 现象
修完 autotune 之后，`logs/main-20260421-231411.txt` 里的 `conv2d-3` 任务仍然有怪异开销：

```
worker-3 (oc=242:256, 14 ch, CPU)  wall:20780ms  compute:13521ms  peripheral:7032ms
worker-1 (oc=0:121,  121 ch, CUDA) wall:110821ms compute:61180ms  peripheral:48800ms
worker-2 (oc=121:242,121 ch, CUDA) wall:111833ms compute:61665ms  peripheral:49121ms
```

CUDA 节点 peripheral **48-49 秒**，CPU 节点只有 7 秒。compute 已经被修快了，peripheral 反而成了 wall-time 的主要来源。

### 根因
`stats_only` 模式下，每个 worker 跑完 runner 之后要调 `_summarize_conv2d_slice_file` 扫一遍输出文件算 `sum` / `sum_sq` / samples。原实现是：

```python
for x in values:           # array.array('f'), 纯 Python 迭代器
    xf = float(x)
    sum_v += xf
    sum_sq += xf * xf
    ...
```

CPython 解释器循环处理浮点的速度大约每秒 2-3M element：
- CPU worker 输出 14×H×W ≈ 1.4 亿 float / 14 = **小 14 倍** → 7 秒；
- CUDA worker 输出 121×H×W ≈ 1.2 亿 float → 49 秒。

比例完全吻合。

### 修复
改成 `numpy` 矢量化，仍保留 1MB 分块流式读取以限制内存：

```python
with path.open("rb") as handle:
    while True:
        chunk = handle.read(1024 * 1024)
        if not chunk:
            break
        values = np.frombuffer(chunk, dtype=np.float32)
        values64 = values.astype(np.float64)
        sum_v += float(values64.sum())
        sum_sq += float(np.dot(values64, values64))
        if remaining > 0:
            take = min(remaining, values.size)
            samples.extend(float(x) for x in values[:take])
            remaining -= take
```

`float64` 累加器保持和原 Python `float` 累加器的精度一致。

### 验证
- 参数一致性测试（100k float）：`sum` bit-exact，`sum_sq` 相对误差 `8e-14`，samples bit-exact。
- 120M float 微基准：**49s → 435ms**（约 113× 加速）。
- `tests/test_task_executor.py` + `tests/test_runner_failure_log.py` 共 18 个测试全通过。

### 教训
**数值数据不要用 Python 循环。** array / list / 解释器每次 float 运算都要经过对象装箱，在 1e8 量级的数据上会把微秒级差异放大到几十秒。下一次看到 peripheral 大于 compute 的十分之一，先怀疑 Python 循环。

---

## 5. Capacity alignment — dispatch 的默认入口由 benchmark 反转为单 pass

### 现象
第 3 节把 `block_size` / `tile_size` pin 死后，CUDA 单通道 compute 已经和基准扫表得到的相对性能一致。但上层 capacity bookkeeping 仍然失真：基准报的 per-channel 时间是 **单次 kernel 的 cudaEvent 时间**（毫秒量级），而 runtime 记录的 `computation_ms_total` 是 **subprocess wall-clock**，后者包含进程 spawn、CUDA context 初始化、D2H、checksum、写文件等常驻开销。CUDA 分到 9× 工作却比 CPU 后完成的异常，就是 subprocess wall 把这些常驻开销折算进了每通道时间。

### 根因
Runner 的**默认入口是 benchmark**：`main()` 进去就是 autotune 14 候选 + measurement 多 pass + 单独的 output/checksum pass。哪怕 executor 已经通过 `--block-sizes`/`--tile-sizes` pin 死了唯一候选，runner 依然要跑多 pass + 切开的 output pass。subprocess wall 天然等于「基准结构 + 进程常驻开销」，capacity 算法拿到的数字系统性偏大，且这个偏大对 CUDA 比对 CPU 更明显（context + D2H）。

### 修复
把 runner 的默认入口反转：**默认 dispatch，`--mode benchmark` 才触发 autotune + measurement**。

- `conv2d_cuda_runner.cu`：新增 `enum class RunnerMode { Dispatch, Benchmark }`、`--mode` / `--shared-input` CLI；dispatch 分支把 kernel launch + sync + D2H + host copy 合在一个 `cudaEventRecord(start)..(stop)` bracket 里，JSON 新增 `mode` 与 `compute_event_ms`。只有 `--mode benchmark` 会走完整的 14 候选 autotune。（代价：dispatch 的 cudaEvent 里含 ~3ms D2H，相比 pure kernel 略虚高，但比 subprocess wall 贴近基准得多。）
- `conv2d_cpu_windows.cpp` / `conv2d_cpu_macos.cpp`：同样的 `--mode` 分支。dispatch 用 `workers[0]`（由 executor 传入 benchmark 选中的值）跑一次 `run_multithreaded`，`compute_event_ms = chrono 单次 compute` 秒转毫秒。
- `conv2d_metal_runner.mm`：`--mode dispatch` 默认只做一次 MPSGraph 提交；`--mode benchmark` 保留 autotune + measurement 双 pass。
- `performance_metrics/conv2d/backends/{cuda,metal}_backend.py` 的 `_run_runner` 都显式注入 `--mode benchmark`，保证基准扫表路径和之前一致。
- `executor.py`：dispatch 路径 pin 住 `--shared-input`（否则 `build_candidate_tiles` 会同时返回 shared=0/1 两个变体，dispatch 挑到的第一个可能与基准选中的不一致）；新增 `_parse_compute_event_ms`，从 runner stdout 的 JSON 解析 `compute_event_ms`，`computation_ms_total += min(subprocess_wall_ms, compute_event_ms)`，subprocess wall 只在回退路径保留。

### 验证
- Unit：306 passed / 8 skipped。新断言：`cuda_backend._run_runner` 命令中含 `--mode benchmark`；executor dispatch 命令含 `--shared-input` 且**不含** `--mode`（走默认 dispatch）。
- Runner 冒烟（`h=w=64, c_in=32, c_out=64, k=3`）：
  - CPU dispatch `compute_event_ms=27.01ms`（worker=1） vs. benchmark 扫 1/2/4 选中 `compute_event_ms=10.47ms`（autotune 9.95ms, 3 candidates）。
  - CUDA dispatch `compute_event_ms=1.22ms`（默认 tile）vs. benchmark 扫 14 候选 `compute_event_ms=0.316ms`（autotune 0.302ms）。
  - 两个后端在 dispatch 和 benchmark 两种 mode 下 JSON 都正确携带 `mode`/`compute_event_ms`/`trials_run` 字段。
- Stage 6c 的 2-worker/3-worker 真集群对比（CUDA vs CPU per-channel 时间是否与基准比例吻合）属于 operator-in-loop 跑，留给下一次上集群时验证。

### 教训
**Capacity signal 必须和基准 signal 在同一个时间域里采样。** benchmark 报 cudaEvent 时间，runtime 就不能用 subprocess wall；否则两者差一个「进程常驻开销 + autotune 扫表 + 分离的 output pass」的系统性 offset，多机调度必然失真。默认入口也要选贴近真实下发路径的那个 —— autotune 作为显式 opt-in 比作为默认值更安全。

---

## 复盘

这五个问题有一个共同模式：**性能或错误信号被某一层默默吞掉了**。

- stderr 被子进程 logger 吞掉 → 包进异常 str
- CUDA 对齐错误被 `CalledProcessError` 吞掉 → 同上
- Autotune 被隐式启用，基准结果被无视 → 对齐两个后端的命令行
- 解释器循环吞掉了 numpy 向量化的机会 → 换 numpy
- Dispatch 把 autotune + 常驻开销当成 compute → runner 默认入口反转，改吃 `compute_event_ms`

下一次遇到「数字和预期相差一个数量级」的情况，优先检查哪一环在默默地做了预期之外的工作。
