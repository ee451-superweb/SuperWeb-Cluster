# 已知问题：native runner 编译逻辑放错层级 — 2026-04-26

本文档记录项目当前架构中**已识别但本次作业周期内未予修复**的一项分层缺陷：每个 compute method 的 native runner 编译逻辑被绑死在 `performance_metrics/` 之下，而不是和源码一起放在 `compute_methods/` 之下。该问题在功能测试中不会暴露（编译路径在两边都能跑通），但在分层依赖、bootstrap CLI 设计与未来扩展上都已经开始造成实际摩擦。

## 1. 现象

1. **源码与编译器分居两地**：以 GEMV 为例，C++/CUDA/HLSL/Metal 源码以及构建产物在 [compute_methods/gemv/{cpu,cuda,dx12,metal}/](../compute_node/compute_methods/gemv/)，而真正驱动编译的 `_compile_if_needed` / `_compile_runner` 函数却定义在 [performance_metrics/gemv/backends/](../compute_node/performance_metrics/gemv/backends/)。两边通过 `from compute_node.compute_methods.gemv import CUDA_SOURCE_PATH, CUDA_BUILD_DIR, ...` 来回拼路径。
2. **runtime 反向依赖 performance_metrics**：真正跑任务的代码绕回 benchmark 模块去借编译能力——
   - [compute_node/task_executor.py:728](../compute_node/task_executor.py#L728)：`MetalBackend()._compile_if_needed(force_rebuild=False)`
   - [compute_node/compute_methods/conv2d/executor.py:359](../compute_node/compute_methods/conv2d/executor.py#L359)：同上
   依赖方向变成 **runtime → performance_metrics**，与"benchmark 是可拔掉的附属物"这一直觉相反。
3. **bootstrap 无法表达"只编译"**：bootstrap.py 的 `--rebuild` 想表达"只编译、不跑 benchmark"时无处下手——编译入口被锁在 `*_backend.py` 的 `.run()` 调用链内部，`_compile_if_needed` 之后立即进入测量循环，没有早退出口。

## 2. 根因

`*_backend.py` 是**两件事缝在同一个文件里**：

| 责任 | 行数级别 | 归属 |
|------|----------|------|
| 编译 native runner（`_compile_if_needed`、`_compile_runner`、stale 判断、toolchain 检测、prebuilt fallback） | ~100 行 | 应属 compute_methods |
| Autotune sweep / benchmark 测量循环 / trial 记录 / JSON 解析 / 评分 | ~500+ 行 | 属 performance_metrics |

GEMM 更极端：[performance_metrics/gemm/benchmark.py](../compute_node/performance_metrics/gemm/benchmark.py) 连 backend 拆分都没做，编译（`_compile_cpu_posix_runner` / `_compile_cpu_windows_runner` / `_compile_posix_runner` / `_compile_windows_runner` / `_ensure_cpu_runner_built` / `_ensure_runner_built`）和 benchmark 测量裸混在同一个文件里。

因为编译逻辑被埋在 benchmark 的内部调用链里，runtime 想要编译产物又不能跑 benchmark，只能直接 `MetalBackend()._compile_if_needed(...)` 这样跨层去戳一个 benchmark 类的私有方法——这就是反向依赖的来源。

## 3. 应有的形态

```
compute_methods/<method>/
  cpu/, cuda/, dx12/, metal/         源码（现有）
  paths.py                           路径常量（现有）
  build.py  或  build/{cpu,...}.py   新：纯编译逻辑，可作为 __main__ 直接调用
  handler.py / executor.py           runtime 分发（现有）

performance_metrics/<method>/backends/
  *_backend.py                       只剩 autotune + 测量，
                                     调 compute_methods.<method>.build 拿 executable_path
```

拆完之后：

- runtime 代码（[task_executor.py](../compute_node/task_executor.py)、[compute_methods/conv2d/executor.py](../compute_node/compute_methods/conv2d/executor.py)）import 的是 `compute_methods.<method>.build`，反向依赖消失。
- bootstrap 想做的 `--rebuild` 只编译、`--retest` 只跑 benchmark、`--regenerate` 只生成数据集这三档正交语义，编译档直接对应 `python -m compute_node.compute_methods.<method>.build --all`，不需要在 benchmark CLI 里塞一个"跑一个不测的 benchmark"的尴尬开关。
- `performance_metrics/` 整个模块退回成"benchmark 测量与上报"——可以独立删除而不影响 runtime。

## 4. 为什么本次未实施

- 改动覆盖 3 个 method（gemv / conv2d / gemm）× 4 个 backend（cpu / cuda / dx12 / metal），且 GEMM 的现有结构与另外两个不一致，需要先拉齐再拆分；同时还要修两处 runtime 跨层 import 与所有相关测试，工程量明显超出本次作业窗口。
- 当前布局**不影响功能正确性**——编译能跑通，benchmark 能跑通，runtime 能跑通；只是分层不干净。
- bootstrap 这一侧的 `--retest` / `--rebuild` / `--regenerate` 三档语义可以**先在 bootstrap 层正交化**而不动下层（`--rebuild` 仍然透传到 benchmark 的 `--rebuild`，副作用是会顺便跑一次 benchmark），代价是"只编译"暂时无法精确表达，但操作员可见行为能接受。

## 5. 相关代码位置

| 关注点 | 文件 | 行 |
|--------|------|---|
| GEMV 编译入口集中地（应迁出） | [performance_metrics/gemv/backends/cuda_backend.py](../compute_node/performance_metrics/gemv/backends/cuda_backend.py) | 667 |
| | [performance_metrics/gemv/backends/cpu_backend.py](../compute_node/performance_metrics/gemv/backends/cpu_backend.py) | 549 |
| | [performance_metrics/gemv/backends/dx12_backend.py](../compute_node/performance_metrics/gemv/backends/dx12_backend.py) | 505 |
| | [performance_metrics/gemv/backends/metal_backend.py](../compute_node/performance_metrics/gemv/backends/metal_backend.py) | 473 |
| Conv2d 编译入口集中地（应迁出） | [performance_metrics/conv2d/backends/cpu_backend.py](../compute_node/performance_metrics/conv2d/backends/cpu_backend.py) | 538 |
| | [performance_metrics/conv2d/backends/dx12_backend.py](../compute_node/performance_metrics/conv2d/backends/dx12_backend.py) | 592 |
| | [performance_metrics/conv2d/backends/metal_backend.py](../compute_node/performance_metrics/conv2d/backends/metal_backend.py) | 508 |
| GEMM 编译与测量未拆分（最严重） | [performance_metrics/gemm/benchmark.py](../compute_node/performance_metrics/gemm/benchmark.py) | 121, 140, 235, 250 |
| Runtime 反向依赖 performance_metrics（GEMV Metal） | [compute_node/task_executor.py](../compute_node/task_executor.py) | 728 |
| Runtime 反向依赖 performance_metrics（Conv2d Metal） | [compute_node/compute_methods/conv2d/executor.py](../compute_node/compute_methods/conv2d/executor.py) | 359 |
| 路径常量已就位、迁移目标 | [compute_methods/gemv/__init__.py](../compute_node/compute_methods/gemv/__init__.py) | 1-52 |

## 6. 后续如要落地的最小步骤建议

1. 在 [compute_methods/gemm/](../compute_node/compute_methods/gemm/) 下新建 `build.py`，把 [performance_metrics/gemm/benchmark.py](../compute_node/performance_metrics/gemm/benchmark.py) 中的 `_compile_*` 与 `_ensure_*_runner_built` 整体搬过来。GEMM 拆分收益最大、暴露的接口最少，先拿它试水。
2. 验证 [performance_metrics/gemm/benchmark.py](../compute_node/performance_metrics/gemm/benchmark.py) 在改 import 后能复用 `compute_methods.gemm.build.ensure_built()` 拿到 executable，benchmark 行为不变；跑一遍 `python -m compute_node.performance_metrics.benchmark --method gemm` 与 `--method gemm --rebuild`。
3. 把 GEMV 的四个 backend 文件中的编译段落抽出去，放到 `compute_methods/gemv/build/{cpu,cuda,dx12,metal}.py`，每个 backend 文件只保留 autotune + 测量。改 [task_executor.py:728](../compute_node/task_executor.py#L728) 的 import。
4. Conv2d 同上，并改 [compute_methods/conv2d/executor.py:359](../compute_node/compute_methods/conv2d/executor.py#L359) 的 import。
5. 在 [bootstrap.py](../bootstrap.py) 把 `--retest` / `--rebuild` / `--regenerate` 三档调整为正交语义：`--regenerate` 调 `compute_node.input_matrix.generate --force`，`--rebuild` 调新的 `compute_methods.<method>.build` 入口，`--retest` 直接调 `performance_metrics.benchmark`（不传 `--rebuild`）。
6. 加一组 import 方向的回归断言：`compute_methods` 的任何模块都不允许 `import compute_node.performance_metrics.*`，可在 CI 用一个简单的 grep 守门即可。
