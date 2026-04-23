# 压缩方案实验报告

## 1. 目标

这轮实验的目标，是在真实部署约束下，为 `Client/download` 中最大的矩阵文件寻找最佳压缩方案。

约束如下：

- 数据是原始 `FP32` 数字文件，用来存矩阵
- 压缩必须在 `30` 秒内完成，否则判定为不可用
- 发送端可以有 `GPU`
- 接收端只能保证有 `CPU`
- 需要分别考虑两种需求：
  - `严格无损`：解压后 `checksum` 必须一致
  - `极低误差`：允许极小数值误差，以换取更高压缩率

## 2. 数据与环境

- 工作目录：`C:\Users\dongm\ee451\Client\download`
- 实验文件：`conv-2-large.bin`
- 文件大小：`4294967296` bytes，约 `4.00 GiB`
- 推断形状：`32768 x 32768` 的 `float32` 矩阵
- 源文件 SHA256：`d29877d9d75a4a811d3916d3326ba0773ae8ada279b1d52c552369e522ce0213`
- 压缩端 GPU：`NVIDIA GeForce RTX 4060 Laptop GPU`
- 驱动版本：`591.86`
- CUDA：`13.2`

### 2.1 软件与依赖版本

本次实验的 Python 依赖全部安装在 `Client/download/.vendor`，避免污染其他环境。下面把这次实验实际用到、或为了让实验在这台机器上可运行而安装的包和版本完整记录下来。

直接参与实验脚本或压缩流程的核心软件如下：

| 类别 | 名称 | 版本 | 说明 |
| --- | --- | --- | --- |
| Python 运行时 | `python` | `3.14.3` | 本次所有脚本执行环境 |
| 数组与数值计算 | `numpy` | `2.4.4` | 矩阵读取、误差计算、流式分块 |
| 无损矩阵压缩 | `blosc2` | `4.1.2` | `zstd/lz4 + shuffle/bitshuffle` 测试 |
| 科学浮点压缩 | `zfpy` | `1.0.1` | `ZFP` Python 绑定 |
| 科学浮点压缩 | `fpzip` | `1.2.5` | `fpzip` Python 包 |
| 科学浮点压缩 | `pysz` | `1.0.3` | `SZ3` Python 绑定，本地源码构建 |
| 通用压缩后端 | `zstandard` | `0.25.0` | `zstd` 后端库 |
| 通用压缩后端 | `lz4` | `4.4.5` | `lz4` 后端库 |
| 编码器集合 | `numcodecs` | `0.16.5` | `blosc2` 相关依赖 |

为了在当前 Windows + Python `3.14` 环境下成功构建或运行这些实验，还安装了这些辅助包：

| 类别 | 名称 | 版本 | 说明 |
| --- | --- | --- | --- |
| 本地构建 | `cmake` | `4.3.1` | 构建 `pysz` 时使用 |
| 本地构建 | `Cython` | `3.2.4` | 构建 `pysz` 时使用 |
| 本地构建 | `ninja` | `1.13.0` | 构建依赖 |
| 数值依赖 | `numexpr` | `2.14.1` | `blosc2` 依赖 |
| 编码依赖 | `msgpack` | `1.1.2` | `blosc2` 依赖 |
| 索引依赖 | `ndindex` | `1.10.1` | `blosc2` 依赖 |
| 网络依赖 | `requests` | `2.33.1` | 安装与依赖解析时引入 |
| 网络依赖 | `urllib3` | `2.6.3` | `requests` 依赖 |
| 网络依赖 | `certifi` | `2026.2.25` | `requests` 依赖 |
| 网络依赖 | `charset_normalizer` | `3.4.7` | `requests` 依赖 |
| 网络依赖 | `idna` | `3.11` | `requests` 依赖 |
| 类型兼容 | `typing_extensions` | `4.15.0` | 兼容依赖 |

构建与来源说明：

- `zfpy`：直接通过 PyPI 安装
- `fpzip`：通过 `pip` 在本机编译安装
- `pysz`：没有现成 wheel，使用本地源码包 `Client/download/_pkgsrc/pysz-1.0.3.tar.gz`，借助 `cmake` 和 `Cython` 在本机编译后放入 `.vendor`
- GPU 仅用于环境可用性确认，本轮最终推荐方案的可部署性结论以“接收端只有 CPU”为前提

## 3. 实验范围

这份结论不是只来自一轮测试，而是把这个目录里前面几轮实验汇总之后得到的。

已经测试过的类别包括：

- 通用无损压缩：
  - `gzip`
  - `bz2`
  - `lzma`
- 面向二进制数值数组的无损压缩：
  - `blosc2 + zstd/lz4 + shuffle/bitshuffle`
- 科学浮点压缩：
  - `zfp / zfpy`
  - `fpzip`
  - `SZ3`
- 结构近似压缩：
  - `SVD`
  - `H-matrix / HODLR`
- 最后一轮“可部署性基准”：
  - 只保留`接收端可用 CPU 解压`的候选方案
  - 同时测量`压缩时间`、`CPU 解压时间`、`精确一致性`或`误差`

### 3.1 测试方法与判定规则

为了让结果可复现，这里把各轮实验实际采用的方法统一写清楚。

通用规则：

- 实验工作目录固定在 `Client/download`
- 在同一轮候选文件中，选择“最大的 `.bin` 文件”；如果有多个并列最大文件，则按文件名字典序选择第一个
- 对于 `FP32` 文件，如果 `文件字节数 / 4` 是完全平方数，则推断为二维方阵；本次主样本被识别为 `32768 x 32768`
- 用户定义的可用性规则是：`30 秒内不能完成压缩，就判不可用`
- 无损方案优先用整文件 `SHA256` 校验“压缩后再解压是否完全一致”
- 有损方案使用全矩阵 `max absolute error` 和 `RMSE` 作为误差指标

各轮实验方法：

1. 通用压缩基线
- 算法：`gzip`、`bz2`、`lzma`
- 参数：`gzip compresslevel=6`，`bz2 compresslevel=6`，`lzma preset=6`
- 实现方式：流式读写，块大小 `8 MiB`
- 目的：给出“传统通用压缩器”在这类 `FP32` 矩阵文件上的基准

2. 矩阵/数值数组无损压缩
- 算法：`blosc2 + lz4/zstd + shuffle/bitshuffle`
- 实现方式：`64 MiB` 块，`typesize=4`，最多 `16` 线程
- 测试组合：
  - `blosc2_lz4_shuffle_c5`
  - `blosc2_lz4_bitshuffle_c5`
  - `blosc2_zstd_shuffle_c1`
  - `blosc2_zstd_shuffle_c3`
  - `blosc2_zstd_shuffle_c5`
  - `blosc2_zstd_bitshuffle_c3`
- 目的：验证对 `FP32` 矩阵应用 `shuffle/bitshuffle` 是否能显著改善无损压缩表现

3. 科学浮点压缩
- 算法：`ZFP(zfpy)`、`fpzip`、`SZ3(pysz)`
- 实现方式：按 `1024` 行做分块压缩，避免一次性复制整个矩阵
- 测试参数：
  - `zfpy_lossless`
  - `zfpy_tolerance_1e-3`
  - `zfpy_rate_16`
  - `fpzip_lossless`
  - `fpzip_precision_24`
  - `sz3_abs_1e-3_interp_lorenzo`
  - `sz3_abs_1e-3_nopred`
- 超时策略：若在 `30s` 内未完成，则保留已处理数据的压缩比例并给出整文件投影值
- 目的：比较专门面向科学浮点数组的压缩器在速度、压缩率、误差上的表现

4. 结构近似压缩
- 方法：`randomized SVD` 与 `HODLR / H-matrix-style` 近似
- `SVD` 测试秩：`8`、`32`、`128`、`256`
- `HODLR` 测试配置：`rank=32`，`leaf_size=16384/8192/4096`
- 评价指标：`relative Frobenius error`、捕获能量比例
- 目的：判断矩阵是否存在足够强的低秩或层次低秩结构

5. 最终可部署性基准
- 假设：发送端可以灵活，接收端只能保证 `CPU`
- 重点指标：`压缩时间`、`CPU 解压时间`、`压缩率`、`SHA256 是否一致`、`误差`
- 最终候选：
  - 无损：`blosc2_zstd_shuffle_c1`、`blosc2_lz4_bitshuffle_c5`、`zfpy_lossless`、`fpzip_lossless`
  - 低误差：`zfpy_tolerance_1e-4`、`zfpy_tolerance_5e-4`、`zfpy_tolerance_1e-3`、`zfpy_precision_23`、`zfpy_rate_16`、`fpzip_precision_28`
- 目的：给出真正适合上线部署的候选，而不只是在单一指标上最优的算法

6. 网络传输 speedup 评估
- 使用的端到端模型：
  - `总时间 = 压缩 + 发送前对传输字节流做 checksum + 网络传输 + 接收后对传输字节流做 checksum + CPU 解压`
- `SHA256` 吞吐通过本机顺序读取 `1 GiB` 同类二进制文件实测，约为 `1354 MiB/s`
- 分析链路重点放在：
  - `300 Mbps WiFi`
  - `1 Gbps Ethernet`
- 目的：判断压缩对“真正传得更快”有没有意义，而不是只看节省了多少字节

## 4. 关键结论

### 4.1 最佳无损方案

如果要求“解压后 checksum 一致”，目前最佳方案是：

`blosc2_zstd_shuffle_c1`

实测结果：

- 压缩时间：`6.64s`
- CPU 解压时间：`4.84s`
- 空间节省：`15.52%`
- 解压后 SHA256：与原文件完全一致

它是当前最实用的无损方案，原因很明确：

- 明显满足 `30s` 约束
- 压缩率比通用无损压缩器更好
- 解压速度也很快
- 已经做过整文件 `SHA256` 校验，不是抽样验证

无损方案对比：

| 方法 | 压缩 | CPU 解压 | 节省空间 | 是否精确一致 |
| --- | ---: | ---: | ---: | --- |
| `blosc2_zstd_shuffle_c1` | `6.64s` | `4.84s` | `15.52%` | 是 |
| `blosc2_lz4_bitshuffle_c5` | `6.57s` | `4.70s` | `9.37%` | 是 |
| `zfpy_lossless` | `28.02s` | `29.96s` | `0.61%` | 是 |
| `fpzip_lossless` | `30.13s` | n/a | `~5.42%` | 超时 |
| `gzip` | `135.50s` | 未纳入最终 CPU 解压轮 | `7.35%` | 是 |
| `bz2` | `376.71s` | 未纳入最终 CPU 解压轮 | `5.01%` | 是 |

### 4.2 最佳极低误差方案

如果允许“极低误差”，当前最稳妥的方案是：

`zfpy_tolerance_1e-4`

实测结果：

- 压缩时间：`23.33s`
- CPU 解压时间：`24.38s`
- 空间节省：`25.42%`
- 最大绝对误差：`2.861e-05`
- `RMSE`：`5.268e-06`

这组的特点是：误差非常小，而且仍然能在 `30s` 内完成压缩。

极低误差候选对比：

| 方法 | 压缩 | CPU 解压 | 节省空间 | 最大绝对误差 | RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `zfpy_tolerance_1e-4` | `23.33s` | `24.38s` | `25.42%` | `2.861e-05` | `5.268e-06` |
| `zfpy_tolerance_5e-4` | `21.72s` | `22.21s` | `34.79%` | `2.356e-04` | `4.198e-05` |
| `zfpy_tolerance_1e-3` | `21.18s` | `21.46s` | `37.92%` | `4.616e-04` | `8.395e-05` |
| `zfpy_precision_23` | `22.01s` | `22.54s` | `34.33%` | `4.482e-04` | `4.056e-05` |
| `zfpy_rate_16` | `19.22s` | `18.88s` | `50.00%` | `2.037e-02` | `1.401e-03` |

这里的理解可以分成三档：

- 如果你最在意“误差尽可能小”，选 `zfpy_tolerance_1e-4`
- 如果你想在“误差仍很小”和“节省更多空间”之间折中，选 `zfpy_tolerance_5e-4`
- `zfpy_rate_16` 压得更狠，但误差已经明显不是“极低误差”这一档了

### 4.3 不适合当前场景的方法

`fpzip`

- `fpzip_lossless` 没过 `30s`
- `fpzip_precision_28` 误差表现其实不差，但仍然超时

`SZ3`

- 之前的实验里，`SZ3` 在压缩率上有竞争力
- 但在这台机器和当前入口下，压缩时间过不了 `30s`

`SVD` 和 `H-matrix`

- 都能很快完成
- 但这份矩阵并不低秩，或者说不适合这类结构近似
- 压缩率是靠引入很大重建误差换来的，不适合当前需求

`gzip / bz2 / lzma`

- 可以无损
- 但速度太慢，不满足你的时间门槛

### 4.4 从网络传输角度看，压缩是否有意义

如果目标不是“节省空间”，而是“缩短端到端传输时间”，那判断标准应该是 `speedup`，而不是单看压缩率。

这轮报告采用的端到端模型是：

`总时间 = 压缩 + 发送前对传输字节流做 checksum + 网络传输 + 接收后对传输字节流做 checksum + CPU 解压`

这里专门把 `checksum` 也算进去，是因为压缩后传输的字节更少，传输字节流的校验时间也会下降。

本机实测 `SHA256` 吞吐约为 `1354 MiB/s`，因此：

- 原始 `4 GiB` 文件做一次 `SHA256` 约 `3.03s`
- `blosc2_zstd_shuffle_c1` 压缩包做一次 `SHA256` 约 `2.56s`
- `zfpy_tolerance_5e-4` 压缩包做一次 `SHA256` 约 `1.97s`

也就是说，压缩确实能省掉一部分 `checksum` 时间，但量级并不大，通常只有几百毫秒到两秒左右，不足以扭转整体趋势。

在这个模型下，典型链路上的端到端 `speedup` 如下：

| 方法 | 300 Mbps WiFi | 1 Gbps Ethernet |
| --- | ---: | ---: |
| `blosc2_zstd_shuffle_c1` | `1.064x` | `0.886x` |
| `zfpy_tolerance_1e-4` | `0.876x` | `0.519x` |
| `zfpy_tolerance_5e-4` | `0.984x` | `0.575x` |
| `zfpy_tolerance_1e-3` | `1.026x` | `0.597x` |

这里的解释很关键：

- `300 Mbps WiFi` 下，只有 `blosc2_zstd_shuffle_c1` 有比较明确的正收益，但也只有约 `6.4%` 加速
- 同样在 `300 Mbps` 下，有损 `zfp` 方案即使压缩率更高，算上压缩和解压后，收益仍然很小，甚至可能负加速
- 到了 `1 Gbps Ethernet`，这些方案全部是负加速

从盈亏点来看：

- `blosc2_zstd_shuffle_c1` 的盈亏点约为 `506 Mbps`
- `zfpy_tolerance_5e-4` 的盈亏点约为 `286 Mbps`
- `zfpy_tolerance_1e-3` 的盈亏点约为 `323 Mbps`

这说明：

- 如果实际有效带宽低于几百 `Mbps`，压缩可能对端到端时间有帮助
- 如果已经接近 `1 Gbps`，压缩基本没有传输加速意义

还有一个更严格的补充：

- 上面的模型假设 `checksum` 是针对“传输字节流”来做
- 如果系统真正需要的是“对最终解压后的原始矩阵再做 checksum”
- 那么这部分时间几乎不会随着压缩而减少，因为最终还是要对同样大小的原始数据做校验

所以从网络传输角度，这份数据上的结论是：

- `WiFi 300 Mbps`：压缩只有轻微意义，而且主要只有 `blosc2` 无损方案还算说得过去
- `Ethernet 1 Gbps`：为了提速而压缩，基本不值得做
- 如果真正目标是提速，通常更值得优先考虑的是换数据表示，例如 `FP16/BF16`、量化、按块传输、只传增量等方法

## 5. 为什么最终没有把 GPU-only 方案当作首选

这台机器确实有 NVIDIA GPU，但真正决定方案选择的不是“能不能用 GPU 压”，而是“对方只有 CPU 能不能稳妥解压”。

这会直接改变最优解：

- GPU 专用格式不一定适合部署
- 一个 `CPU` 端可稳定解压、而且压缩已经只要 `6` 到 `23` 秒的方案，通常比一个更快但兼容性更差的 GPU 路线更实用

我也对官方项目说明做了核对：

- `cuSZp` 官方说明明确是面向 GPU 的压缩/解压框架，而且环境要求里直接写了 `Linux OS with NVIDIA GPUs`
- `nvCOMP` 官方仓库定位也是 GPU 压缩/解压库，虽然它提供了一些 CPU 互操作示例
- `MGARD` 同时有 CPU 和 GPU 分支，但它主要是科学数据有损压缩框架，不是当前“默认先求精确一致”的最自然首选

所以在你这个项目里，GPU 不是没有价值，而是：

- 它可以作为以后进一步优化压缩端吞吐的方向
- 但它并不是当前“最佳可部署方案”的决定因素

## 6. 最终推荐

### 6.1 默认生产推荐

如果你现在要一个单一默认方案，我建议用：

`blosc2 + zstd + shuffle`

对应本轮最佳配置：

- `blosc2_zstd_shuffle_c1`

推荐理由：

- 无损
- 速度快
- 压缩率是当前可用无损方案里最好的
- CPU 解压快
- 已经做过整文件一致性验证

### 6.2 如果允许极低误差

如果你愿意接受极低误差来换更高压缩率，我建议优先看：

- 最保守方案：`zfpy_tolerance_1e-4`
- 更平衡方案：`zfpy_tolerance_5e-4`

### 6.3 一个简单的决策规则

- 如果下游必须 `checksum` 一致：选 `blosc2_zstd_shuffle_c1`
- 如果下游允许极小数值误差，并希望多省一些空间：选 `zfpy_tolerance_1e-4` 或 `zfpy_tolerance_5e-4`
- 不建议把 `zfpy_rate_16` 当作“低误差方案”
- 如果目标是“网络传得更快”：只有在有效带宽低于约 `500 Mbps` 时，才值得认真考虑启用压缩

## 7. 相关输出文件

本报告归档到 `SuperWeb-Cluster/docs` 后，实验脚本、原始结果和中间记录仍保留在 `Client/download`。从 `docs` 目录看，相对路径如下：

- `../../Client/download/code/practical_cpu_decode_benchmark.py`
- `../../Client/download/artifacts/practical_cpu_decode_results.json`
- `../../Client/download/artifacts/practical_cpu_decode_summary.md`
- `../../Client/download/code/compression_benchmark.py`
- `../../Client/download/artifacts/compression_benchmark_30s_summary.md`
- `../../Client/download/code/matrix_compression_benchmark.py`
- `../../Client/download/artifacts/matrix_compression_30s_results.json`
- `../../Client/download/artifacts/matrix_compression_30s_summary.md`
- `../../Client/download/code/scientific_float_compression_benchmark.py`
- `../../Client/download/artifacts/scientific_float_compression_30s_results.json`
- `../../Client/download/artifacts/scientific_float_compression_30s_summary.md`
- `../../Client/download/artifacts/scientific_float_compression_notes.md`
- `../../Client/download/code/structural_matrix_compression_benchmark.py`
- `../../Client/download/artifacts/structural_matrix_compression_30s_results.json`
- `../../Client/download/artifacts/structural_matrix_compression_30s_summary.md`

## 8. 参考资料

本轮结论中，涉及算法能力和可部署性判断时，参考了这些官方项目主页：

- ZFP: https://github.com/LLNL/zfp
- fpzip: https://github.com/LLNL/fpzip
- Python-Blosc2: https://github.com/Blosc/python-blosc2
- SZ3: https://github.com/szcompressor/SZ3
- cuSZp: https://github.com/szcompressor/cuSZp
- nvCOMP: https://github.com/NVIDIA/nvcomp
- MGARD: https://github.com/CODARcode/MGARD
- Turbo-Transpose: https://github.com/powturbo/Turbo-Transpose
