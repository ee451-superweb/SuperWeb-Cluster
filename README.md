# superweb-cluster

## Status

`superweb-cluster` has completed Sprint 1, Sprint 2, and Sprint 3, and is now
in Sprint 4.

Sprint 1 established the baseline runtime: bootstrap, discovery, registration,
protobuf messaging, heartbeat, and the first end-to-end distributed compute
path.

Sprint 2 hardened that baseline around the first production method, expanded
hardware benchmark coverage, and reshaped the repository so new compute methods
can be added without collapsing everything back into one runtime path.

Sprint 3 turned the cluster into a multi-method platform, brought `conv2d`
fully online end-to-end (including a separate TCP data plane for large
artifacts), and introduced the first WinUI3 and iOS frontends so the cluster is
no longer terminal-only.

Sprint 4 is hardening the production path for release: fault tolerance,
richer per-slice timing diagnostics, multi-mode client response shapes, faster
aggregation, a third compute method (`gemm`), an OS-driven adaptive capacity
signal, and frontend / per-platform-backend work to ship the clients.

## Features

| Capability | Description |
|---|---|
| **Auto-Discovery** | Zero-configuration LAN discovery via mDNS, with `discover` and `announce` bootstrap roles |
| **Method-Aware Benchmarking** | Each node benchmarks supported methods locally and reports per-method GFLOPS summaries during registration |
| **GFLOPS-Aware Scheduling** | The main node tracks benchmark-derived processor inventories and partitions work proportionally to measured throughput |
| **Worker-Level Failover** | When a worker dies mid-slice, its row/output-channel range is re-partitioned across surviving workers proportional to their original throughput; the in-flight client request still completes |
| **Multi-Mode Client Response** | Conv2d clients can opt into stats-only summaries, sampled previews, or full artifact downloads, so very large outputs do not have to be materialized client-side |
| **Per-Slice Timing Diagnostics** | The runtime protocol carries compute, fetch, and peripheral milliseconds per worker slice, so the client response captures where time was actually spent |
| **Memmap Fan-In Aggregation** | Conv2d aggregation streams worker artifacts into the final output via `np.memmap` strided assignment instead of per-pixel Python I/O |
| **Deterministic Input Data** | Shared fixed-seed generators create byte-stable GEMV and conv2d datasets on every machine |
| **Native Runner Stack** | In-tree CPU, CUDA, and Metal runners back both benchmarking and compute-node task execution |
| **Structured Runtime Protocol** | Registration, heartbeat, client requests, task assignment, worker updates, and task results all flow through framed protobuf messages, with large artifacts moved onto a separate TCP data plane |
| **Crash-Survivable Benchmark Tracing** | Benchmark progress is persisted to `result_status.json` and `result_trace.jsonl` for postmortem analysis |

## Architecture

```text
+============================== LAN =============================+
|                                                                |
|   +-----------+                                                |
|   |  Client   |                                                |
|   | (CLI /    |---+                                            |
|   |  WinUI3 / |   |  CLIENT_REQUEST / CLIENT_RESPONSE          |
|   |  iOS)     |   |  + artifact upload / download (data plane) |
|   +-----------+   |                                            |
|                   v                                            |
|                 +-------------------------------+              |
|                 |           Main Node           |              |
|                 |                               |              |
|                 |  discovery      :5353  UDP    |              |
|                 |    mDNS 224.0.0.251           |              |
|                 |  control plane  :52020 TCP    |              |
|                 |    REGISTER / HEARTBEAT       |              |
|                 |    TASK_ASSIGN / TASK_RESULT  |              |
|                 |    CLIENT_REQUEST / RESPONSE  |              |
|                 |  data plane     :52021 TCP    |              |
|                 |    DELIVER frames,            |              |
|                 |    weight upload / full       |              |
|                 |    artifact download          |              |
|                 |                               |              |
|                 |  scheduler                    |              |
|                 |   - method-aware partition    |              |
|                 |   - GFLOPS-proportional split |              |
|                 |   - worker-level failover     |              |
|                 |   - memmap fan-in aggregation |              |
|                 +-------------------------------+              |
|                     ^         ^          ^                     |
|                     |  TASK_ASSIGN / HEARTBEAT / TASK_RESULT   |
|                     |  (+ slice artifacts on data plane)       |
|                     v         v          v                     |
|              +----------+ +----------+ +----------+            |
|              | Compute  | | Compute  | | Compute  |            |
|              | Node #1  | | Node #2  | | Node #n  |            |
|              |          | |          | |          |            |
|              | native   | | native   | | native   |            |
|              | runners: | | runners: | | runners: |            |
|              |  CPU /   | |  CPU /   | |  CPU /   |            |
|              |  CUDA /  | |  CUDA /  | |  CUDA /  |            |
|              |  Metal   | |  Metal   | |  Metal   |            |
|              |          | |          | |          |            |
|              | methods: | | methods: | | methods: |            |
|              |  gemv,   | |  gemv,   | |  gemv,   |            |
|              |  conv2d  | |  conv2d  | |  conv2d  |            |
|              +----------+ +----------+ +----------+            |
|                                                                |
+================================================================+
```

The main node exposes three network surfaces. UDP `:5353` handles mDNS
discovery and announcement. TCP `:52020` carries the protobuf control
plane: worker registration, heartbeats, task assignment, task results,
and client requests / responses. TCP `:52021` carries the bulk data
plane: framed `DELIVER` transfers for large inputs (such as conv2d
weight uploads) and large outputs (such as conv2d full-artifact
downloads).

Compute nodes serve as both benchmarkers and workers. On startup, each
node benchmarks every supported method on every surviving hardware
backend, then registers per-method GFLOPS summaries. At runtime, the
main node uses those summaries to partition work proportionally and
dispatch method-aware task slices to the native runner best suited for
each backend. When a worker dies mid-slice, the main node
re-partitions the lost range across survivors so the in-flight client
request still completes.

## Method Workloads

The project currently uses two representative methods:

| Method | Autotune Workload | Reported Workload | Primary Partition |
|---|---|---|---|
| `gemv` | `4096 x 8192` | `16384 x 32768` (`2 GiB` matrix) | contiguous row ranges |
| `conv2d` | `512 x 512`, `64 -> 128`, `k=3`, `pad=1`, `stride=1` | `2048 x 2048`, `128 -> 256`, `k=3`, `pad=1`, `stride=1` | contiguous output-channel ranges |

We use them as two different workload styles:

- `gemv`
  - represents a more bandwidth-sensitive, inference-like proxy workload
- `conv2d`
  - represents a more compute-dense, training-like proxy workload

The benchmark autotunes on the smaller workload, then reports speed on the
larger workload. For `conv2d`, protocol and runtime plumbing are
already in-tree, while the large-result data-plane path is still being
hardened for very large outputs. This is also a deliberate fallback tradeoff:
Windows places practical constraints on very large Python memory materialization,
home devices usually have much less RAM than disk capacity, and widespread SSD
adoption means sequential disk I/O is now fast enough that a disk-first artifact
path is often the safer choice when we are forced to choose.

## Dropped Approach: Payload Compression

Early in Sprint 4 we evaluated whether compressing large `FP32` matrix
artifacts on the data plane would shorten end-to-end transfer time.
The benchmark target was a representative `4 GiB` (`32768 x 32768` float32)
matrix from `Client/download`, with the constraint that compression must
finish in under `30s` on the sender (which may have a GPU) and that the
receiver is only guaranteed to have a CPU.

Tested categories:

- general lossless: `gzip`, `bz2`, `lzma`
- numeric-array lossless: `blosc2 + zstd / lz4 + shuffle / bitshuffle`
- scientific float: `zfp / zfpy`, `fpzip`, `SZ3`
- structural approximation: randomized `SVD`, `HODLR / H-matrix`

The best lossless candidate, `blosc2_zstd_shuffle_c1`, compressed in
`6.64s`, decompressed on CPU in `4.84s`, and saved `15.52%` of bytes
with full SHA256 equality. The best low-error candidate,
`zfpy_tolerance_1e-4`, saved `25.42%` at `2.86e-05` max absolute error.

The deciding question was not the compression ratio but the end-to-end
model `compress + send-side checksum + transfer + recv-side checksum + decompress`,
which gave these speedups:

| Method | 300 Mbps WiFi | 1 Gbps Ethernet |
|---|---:|---:|
| `blosc2_zstd_shuffle_c1` | `1.064x` | `0.886x` |
| `zfpy_tolerance_1e-4`    | `0.876x` | `0.519x` |
| `zfpy_tolerance_5e-4`    | `0.984x` | `0.575x` |

The break-even point for the best lossless option lands at roughly
`506 Mbps`. On the LAN this cluster actually runs on (`1 Gbps`
Ethernet, often higher), every candidate was a net regression.
We dropped compression from the data-plane path: at LAN speeds the
compress + decompress cost exceeds the byte savings. The full
methodology, dependency manifest, and per-method numbers are archived
under [`docs/compression_solution_report.md`](docs/compression_solution_report.md).
If a future deployment is bandwidth-limited (sub-`500 Mbps` WAN, mobile
uplink), `blosc2_zstd_shuffle_c1` is the candidate to reach for first.

## Progress Through Sprint 3

Sprint 1 delivered:

- bootstrap-driven startup
- Windows/Linux/macOS/WSL platform detection
- mDNS main-node discovery
- TCP runtime registration between main node and compute node
- protobuf runtime messaging
- worker heartbeat with failure-counter-based liveness eviction
- structured client request / response messaging
- task assignment / accept / fail / result messaging
- local CPU/CUDA benchmark generation and ranking
- compute-node performance summary upload during worker registration
- main-node tracking of total reported cluster GFLOPS

Sprint 2 delivered the current baseline:

- fixed-matrix-vector task distribution and result aggregation
- initial Windows DX12 compute benchmarking experiments for non-CUDA GPU paths
- a cleaner repository split across `app/`, `common/`, `wire/`, `main_node/`, and `compute_node/`
- explicit separation between `setup.py` environment preparation and `bootstrap.py` runtime startup
- shared compute-method source trees under `compute_node/compute_methods/`
- a deterministic shared GEMV dataset workspace under `compute_node/input_matrix/`
- three active in-tree GEMV backend families: CPU, CUDA, and Metal
- the DX12 source tree is retained for debugging, but its entry points are disabled because the module can trigger fatal system instability
- Windows GPU backend routing now defaults to the routine-safe CUDA path when an NVIDIA adapter is present
- a two-phase benchmark flow with autotune plus measurement for the reported result
- runtime use of benchmark summaries to register worker compute capacity
- expanded tests and documentation for the reorganized runtime model
- planning groundwork for method-aware runtime evolution beyond GEMV

Sprint 3 delivered:

- `conv2d` as the second production method, end-to-end through dispatch, worker execution, and aggregation
- method-aware worker registration so one worker reports separate GFLOPS summaries for `gemv` and `conv2d`
- method-aware main-node dispatch that partitions row ranges (`gemv`) or output-channel ranges (`conv2d`) per the client-requested method
- a separate TCP data plane for large artifacts, with DELIVER-frame upload for client-supplied conv2d weights and download IDs for large client responses
- shared backend contracts under `wire/external_protocol/` so the WinUI3 desktop frontend, the iOS frontend, and the Python CLI client all evolve against the same runtime concepts
- first WinUI3 desktop frontend covering cluster visibility, request submission, method selection, and result inspection
- first iOS frontend covering mobile monitoring and lightweight control flows
- a cleaner compute-method tree under `compute_node/compute_methods/` so runtime executors and benchmark backends share method source rather than duplicating it

A detailed Sprint 3 planning document lives at
`docs/sprint3_plan_2026-04-15.txt`.

## Sprint 4 Plan

Sprint 4 hardens the production path for release: fault tolerance, richer
observability, faster aggregation, a third compute method, an OS-driven
adaptive capacity signal that replaces the experimental idle benchmark
refresh, and the frontend / per-platform-backend work needed to ship the
clients.

Status legend: `[x]` shipped, `[~]` in progress, `[ ]` not started.

Runtime hardening:

- `[x]` worker-level task failover: when a worker dies mid-slice, the failed row / output-channel range is re-partitioned across surviving workers proportional to their original throughput, and the in-flight client request still completes
- `[x]` runtime protocol carries per-slice timing diagnostics (compute, fetch, peripheral milliseconds) so the final `CLIENT_RESPONSE` records where time was spent on every worker
- `[x]` multi-mode conv2d client response: `stats_only` summary, sampled preview, or full artifact download, so very large outputs do not have to be materialized client-side
- `[x]` faster fan-in / fan-out aggregation: conv2d output assembly streams worker artifacts via `np.memmap` strided assignment instead of per-pixel Python I/O

Compute method:

- `[ ]` add `gemm` as a third production method, including benchmark, dispatch, runtime executor, and aggregation paths

Adaptive capacity:

- `[~]` replace the experimental idle benchmark refresh feature with a formal adaptive capacity update driven by the OS, reading load from Windows Task Manager and macOS Activity Monitor instead of re-running benchmarks while a worker is idle

Frontend and release:

- `[ ]` update the WinUI3 desktop frontend to consume the latest control-plane and data-plane protocol, including timing diagnostics and the new conv2d response modes
- `[ ]` update the iOS frontend the same way
- `[ ]` give each platform's client its own backend logic so per-platform clients can be packaged and released independently
- `[ ]` Android native app — included in the plan for completeness, but likely out of scope for this submission window

## Project Entry

There are two entrypoints, split along the internet boundary:

**`setup.py`** — the *internet* step. Prepares the machine: creates
`.venv`, installs `requirements.txt`, and reaches out to PyPI to resolve
Python dependencies. This is the only entrypoint that needs WAN access.

```bash
python setup.py
```

**`bootstrap.py`** — the *LAN-only* step. Starts the runtime on an
already-prepared machine. Once `.venv` is in place, bootstrap never
touches the wide-area network: mDNS discovery runs on the local
subnet's multicast group, and all inter-node traffic is TCP inside the
LAN.

```bash
python bootstrap.py --role discover
```

or:

```bash
python bootstrap.py --role announce
```

`bootstrap.py` stays at the repository root on purpose. It is the one
file a human should be able to find immediately. On a fresh clone where
`.venv` is missing, bootstrap will invoke `setup.py` on your behalf so
the first-run flow still works end-to-end — but that first run is the
only time bootstrap transitively reaches WAN, and it does so strictly
by delegating to `setup.py`.

## Current Layout

The repository is now organized by responsibility:

- `bootstrap.py`
  - root-level entrypoint
  - expects the local project environment to be prepared already
  - relaunches itself with the project `.venv` interpreter when needed
  - ensures compute benchmark results exist
  - hands control to the runtime supervisor
- `setup.py`
  - environment-preparation entrypoint
  - creates `.venv`
  - installs `requirements.txt` when needed
  - makes "local-only" versus "may need network" setup steps explicit
- `app/`
  - application-level support modules
  - runtime config, constants, supervisor, logging, recovery, and tracing
- `common/`
  - shared dataclasses and reusable helpers
  - float32 packing, hardware types, message labels, and work partitioning
- `adapters/`
  - platform, firewall, process, audit-log, and socket adapters
- `discovery/`
  - mDNS discovery, packet handling, and manual fallback
- `wire/proto/`
  - `.proto` source definitions
- `wire/discovery_protocol/`
  - discovery wire-format helpers
- `wire/internal_protocol/`
  - main-node <-> compute-node control plane, transport framing, and data-plane helpers
- `wire/external_protocol/`
  - client-facing control-plane and data-plane models
- `main_node/`
  - scheduler-side registry, dispatch, aggregation, heartbeat, and runtime loop
- `compute_node/`
  - worker-side runtime session, benchmark summary loading, and task execution
- `compute_node/compute_methods/`
  - shared method implementations and hardware-specific runners
  - used by both runtime task execution and local benchmarking
- `compute_node/input_matrix/`
  - shared deterministic dataset generator and local dataset cache
- `compute_node/performance_metrics/`
  - local benchmark orchestration, ranking, and result reporting
- `experiments/networking/`
  - standalone networking experiments kept outside the main runtime path
- `docs/`
  - source-focused project tree and planning documents
- `tests/`
  - automated tests

This keeps the root directory small and recognizable while still leaving the
main operational entrypoint in the obvious place.

## Bootstrap Behavior Tree

The current startup flow is:

```mermaid
flowchart TD
    A([python bootstrap.py]) --> B{project .venv<br/>ready?}
    B -- no --> C[run setup.py,<br/>relaunch under .venv]
    C --> F
    B -- yes --> D{running under<br/>project .venv?}
    D -- no --> E[relaunch self with<br/>.venv interpreter]
    E --> F
    D -- yes --> F{benchmark result.json<br/>present?<br/>--retest / --rebuild off?}
    F -- no --> G[run benchmark.py for<br/>every supported method]
    G --> H
    F -- yes --> H[detect platform,<br/>configure firewall rules]
    H --> I{--role?}
    I -- announce --> J([start as main node<br/>answer mDNS,<br/>accept TCP connections])
    I -- discover --> K[mDNS discovery<br/>--discover-attempts rounds]
    K --> L{main node<br/>found?}
    L -- yes --> M[connect over TCP<br/>as compute node]
    L -- no --> N{--manual-fallback?}
    N -- yes --> O[prompt operator for<br/>main-node address]
    O --> M
    N -- no --> P([hard fail])
    M --> Q[send REGISTER_WORKER<br/>with hardware + GFLOPS summary]
    Q --> R[receive REGISTER_OK,<br/>runtime id assigned]
    R --> S([runtime loop:<br/>HEARTBEAT / TASK_ASSIGN /<br/>TASK_RESULT / CLIENT_REQUEST])
    J --> S
```

In detail:

1. `bootstrap.py` starts.
2. It checks whether the project `.venv` and dependency stamp are already ready.
3. If the environment is not ready, it invokes `python setup.py` — the only step in the whole startup path that reaches the wide-area network, since `setup.py` installs from PyPI. On every subsequent run with a prepared `.venv`, this step is skipped and bootstrap stays LAN-only.
4. If the environment is ready but the current interpreter is not the project `.venv`, it relaunches itself with the project interpreter.
5. It checks for `compute_node/performance_metrics/result.json`.
6. If benchmark results are missing, it runs `compute_node/performance_metrics/benchmark.py`.
7. It detects platform capabilities and configures firewall rules where supported.
8. It enters `app/supervisor.py`.
9. If `--role announce`, it becomes the main node immediately.
10. If `--role discover`, it tries mDNS discovery several times.
11. If discovery succeeds, it joins the discovered main node as a compute node.
12. If discovery fails, it promotes itself into the main node runtime.
13. A compute node sends `REGISTER_WORKER` with:
    - host hardware profile
    - filtered hardware backend count
    - ranked backend GFLOPS summary from the benchmark
14. The main node assigns a runtime id to the compute node, assigns per-hardware ids, updates total cluster GFLOPS, then replies with `REGISTER_OK`.
15. Clients join, receive their own runtime ids, and send structured `CLIENT_REQUEST` messages.
16. Each `CLIENT_REQUEST` can include an `iteration_count`, and the main node emits matching `TASK_ASSIGN` slices to workers.
17. While a request is active, clients can poll `CLIENT_INFO_REQUEST` / `CLIENT_INFO_REPLY` for active-request visibility.
18. Workers send periodic `HEARTBEAT_OK`, plus `WORKER_UPDATE` messages whenever their effective performance changes.
19. The main node aggregates worker slices and returns one `CLIENT_RESPONSE`, using artifact descriptors plus the TCP data plane for large outputs.

## Local, LAN, And Internet Steps

The runtime splits cleanly across three scopes. Only one of them
requires the wide-area network.

- **Machine-local only** (no network at all):
  - `python setup.py --venv-only` (create `.venv` without installing)
  - reading config and local files
  - running the local benchmark (`compute_node/performance_metrics/benchmark.py`)
  - starting the main node directly with `--role announce`
- **LAN only** (local subnet, no WAN):
  - mDNS discovery in `--role discover` (UDP multicast to `224.0.0.251:5353`)
  - TCP runtime registration between nodes (`:52020`)
  - TCP data-plane artifact transfers (`:52021`)
  - heartbeats, task assignment, results, client requests
- **Internet required** (the only WAN step):
  - `python setup.py` — resolves Python dependencies against PyPI
  - `pip install -r requirements.txt` — the specific call `setup.py` makes

`bootstrap.py` itself never opens a WAN connection. If the local
environment is missing or stale, `bootstrap.py` delegates to
`python setup.py` and then relaunches under `.venv` — so the WAN hop
stays fully owned by `setup.py` and is skipped entirely on every
subsequent run. In an offline or air-gapped deployment, run `setup.py`
once against a local mirror (or pre-populate `.venv`), after which the
cluster operates end-to-end on LAN only.

## Current Runtime Model

The runtime currently supports two bootstrap roles:

- `discover`
  - behaves like a compute node first
  - searches for an existing main node with mDNS
  - connects over TCP if found
  - otherwise promotes itself into the main node
- `announce`
  - starts directly as the main node
  - answers mDNS
  - accepts worker and client TCP runtime connections

The main runtime protobuf messages are:

- `REGISTER_WORKER`
- `REGISTER_OK`
- `HEARTBEAT`
- `HEARTBEAT_OK`
- `CLIENT_JOIN`
- `CLIENT_INFO_REQUEST`
- `CLIENT_INFO_REPLY`
- `CLIENT_REQUEST`
- `CLIENT_RESPONSE`
- `TASK_ASSIGN`
- `TASK_ACCEPT`
- `TASK_FAIL`
- `TASK_RESULT`
- `ARTIFACT_RELEASE`
- `WORKER_UPDATE`

The runtime and benchmark stack now recognize two methods:

- `gemv`
- `conv2d`

`gemv` is the most mature end-to-end path today:

- the client sends one FP32 vector payload
- the main node partitions matrix rows by registered GFLOPS
- compute nodes execute their assigned row ranges on the processors that
  survived local benchmark filtering
- the main node aggregates row slices into one `CLIENT_RESPONSE`

`conv2d` is already integrated into the method registry, benchmark
workspace, and runtime handler structure:

- the benchmark path is fully method-aware and uses test/runtime dataset pairs
- the runtime protocol and executors understand output-channel slicing
- large-result transport is the remaining hardening area for the biggest
  outputs, so that path is still under active development

Across both methods, `iteration_count` is a compute-side loop count for one
structured request, not a request-resend count at the client or main-node
layer.

Worker liveness is now driven by the periodic heartbeat loop and a consecutive
failure counter. The main node no longer applies an additional hard task-result
deadline on top of heartbeat-based liveness. Client-side liveness works the
same way: once a client has an active request in flight, it periodically sends
`CLIENT_INFO_REQUEST`, treats each matching `CLIENT_INFO_REPLY` as a successful
refresh, and only marks the cluster dead after repeated missed replies.

## Quick Start

The simplest invocation is:

```bash
python bootstrap.py
```

With no flags, `bootstrap.py` defaults to `--role discover`. That triggers
this sequence:

1. Verify the project `.venv` and `requirements.txt` stamp are current; run
   `setup.py` and self-relaunch into `.venv` if not. **This relaunch is
   synchronous** — the parent blocks on the child with `subprocess.run`,
   then forwards its exit code. Parent and child share the same console.
2. Detect the platform; on Windows, optionally elevate to admin (only when
   `--elevate-if-needed` is also set). **This relaunch is asynchronous and
   detached** — `ShellExecuteW` hands off to a new elevated process with
   its own console, and the parent exits immediately. This is a one-way
   handoff: any future "relaunch under a different identity" flag
   (headless / dashboard mode, alternate user, etc.) should merge into
   this same decision point rather than introduce a third self-relaunch,
   otherwise order and detach semantics stop composing cleanly.
3. Make sure `compute_node/performance_metrics/result.json` exists; if it
   does not (or `--retest` / `--rebuild` was passed), run the local
   benchmark first.
4. Configure firewall rules for the UDP discovery port and the TCP
   data-plane port.
5. Try mDNS discovery `--discover-attempts` times (default `3`), waiting up
   to `--timeout` seconds (default `1.0`) per attempt and `--retry-delay`
   seconds (default `0.3`) between attempts.
6. If a main node is found, register as a compute node over TCP. If
   discovery exhausts every attempt, promote this process into the main
   node itself (unless `--no-manual-fallback` was passed and you want a
   hard failure instead).

`--role announce` skips the discovery loop and starts as the main node
immediately.

Start in discover mode (the implicit default, written explicitly):

```bash
python bootstrap.py --role discover --udp-port 5353 --tcp-port 52020
```

Start in announce mode:

```bash
python bootstrap.py --role announce --udp-port 5353 --tcp-port 52020
```

### Bootstrap Options

| Flag | Default | Purpose |
| --- | --- | --- |
| `--role {discover,announce}` | `discover` | `discover` looks for an existing main node first and promotes self only if discovery fails. `announce` starts directly as the main node. |
| `--node-name NAME` | `node` | Cluster label this process advertises. The default resolves to `main node` when role is `announce` and `compute node` otherwise; pass an explicit name only when you want a custom label. |
| `--multicast-group ADDR` | `224.0.0.251` | IPv4 multicast group used for discovery traffic. Override only if your LAN reserves the default. |
| `--udp-port PORT` | `5353` | UDP port used for multicast announce / discover packets. |
| `--tcp-port PORT` | `52020` | TCP port the main-node runtime server listens on for worker and client control connections. |
| `--data-plane-port PORT` | `52021` | TCP port the main-node artifact data plane listens on for large-artifact upload / download (DELIVER frames, conv2d weight uploads, conv2d full-output downloads). |
| `--timeout SECONDS` | `1.0` | Max seconds to wait for a discovery reply per attempt. Raise on lossy networks. |
| `--discover-attempts N` | `3` | How many discovery rounds to run before falling back. With the defaults this is roughly 3–4 seconds of total discovery time before self-promotion. |
| `--retry-delay SECONDS` | `0.3` | Pause between discovery attempts. |
| `--manual-fallback / --no-manual-fallback` | `--manual-fallback` (on) | When discovery fails, prompt the operator for a manual main-node address instead of immediately promoting self. Pass `--no-manual-fallback` for non-interactive runs that should hard-fail when discovery does not find a main node. |
| `--elevate-if-needed` | off | On Windows, relaunch the process with administrator privileges if not already admin. **The name is aspirational** — the bootstrap does not probe whether elevation is actually needed; it elevates eagerly whenever this flag is set and the current process is not admin. The only thing in the runtime that genuinely requires admin is Windows firewall rule installation; non-admin runs simply skip rule install and log a warning. The bootstrap auto-injects this flag into its own `.venv` self-relaunch so behavior is consistent. |
| `--no-cli` | off | Run headless — no console window for the main process, and spawned peers (dual-purpose compute-node) also detach instead of opening their own console. Merges with `--elevate-if-needed` at step 2: when both are set, the UAC handoff uses `SW_HIDE` so the elevated child has no console; when `--no-cli` is set alone, the bootstrap detaches via `DETACHED_PROCESS` after UAC. Only Task Manager / `kill` can stop the process; logs still land in the role-specific log file. Intended for dashboard / service deployments. |
| `--retest` | off | Regenerate the deterministic input matrices and rerun the local benchmark before startup. Use after changing dataset shapes or after replacing hardware. |
| `--rebuild` | off | Rerun the local benchmark and force the native runner binaries to rebuild, but do **not** regenerate input matrices. Use after editing a backend's CUDA / Metal / CPU source. |
| `--log-start-mode {normal,clean,cleanse}` | `normal` | `normal` keeps prior session logs untouched. `clean` archives previous loose log files into a dated archive directory. `cleanse` permanently removes them. |
| `--verbose` | off | Enable DEBUG-level logging in the bootstrap session log. |

Run only the benchmark workspace:

```bash
python "compute_node/performance_metrics/benchmark.py"
```

That default benchmark now runs both supported methods in sequence and writes a
combined report plus crash-survivable progress files.

Generate only the shared matrix/vector dataset:

```bash
python "compute_node/input_matrix/generate.py"
```

## Notes

- The public project/app name is `superweb-cluster`.
- `bootstrap.py` is the intended top-level entry. Running lower-level modules
  directly is mainly for development and debugging.
- Root-level support files were moved into `app/` so the repository root stays
  focused on entrypoints and major subsystems.
- Standalone networking experiments now live under `experiments/networking/`.
- Tree and planning documents now live under `docs/`.
- Generated datasets under `compute_node/input_matrix/generated/` and benchmark
  reports such as `compute_node/performance_metrics/result.json`,
  `compute_node/performance_metrics/result_status.json`, and
  `compute_node/performance_metrics/result_trace.jsonl` are local machine
  artifacts and stay git-ignored.
- Shared compute-method implementations now live under
  `compute_node/compute_methods/`, so runtime executors and benchmark backends
  no longer hide method source trees inside `performance_metrics/`.
- The GEMV compute method currently exposes three routine-safe native backend
  families in-tree: CPU, CUDA, and Metal.
- The DX12 module is currently disabled in this build. Repeated
  `conv2d` benchmark runs on the AMD Radeon 780M path caused
  system-level crashes severe enough to require a BIOS power reset before the
  machine would respond to the power button again.
- DX12 source files are still kept in-tree for postmortem debugging, but
  benchmark and runtime entry points now reject DX12 requests with a fatal
  warning instead of attempting to run them.
- The fixed GEMV benchmark uses a two-phase workload:
  autotune each candidate config with `3` repeats, then measure the winning
  config with `20` repeats for the reported result.
- The dataset generator is independent from `performance_metrics/`; the
  benchmark workspace consumes `compute_node/input_matrix/` rather than owning
  the input-file format itself.
