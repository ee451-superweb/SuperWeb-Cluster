# superweb-cluster

A LAN-only distributed compute cluster for heterogeneous home hardware.
One main node coordinates work across compute nodes that auto-discover via
mDNS, benchmark themselves on startup, and execute method-aware task slices
on whichever native backend (CPU / CUDA / Metal) is fastest on each box.

This repository hosts the cluster runtime — main node, compute node,
benchmark stack, and shared protocol. The user-facing clients (CLI,
WinUI3 desktop, iOS) live in the sibling [`Client/`](../Client/) repo.

## Status

Sprint 4 has shipped — this is the final sprint of the submission cycle.
Runtime fault tolerance, per-slice timing diagnostics, multi-mode `conv2d`
client responses, memmap-based aggregation, and a third compute method
(`gemm`) are all in place. The OS-driven adaptive capacity probe and the
per-platform frontend updates are partially scoped and have been written
up under [Future Work](#future-work) rather than rushed.

## Quick start

Requires Python 3.10+ on Windows / Linux / macOS / WSL.

```bash
# Prepare .venv and resolve dependencies (the only step that uses WAN)
python setup.py

# Start as the main node — answers mDNS, accepts workers and clients
python bootstrap.py --role announce

# On every other LAN box: discover the main node, register as a worker
python bootstrap.py --role discover
```

`--role discover` (the default if you omit `--role`) tries mDNS three times,
then promotes itself to main node if no main node was found, so a fresh LAN
brought up with `python bootstrap.py` on every box converges automatically.

On Windows, double-clicking `run.cmd` runs `setup.py` then `bootstrap.py`
with UAC elevation in one shot.

To submit work, clone [`Client/`](../Client/) on any LAN box (or the same
box) and follow its README — the Python reference client at
`Client/python/oneshot.py` is the simplest entry point.

## What's in the box

- **Auto-discovery.** Zero-config mDNS on the LAN. No central registry,
  no static config file, no DNS dependency.
- **Method-aware benchmarking.** Each compute node benchmarks every
  supported method on every viable hardware backend during startup, and
  reports per-method GFLOPS during `REGISTER_WORKER`. The main node uses
  those numbers to size each worker's slice.
- **GFLOPS-proportional scheduling.** Slices are partitioned by measured
  throughput, not by worker count. A 4090 box and a Raspberry Pi register
  on the same cluster and get appropriately different slice sizes.
- **Worker-level failover.** If a worker dies mid-slice, its row /
  output-channel range is re-partitioned across survivors proportional to
  their original throughput, and the in-flight client request still
  completes.
- **Per-slice timing diagnostics.** Compute / fetch / peripheral
  milliseconds flow back through the result protocol so the client
  response shows where time was actually spent on each worker.
- **Three compute methods**, three workload styles:

  | Method | Workload style | Reported workload |
  |---|---|---|
  | `gemv` | bandwidth-sensitive, inference-like | `16384 × 32768` matrix × vector |
  | `conv2d` | compute-dense, training-like | `2048 × 2048`, `128→256`, `k=3` |
  | `gemm` | vendor-tuned dense linear algebra | CPU SGEMM + cuBLAS where available |

- **Two TCP planes.** Control plane on `:52020` carries protobuf
  messages (registration, heartbeats, task assignment / result, client
  requests). Data plane on `:52021` carries large artifacts as DELIVER
  frames — used for `conv2d` weight uploads and full-output downloads.
  `:5353` UDP carries mDNS only.
- **Crash-survivable benchmark traces.** `result_status.json` and
  `result_trace.jsonl` persist benchmark progress so a mid-run crash is
  recoverable without rerunning the whole sweep.

## Architecture at a glance

```text
   +---------+        mDNS                +-------------+
   | Client  | -----  :5353 UDP   ---->   |  Main node  |  scheduler:
   |  CLI /  | ----- :52020 TCP ---->     |             |   - method-aware
   |  WinUI3 | ----- :52021 TCP ---->     |  +-------+  |   - GFLOPS split
   |  iOS    | <------- responses -----   |  | regis |  |   - failover
   +---------+        artifacts           |  +-------+  |   - memmap fan-in
                                          +------+------+
                                                 |
                          control + data plane   |
                          + TASK_ASSIGN / RESULT |
                                                 v
                  +--------------+  +--------------+  +--------------+
                  | Compute node |  | Compute node |  | Compute node |
                  | CPU+CUDA     |  | CPU+CUDA     |  | CPU+Metal    |
                  | gemv/conv2d/ |  | gemv/conv2d/ |  | gemv/conv2d/ |
                  | gemm         |  | gemm         |  | gemm         |
                  +--------------+  +--------------+  +--------------+
```

A discover-mode bootstrap behaves as a compute node first; if no main node
answers mDNS within `--discover-attempts` rounds (default 3), it promotes
itself into the main-node role. That makes "first machine up" automatically
become the coordinator without any role configuration.

## Repository layout

```text
SuperWeb-Cluster/
├── bootstrap.py              # the only entry point operators run
├── setup.py                  # creates .venv, installs requirements.txt
├── run.cmd                   # Windows double-click launcher
├── adapters/                 # platform / firewall / process / audit-log adapters
├── core/                     # config, constants, logging, recovery, tracing
├── supervision/              # supervisor lifecycle, peer heartbeat
├── discovery/                # mDNS discovery + manual fallback
├── transport/                # framed TCP, data-plane sender / receiver
├── wire/
│   ├── proto/                # .proto sources
│   ├── discovery_protocol/   # mDNS wire format
│   ├── internal_protocol/    # main_node ↔ compute_node control + data plane
│   └── external_protocol/    # client-facing models (mirrored in Client/proto/)
├── main_node/                # registry, dispatcher, aggregator, request handler
├── compute_node/
│   ├── compute_methods/      # gemv / conv2d / gemm — sources + runtime handlers
│   ├── input_matrix/         # deterministic dataset generator
│   └── performance_metrics/  # benchmark stack (autotune + ranking)
├── tests/                    # 300+ unit / integration tests
├── experiments/networking/   # standalone networking experiments, off the runtime path
├── benchmarks/               # ad-hoc microbenchmark scripts
└── docs/                     # detailed design + postmortems + future work
```

The first line you run is `bootstrap.py`. Everything else is reached
through it. `setup.py` is invoked transitively on a fresh clone where
`.venv` is missing.

## Documentation

- **[`docs/technical-detail.md`](docs/technical-detail.md)** — the deep
  dive: full bootstrap behavior tree, every CLI flag, the protobuf
  message catalog, sprint-by-sprint deliverables, the compression
  evaluation, and the `Future Work` section. If you are reading code, you
  want this open in another tab.
- **[`docs/2026-04-26_known_issues_no_global_task_pool.md`](docs/2026-04-26_known_issues_no_global_task_pool.md)**
  — why multi-client concurrency and failover retry both bottleneck on the
  same scheduling gap, and the global-queue + worker-pull migration plan.
- **[`docs/2026-04-26_known_issues_compile_logic_misplaced.md`](docs/2026-04-26_known_issues_compile_logic_misplaced.md)**
  — why native runner compilation sits in the wrong package today, why
  runtime code reverse-depends on the benchmark module, and the per-method
  `build.py` relocation plan.
- **[`docs/2026-04-21_conv2d_runtime_findings.md`](docs/2026-04-21_conv2d_runtime_findings.md)**
  — postmortem of four conv2d runtime bugs fixed during Sprint 4.
- **[`docs/2026-04-22_gpu_sharing_stress_test.md`](docs/2026-04-22_gpu_sharing_stress_test.md)**
  — single-machine 2×Metal + 1×CPU stress test, explaining why two Metal
  workers on the same Apple Silicon GPU end up at half throughput each
  rather than 2× combined.
- **[`docs/compression_solution_report.md`](docs/compression_solution_report.md)**
  — full methodology and per-method numbers for the dropped data-plane
  compression evaluation.

## Future work

The four threads below are scoped but not landed. Each has a dedicated
document under [`docs/`](docs/) with line-level diagnosis and migration
steps; the technical-detail README has the inlined version. The short
form:

1. **Compile logic in the wrong layer.** Native runner compilation
   currently lives under `compute_node/performance_metrics/<method>/backends/`
   instead of next to the sources under `compute_node/compute_methods/`.
   Side effect: runtime code reverse-imports the benchmark module to get
   compiled binaries, and `bootstrap.py --rebuild` cannot be made
   compile-only. Fix: split each `*_backend.py` into a build half (moves
   to `compute_methods/<method>/build/`) and an autotune-and-measure half
   (stays in `performance_metrics/`).
2. **No global task pool, no work stealing.** The dispatcher is per-request
   and stateless across requests, so two concurrent clients can target the
   same worker, and failover retry slices serialize behind survivors'
   in-flight original slices on the per-worker `task_lock`. Fix: a global
   slice queue plus `WORKER_REQUEST_WORK` / `WORKER_NO_WORK_AVAILABLE`
   pull semantics on the worker side.
3. **Bootstrap CLI orthogonality.** `--retest` should run only the
   benchmark, `--rebuild` should compile only, and a new `--regenerate`
   should regenerate input matrices only. Two of three are already
   supported by the underlying scripts; `--rebuild` blocks on (1).
4. **Sprint 4 carry-over.** OS-driven adaptive capacity probe (Windows
   perf counters / macOS Activity Monitor), WinUI3 + iOS frontend protocol
   catch-up, per-platform packaged client backends, Android native app.

## Notes for peer developers

- The project name on the wire and in audit logs is `superweb-cluster`.
- `bootstrap.py` is the intended entry point. Running scripts under
  `compute_node/performance_metrics/` or `compute_node/input_matrix/`
  directly is fine for development but not part of the operator flow.
- The DX12 backend source tree is kept in-tree but disabled at runtime —
  repeated conv2d runs against the AMD Radeon 780M DX12 path triggered
  fatal system instability requiring a BIOS power reset. See
  [`docs/technical-detail.md`](docs/technical-detail.md) for the full
  context. Do not re-enable on that hardware without first reading it.
- All inter-node traffic is LAN-only after `setup.py` finishes. Air-gapped
  deployments work after a one-time `setup.py` against a local PyPI
  mirror.
