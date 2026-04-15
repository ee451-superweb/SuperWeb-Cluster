# superweb-cluster

## Status

`superweb-cluster` has completed the Sprint 1 and Sprint 2 foundation work and
is now entering Sprint 3.

Sprint 1 established the baseline runtime: bootstrap, discovery, registration,
protobuf messaging, heartbeat, and the first end-to-end distributed compute
path.

Sprint 2 hardened that baseline around the first production method, expanded
hardware benchmark coverage, and reshaped the repository so new compute methods
can be added without collapsing everything back into one runtime path.

## Progress Through Sprint 2

Sprint 1 delivered:

- bootstrap-driven startup
- Windows/Linux/macOS/WSL platform detection
- mDNS main-node discovery
- TCP runtime registration between main node and compute node
- protobuf runtime messaging
- worker heartbeat and removal on timeout
- structured client request / response messaging
- task assignment / accept / fail / result messaging
- local CPU/CUDA benchmark generation and ranking
- compute-node performance summary upload during worker registration
- main-node tracking of total reported cluster GFLOPS

Sprint 2 delivered the current baseline:

- fixed-matrix-vector task distribution and result aggregation
- Windows DX12 compute benchmarking for non-CUDA GPU paths
- a cleaner repository split across `app/`, `common/`, `wire/`, `main_node/`, and `compute_node/`
- explicit separation between `setup.py` environment preparation and `bootstrap.py` runtime startup
- shared compute-method source trees under `compute_node/compute_methods/`
- a deterministic shared FMVM dataset workspace under `compute_node/input_matrix/`
- four in-tree FMVM backend families: CPU, CUDA, DX12, and Metal
- Windows GPU backend routing that distinguishes NVIDIA/CUDA from non-NVIDIA/DX12 paths
- a two-phase benchmark flow with autotune plus measurement for the reported result
- runtime use of benchmark summaries to register worker compute capacity
- expanded tests and documentation for the reorganized runtime model
- planning groundwork for method-aware runtime evolution beyond FMVM

## Sprint 3 Plan

Sprint 3 will focus on turning `superweb-cluster` from a single-method cluster
into a multi-method platform, while also exposing more of the system through
user-facing frontends.

Compute-method goals:

- add more executable methods beyond `fixed_matrix_vector_multiplication`
- start with `spatial_convolution` / `conv2d` as the next major method
- make worker registration method-aware so one worker can report different GFLOPS summaries for different methods
- make main-node dispatch and aggregation method-aware so scheduling follows the client-requested method
- continue restructuring the compute-node runtime around method handlers and reusable task routing

Frontend goals:

- build corresponding frontend features for cluster visibility, request submission, method selection, and result inspection
- expose worker inventory, benchmark rankings, active method, and scheduler decisions in a human-friendly way
- reduce reliance on terminal-only workflows for demos, debugging, and operator control

Frontend coverage goals:

- expand beyond the current CLI-first experience
- add Windows frontend coverage for desktop demos and operator workflows
- add iOS frontend coverage for mobile monitoring and lightweight control scenarios
- keep shared backend contracts so web, Windows, and iOS surfaces can evolve against the same runtime concepts

A detailed Sprint 3 planning document lives at
`docs/sprint3_plan_2026-04-15.txt`.

## Project Entry

The main entrypoint is still:

```bash
python bootstrap.py --role discover
```

or:

```bash
python bootstrap.py --role announce
```

`bootstrap.py` stays at the repository root on purpose. It is the one file a
human should be able to find immediately.

The environment-preparation entrypoint is:

```bash
python setup.py
```

Use this when you want to separate "prepare the machine" from "start the
runtime."

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
- `wire/discovery.py`
  - discovery wire-format helpers
- `wire/runtime.py`
  - framed protobuf runtime wire-format helpers
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

1. `bootstrap.py` starts.
2. It checks whether the project `.venv` and dependency stamp are already ready.
3. If the environment is not ready, it stops and tells the user to run `setup.py`.
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
17. The main node aggregates row slices and returns one `CLIENT_RESPONSE`.

## Local Vs Networked Steps

- Local-only:
  - `python setup.py --venv-only`
  - creating `.venv`
  - reading config and local files
  - running the local benchmark
  - starting the main node directly with `--role announce`
- May require network access:
  - `python setup.py`
  - `pip install -r requirements.txt`
  - mDNS discovery in `--role discover`
  - TCP runtime registration between nodes

`bootstrap.py` itself is not responsible for creating `.venv` or installing
dependencies anymore. If the local environment is not ready, it will stop and
ask you to run `python setup.py` first.

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
- `CLIENT_REQUEST`
- `CLIENT_RESPONSE`
- `TASK_ASSIGN`
- `TASK_ACCEPT`
- `TASK_FAIL`
- `TASK_RESULT`

Right now the active executable method is:

- `fixed_matrix_vector_multiplication`

The client sends one 32K-length FP32 vector, the main node splits matrix rows
across workers in proportion to registered GFLOPS, and compute nodes execute
their assigned row ranges on the processors they kept after local filtering.
Both the client response and worker task messages echo the assigned runtime id
and the requested `iteration_count`.
That `iteration_count` is a compute-side loop count for one structured request,
not a request-resend count at the client or main-node layer.

## Quick Start

Start in discover mode:

```bash
python bootstrap.py --role discover --udp-port 5353 --tcp-port 52020
```

Start in announce mode:

```bash
python bootstrap.py --role announce --udp-port 5353 --tcp-port 52020
```

Run only the benchmark workspace:

```bash
python "compute_node/performance_metrics/benchmark.py"
```

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
  reports such as `compute_node/performance_metrics/result.json` are local
  machine artifacts and stay git-ignored.
- Shared compute-method implementations now live under
  `compute_node/compute_methods/`, so runtime executors and benchmark backends
  no longer hide method source trees inside `performance_metrics/`.
- The FMVM compute method now has four native backend families in-tree:
  CPU, CUDA, DX12, and Metal.
- On Windows, automatic GPU backend selection now inspects display adapters:
  NVIDIA adapters route to CUDA, and non-NVIDIA adapters route to DX12.
  You can still force DX12 explicitly with
  `compute_node/performance_metrics/benchmark.py --backend dx12`.
- The fixed FMVM benchmark uses a two-phase workload:
  autotune each candidate config with `3` repeats, then measure the winning
  config with `20` repeats for the reported result.
- The dataset generator is independent from `performance_metrics/`; the
  benchmark workspace consumes `compute_node/input_matrix/` rather than owning
  the input-file format itself.

