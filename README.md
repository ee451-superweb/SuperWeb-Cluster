# superweb-cluster

## Status

`superweb-cluster` has completed Sprint 1 and is moving into Sprint 2.

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
- fixed-matrix-vector task distribution and result aggregation

Sprint 2 will build on that with broader workload scheduling and more methods.

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
- `compute_node/input_matrix/`
  - shared deterministic dataset generator and local dataset cache
- `compute_node/performance_metrics/`
  - local CPU/CUDA/Metal benchmark workspace
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
14. The main node assigns per-hardware ids, updates total cluster GFLOPS, then replies with `REGISTER_OK`.
15. Clients send structured `CLIENT_REQUEST` messages.
16. The main node emits `TASK_ASSIGN` slices to workers and waits for `TASK_RESULT` or `TASK_FAIL`.
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
- The fixed FMVM benchmark uses a two-phase workload:
  autotune each candidate config with `3` repeats, then measure the winning
  config with `20` repeats for the reported result.
- The dataset generator is independent from `performance_metrics/`; the
  benchmark workspace consumes `compute_node/input_matrix/` rather than owning
  the input-file format itself.

