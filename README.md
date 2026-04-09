# superweb-cluster

## Status

`superweb-cluster` has completed Sprint 1 and is about to enter Sprint 2.

Sprint 1 delivered the LAN control plane and local benchmarking foundation:

- bootstrap-driven startup
- Windows/Linux/macOS/WSL platform detection
- mDNS discovery for the main node
- TCP runtime registration between main node and compute node
- protobuf runtime messaging
- worker heartbeat and removal on timeout
- client join/request/response flow
- local CPU/CUDA benchmark generation and ranking
- compute-node performance summary upload during worker registration
- main-node tracking of total reported cluster GFLOPS

Sprint 2 is expected to build on top of this:

- load distribution
- worker hardware-aware scheduling
- execution backends
- result aggregation
- broader scheduling policy beyond the current registration/runtime skeleton

## Project Entry

The main entrypoint for the whole project is:

```bash
python bootstrap.py --role discover
```

or:

```bash
python bootstrap.py --role announce
```

Everything else is supporting infrastructure:

- `bootstrap.py`
  - top-level entrypoint
  - prepares `.venv`
  - installs `requirements.txt` when needed
  - ensures compute benchmark results exist
  - then hands control to the runtime supervisor
- `supervisor.py`
  - decides whether this process becomes a compute node or the main node
- `compute_node/`
  - worker-side runtime logic and benchmark summary loading
- `main_node/`
  - main-node runtime, registry, and cluster capability tracking
- `compute_node/performance_metrics/`
  - fixed benchmark workspace used to characterize local compute hardware
- `standalone_model/`
  - isolated network experiments, not the main runtime entry

## Bootstrap Behavior Tree

The current Sprint 1 startup flow is:

1. `bootstrap.py` starts.
2. It prepares the project-local `.venv`.
3. It installs or refreshes `requirements.txt` if the dependency hash changed.
4. It checks for `compute_node/performance_metrics/result.json`.
5. If benchmark results are missing, it runs `compute_node/performance_metrics/benchmark.py`.
6. It detects platform capabilities and configures firewall rules where supported.
7. It enters `Supervisor.run()`.
8. If `--role announce`, it becomes the main node immediately.
9. If `--role discover`, it tries mDNS discovery several times.
10. If discovery succeeds, it joins the discovered main node as a compute node.
11. If discovery fails, it promotes itself into the main node runtime.
12. Once connected as a compute node, it sends `REGISTER_WORKER` with:
    - host hardware profile
    - hardware backend count
    - ranked backend GFLOPS summary from the benchmark
13. The main node assigns per-hardware ids, updates total cluster GFLOPS, then replies with `REGISTER_OK`.
14. The runtime stays alive with worker heartbeats and client/runtime traffic until shutdown.

## Current Runtime Model

Sprint 1 currently uses two bootstrap roles:

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

`REGISTER_WORKER` now includes a compact performance summary rather than only
raw host information. The main node records each reported backend as its own
worker-hardware capability object and keeps a running cluster-wide GFLOPS total.

## Directory Layout

- `bootstrap.py`: total project entrypoint
- `supervisor.py`: behavior-tree style startup coordinator
- `config.py`: runtime defaults
- `protocol.py`: mDNS/DNS-SD packet helpers
- `runtime_protocol.py`: framed protobuf wire-format helpers
- `common/`: shared dataclasses and common helpers
- `discovery/`: mDNS discovery and manual fallback
- `adapters/`: platform, firewall, and socket adapters
- `proto/`: protocol documentation, including `proto/superweb_cluster_runtime.proto`
- `main_node/`: main-node runtime logic
- `compute_node/`: worker-side runtime logic
- `compute_node/input matrix/`: fixed benchmark dataset generator and local dataset cache
- `compute_node/performance_metrics/`: CPU/CUDA/Metal benchmark workspace and result summary
  - writes a schema-versioned `result.json` with host metadata, hardware probes,
    backend rankings, a two-phase workload definition, and best measured GFLOPS
- `standalone_model/`: separate network experiments
- `tests/`: automated tests

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

Generate only the fixed benchmark dataset:

```bash
python "compute_node/input matrix/generate.py"
```

## Notes

- The public project/app name is now `superweb-cluster`.
- The runtime protocol and mDNS service naming now use `superweb-cluster`,
  `main node`, and `compute node` terminology consistently.
- `bootstrap.py` is the intended top-level entry. Running lower-level modules
  directly is mainly for development and debugging.
- Benchmark datasets and `result.json` are local machine artifacts and stay
  git-ignored.
- The fixed FMVM benchmark now uses a two-phase workload:
  autotune each candidate config with `3` repeats, then measure the winning
  config with `20` repeats for the reported result.
- The benchmark prefers the executable that matches the current OS and falls
  back to compiling that OS's source when the matching binary is missing or
  older than the source.
- On macOS, the Metal benchmark runner now embeds its compiled `metallib`, so a
  prebuilt runner can be copied to another compatible Mac and run without
  Xcode or the Metal toolchain.
- The checked-in exceptions are the Windows benchmark executables so a fresh
  Windows checkout can run the benchmark backends without rebuilding first.
- Metal artifacts are compiled locally on macOS and remain git-ignored.
