# Home Cluster (Kickoff Version)

## Overview

This repository contains the current kickoff version of Home Cluster. The
implementation now covers discovery, runtime registration, worker heartbeat
tracking, and a minimal client request/response channel so we can validate the
core LAN control plane before building scheduling and execution logic.

The current version implements:

- OS detection for Windows, Linux, macOS, and WSL
- privilege detection and a Windows elevation hook
- firewall adapter entry points, with a real Windows implementation
- mDNS-based home scheduler discovery
- TCP runtime sessions between the home scheduler and home computers
- protobuf wire-format runtime messages for workers and clients
- worker registration with hardware profile upload
- scheduler heartbeat plus `HEARTBEAT_OK` acknowledgement handling
- separate home scheduler connection pools for workers and clients
- minimal client join and request/response handling on the scheduler
- a local `performance metrics/` workspace for repeatable CPU/CUDA compute
  benchmarking
- automatic promotion into the home scheduler runtime in `main_node/runtime.py`
  when no home scheduler is found
- manual address fallback when discovery fails
- bootstrap, supervisor, and status-printing scripts

The current version does not implement:

- task dispatch and result aggregation
- job scheduling and execution orchestration
- home computer execution backends
- a dedicated end-user client CLI built on top of `CLIENT_JOIN` /
  `CLIENT_REQUEST` / `CLIENT_RESPONSE`

Out-of-scope modules are present as skeletons so the project structure matches
the implementation plan and can be extended incrementally.

## Directory Layout

- `bootstrap.py`: main entry point
- `supervisor.py`: minimal lifecycle controller
- `config.py`: kickoff configuration defaults
- `protocol.py`: minimal mDNS/DNS-SD packet helpers
- `runtime_protocol.py`: length-prefixed protobuf wire-format helpers for the
  TCP runtime
- `common/`: shared types and runtime state
- `discovery/`: multicast pairing and manual fallback
- `adapters/`: platform, network, and firewall adapters
- `proto/`: protobuf schema documents for cross-language interoperability
- `performance metrics/`: fixed matrix-vector multiplication benchmark suite
  with a fixed 2 GiB dataset plus CPU and CUDA compute backends
- `standalone_model/`: small self-contained network experiments for mDNS, TCP,
  and ZeroMQ throughput checks
- `scripts/`: helper entry points for manual testing
- `tests/`: automated checks for discovery and runtime behavior

## Discovery Model

The kickoff version supports two bootstrap roles:

- `discover`: acts as a `home computer`, sends an mDNS PTR query for
  `_homecluster-hs._tcp.local.`, waits for a home scheduler response, and then
  enters the compute-node TCP runtime by sending `REGISTER_WORKER`
- `announce`: acts as a `home scheduler`, listens for the home scheduler mDNS
  query, replies with the scheduler address, accepts runtime TCP registrations,
  separates workers and clients into different pools, and sends worker
  `HEARTBEAT` messages every 10 seconds

The discover role can fall back to manual text input if the mDNS browse query
times out.

## Runtime Protocol

The current runtime protobuf schema supports these message kinds:

- `REGISTER_WORKER`: sent by a home computer when joining the scheduler runtime
- `REGISTER_OK`: sent by the home scheduler after worker registration succeeds
- `HEARTBEAT`: sent by the home scheduler to connected workers
- `HEARTBEAT_OK`: sent by a worker to acknowledge a heartbeat
- `CLIENT_JOIN`: sent by a client when opening a TCP control session
- `CLIENT_REQUEST`: sent by a client after joining the scheduler runtime
- `CLIENT_RESPONSE`: sent by the scheduler in reply to client join/request

The home scheduler currently retries a heartbeat up to 3 extra times before the
next normal heartbeat interval. If no matching `HEARTBEAT_OK` arrives, that
worker is removed from the worker pool.

## Quick Start

Run bootstrap in discover mode:

```bash
python bootstrap.py --role discover --udp-port 5353 --tcp-port 52020
```

Run bootstrap in announce mode:

```bash
python bootstrap.py --role announce --udp-port 5353 --tcp-port 52020
```

Show current platform and kickoff status:

```bash
python scripts/print_status.py
```

Trigger a one-off multicast send:

```bash
python scripts/multicast_sender.py --udp-port 5353
```

Start a one-off multicast listener:

```bash
python scripts/multicast_receiver.py --udp-port 5353
```

Run the local benchmark suite:

```bash
python "performance metrics/benchmark.py"
```

Generate the fixed benchmark dataset only:

```bash
python "performance metrics/fixed_matrix_vector_multiplication/input matrix/generate.py"
```

## Notes

- The project is implemented with the Python standard library only.
- Discovery uses standard mDNS IPv4 multicast `224.0.0.251:5353`.
- Home scheduler discovery is modeled as a DNS-SD browse query for
  `_homecluster-hs._tcp.local.`.
- Runtime TCP traffic uses protobuf wire format documented in
  `proto/home_cluster_runtime.proto`.
- The `_homecluster-hs._tcp.local.` service type is intentionally short,
  project-specific, and visually distinct so it stands out from generic mDNS
  traffic on shared port `5353`.
- Port `5353` is shared with other mDNS software on the machine, so the
  implementation filters packets by this project-specific service type instead
  of assuming exclusive ownership of the port.
- TCP port `52020` is used by the home scheduler runtime in `main_node/`.
- On Windows, the kickoff firewall helper now creates both inbound and outbound
  UDP rules for the discovery port across all firewall profiles.
- On non-Windows systems, firewall functions are present but intentionally
  report "not implemented" in this kickoff version.
- Cleanup is best-effort. Windows firewall cleanup can also be called manually
  with `python scripts/cleanup_firewall.py`.
- The benchmark workspace writes `performance metrics/result.json` with one
  best entry per backend plus a backend `ranking`, and auto-builds local
  backend binaries under backend-specific `build/` directories. The Windows
  CPU/CUDA benchmark `.exe` files are checked in intentionally; other generated
  benchmark artifacts remain local-only.
- The benchmark uses a fixed input dataset:
  `A[16384,32768]` float32 and `x[32768]`.
- Benchmark datasets under `performance metrics/.../input matrix/generated/`
  and benchmark reports are local generated artifacts and are not meant to stay
  tracked in git.
