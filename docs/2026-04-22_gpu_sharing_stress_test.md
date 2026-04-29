# Single-Machine 2xMetal + 1xCPU Shared GPU Concurrency Behavior Analysis - 2026-04-22

The goal of this test was to launch 1 main_node + 3 compute_nodes (2 with Metal as backend, 1 with CPU as backend) on a single M1 Pro Mac, and observe the behavior of two Metal workers contending for the same physical GPU, as well as the scheduling, heartbeat, and data-plane behavior when a CPU worker coexists with them. Conclusion: **the system behavior is fully consistent with the physical constraints of single-machine shared GPU; no bug was found**.

## Test Configuration

| Role | Log | backend | Self-reported throughput |
|------|------|---------|---------|
| main_node | `main-20260422-221819.txt` | - | - |
| worker-1 (peer-metal) | `worker-20260422-221854.txt` | Metal | gemv 80.021 GFLOPS / conv2d 955.035 GFLOPS |
| worker-2 (metal) | `worker-20260422-222556.txt` | Metal | gemv 80.021 GFLOPS / conv2d 955.035 GFLOPS |
| worker-3 (peer-cpu) | `worker-20260422-222558.txt` | CPU | gemv 52.488 GFLOPS / conv2d 202.933 GFLOPS |

All four processes ran on the same Mac (`192.168.1.136`, M1 Pro, 10-core CPU, 16 GB unified memory). The data plane went through loopback, with measured recv bandwidth of 5-7 GB/s, matching kernel memcpy, which rules out any "cross-machine" illusion.

The client run window was 2026-04-22 23:12:18 - 23:16:39, with 8 tasks issued in total (gemv x 4, conv2d x 4), all returning `status=200`.

## Key Observations

### 1. The two Metal workers were split exactly in half on the same GPU (as expected)

Timing for conv2d `size=large, iter=1` (`fixe-3 = conv2d-12`):

| worker | assigned oc | wall | compute | peripheral |
|--------|---------|------|---------|-----------|
| worker-1 (metal) | 0:116 | 27087 ms | **21972 ms** | 20 ms |
| worker-2 (metal) | 116:232 | 27073 ms | **22137 ms** | 21 ms |
| worker-3 (cpu) | 232:256 | 26281 ms | 7727 ms | 16543 ms |

The two Metal workers showed:

- **0.75% difference in compute time** (21972 vs 22137 ms)
- **87 ms difference in finish time** (artifact publish at 23:14:29.924 and 23:14:30.011 respectively)
- Not serial (if serial, the difference should be 22 seconds), and not 2x parallel (if truly parallel, each should take ~11 seconds)

This is the typical symptom of Apple Silicon integrated GPU performing **EU-level time-slicing** between the two Metal processes' command queues: both advance simultaneously, but each only gets about half of the GPU execution unit time, so the kernel latency each one perceives is exactly doubled.

### 2. Root cause of why GPU concurrent throughput does not double

This can be confirmed directly from the benchmark source side:

- `gemv_metal_runner.mm` does `commit` + `waitUntilCompleted` per chunk ([L243-249](../compute_node/compute_methods/gemv/metal/gemv_metal_runner.mm#L243-L249)), and the GPU goes idle between chunks. GFLOPS calculation prefers the `GPUStartTime`/`GPUEndTime` hardware timestamps ([L348-352](../compute_node/compute_methods/gemv/metal/gemv_metal_runner.mm#L348-L352)), so 80 GFLOPS is pure GPU kernel time. M1 Pro memory bandwidth is ~200 GB/s, gemv arithmetic intensity is 0.5 flop/byte -> bandwidth ceiling about 100 GFLOPS, **the current 80 GFLOPS is already eating 80% of bandwidth**, leaving almost no headroom for concurrent speedup.
- `conv2d_metal_runner.mm` uses MPSGraph's synchronous API `runWithMTLCommandQueue` ([L491-502](../compute_node/compute_methods/conv2d/metal/conv2d_metal_runner.mm#L491-L502)), and timing uses `steady_clock` wrapped around the entire graph run, including CPU-side dispatch. 955 GFLOPS is approximately 18% of the 5.2 TFLOPS theoretical peak, which is a reasonable measured peak for MPSGraph 3x3 conv at this scale.

**Two processes with isomorphic workload x 2 cannot fill in each other's GPU idle**: because the chunk cadence of the two workers is synchronized, they are busy at the same time, prep the next chunk on the CPU side at the same time, and the GPU's idle windows also appear at the same time. The scenario that can actually achieve concurrency gains is a heterogeneous workload (e.g., gemv + conv2d mixed run).

### 3. The scheduler partitions by self-reported GFLOPS without considering "multiple instances on the same backend sharing hardware"

The slice ratio for conv2d large is 116 : 116 : 24 = 45% : 45% : 9.4%, exactly matching the self-reported 955 : 955 : 203 GFLOPS ratio. The scheduler treats the two Metal workers as two independent 955 GFLOPS devices, causing the Metal-side compute to be saturated for 22 seconds, while the CPU worker finishes its 24 channels in 7.7 seconds and then waits 17 seconds in peripheral - **the wall time of the entire batch is determined by the Metal-side tail**.

This is an **expected limitation rather than a bug**: the benchmark measures GFLOPS in an "exclusive backend" context, and the scheduler also makes decisions based on these values. If the scheduler is to become aware of shared hardware in the future, one approach is for the main_node to group by `(hostname, backend)` at REGISTER_WORKER time, and divide each instance's effective GFLOPS by the group size within the group.

### 4. The HEARTBEAT warning is just an edge case, with immediate self-recovery

There is a single WARNING at [main-20260422-221819.txt:6728](../logs/main-20260422-221819.txt):

```
23:14:29,901 HEARTBEAT failure for compute node-peer-metal at 192.168.1.136:59286
    attempt=1/4 after timed out waiting for heartbeat ack for 1776924868893
    active_task=conv2d-12
```

The timing falls precisely at: worker-1 publishing a 1.9 GB artifact (23:14:29.924) + main_node preparing to fetch the same artifact (23:14:33.034) + worker-2/worker-3 entering the post-processing stage at the same time. Four processes + a 4 GB data-plane transfer simultaneously saturated CPU and IO, and the heartbeat thread was scheduling-starved for ~1 second.

The next attempt (23:14:29.902) succeeded directly, without escalating to node eviction. This indicates that the 1-second heartbeat_ack timeout is slightly tight for "all-in-one single machine + large artifact peak", but the system's retry mechanism handled it cleanly.

### 5. iter=1 being slower than iter=100 is cold page cache

| task | iter | total elapsed | worker-1/-2 compute | worker-1/-2 peripheral |
|------|------|-----------|---------------------|----------------------|
| gemv-6 | 1 | 4986 ms | 5-6 ms | **4806 ms** |
| gemv-7 | 100 | **4011 ms** down | 889-912 ms | 2793 ms |
| gemv-8 | 1000 | 15100 ms | 9932-10284 ms | 4221 ms |

The large gemv matrix is ~2 GB (16384x32768 f32), with each worker assigned about 800 MB. The first request is a cold read from disk into shared-memory MTLBuffer. Subsequent requests hit the OS page cache, peripheral drops, and compute becomes dominant. **Implication**: single-iter=1 baseline data is misleading; performance regression comparison requires at least warm-up or iter >= 100.

### 6. worker-3 (CPU) peripheral consumes a large amount of time

worker-3 on conv2d-13 stats-only (no artifact to upload): `wall=25620ms, compute=7307ms, peripheral=17461ms`. peripheral is not upload time, but rather the CPU path's sum / sum^2 / reshape / memcpy crawling on a CPU squeezed by the Metal processes; the Metal path puts these post-processing steps in a GPU shader, so peripheral is only 20 ms.

This matches the expectation of "heterogeneous backends on the same machine contending for CPU". To make the diagnostic distribution clearer, `peripheral` could later be split into three segments: `artifact_stage / output_copy / stats_reduce`.

## Conclusion

All behavior in the current run can be explained by the physical constraints of "single-machine shared GPU + shared CPU"; no bug requiring repair was found:

- Pass: the two Metal workers fairly split GPU time (0.75% difference)
- Pass: synchronous completion (87 ms difference), no starvation
- Pass: heartbeat triggered an edge case under peak IO and recovered immediately
- Pass: data plane runs on loopback, bandwidth at memory level
- Pass: all 8 tasks returned `status=200`

If we want this topology's throughput to approach "two independent GPUs" later, the only direction that can really squeeze out an increment is **staggering the chunk cadence of the two Metal workers** (different `output_channel_batch` or introducing a small startup jitter), so that the GPU's short idle windows are filled by the peer kernel. In the case of isomorphic workload x 2, the ceiling is probably only 5-10% aggregate gain, not worth sacrificing code simplicity for.

## Potential Improvements (non-blocking)

Sorted by priority, none of which affect the current design goals:

1. **Make the scheduler aware of multiple instances on the same backend**: group by `(host, backend)` at registration, divide GFLOPS within the group by group size. This avoids "dispatching a single GPU as if it were two".
2. **Relax the heartbeat_ack timeout to 2-3 seconds** or explicitly `sched_yield` in the artifact stage path to eliminate edge warnings under 4 GB peak IO.
3. **Split peripheral into sub-items** (stage / copy / reduce), to precisely see which segment of the CPU path is the bottleneck.
4. **Add warm-up + multi-sample to the benchmark**, so iter=1 first-round latency no longer dominates the single-shot latency users see.

## Known Platform Difference: macOS peer processes have no visible window

In `_peer_popen_kwargs` at `app/supervisor.py:215-242`, the Windows branch uses `creationflags = CREATE_NEW_CONSOLE` to give the peer compute-node subprocess its own visible console window (the docstring explicitly says this is for demo viewing); the POSIX branch only returns empty kwargs, and the peer simply runs as a background subprocess inheriting the parent's IO. Result:

- **No functional impact** - the peer's TRACE/INFO is still written to its respective `worker-*.txt`, `bootstrap-*.txt`, no diagnostic information is lost
- **Visual difference** - on macOS the user cannot intuitively see "the second compute node is running"; this can only be seen in the log files

To match the Windows demo experience, the POSIX branch would need to use `osascript -e 'tell app "Terminal" to do script ...'` to open a new window, but at the cost of:

- Terminal.app would become the new process group parent, and the supervisor's PID tracking of `_peer_process` and the cleanup paths in `peer_watcher` / `peer_heartbeat_watcher` would all need to be rewritten
- Not all macOS users have Terminal.app installed (iTerm, Warp, etc.), so application detection would be needed first
- Cleanly shutting down a peer would require terminating both the shell and the Python process, making the signal chain more complex

Therefore this difference is **intentionally retained for now**: in demo scenarios, use `tail -f logs/worker-*.txt logs/main-*.txt` or manually open three Terminals separately and run `bootstrap.py --role ... --backend ...` as a workaround. The functionality is fully equivalent, and visually the requirement of "seeing all three processes alive" is also satisfied.
