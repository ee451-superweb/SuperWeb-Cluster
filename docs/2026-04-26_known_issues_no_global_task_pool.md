# Known issue: lack of a global task pool causes multi-client and failover performance bottlenecks — 2026-04-26

This document records a core performance defect in the project's current architecture that has been **identified but intentionally not fixed within this assignment cycle**, for reference in later iterations or follow-up work. The issue does not surface in functional tests (every request still returns the correct result), but in multi-client concurrency or worker-failure scenarios it noticeably degrades end-to-end latency.

## 1. Symptom

1. **No global scheduling arbitration when multiple clients submit concurrently.** When two clients submit requests at the same time, main_node spins up an independent thread per client and each one walks its own `dispatch -> ThreadPoolExecutor.submit -> wait` flow. The dispatcher is stateless across requests, so it can hand the same time window on the same worker to two different requests. There is no queueing, no admission control, and no backpressure.
2. **Failover retry slices cannot start until the survivor finishes its current slice.** Within a single request, when a worker fails the re-partitioned retry slice is immediately submitted to the executor, but actual execution is blocked by the connection lock on the worker side, so the user-visible behavior looks like "wait for everyone to finish."

## 2. Root cause (important: counterintuitive)

The two symptoms are **two faces of the same root cause** — but the root cause is not "missing task pool" itself. It is the absence of a **scheduler with a global view of all queued and in-flight slices**.

Concretely on the failover path, the dispatcher loop at [request_handler.py:369-372](../main_node/request_handler.py#L369-L372) uses `concurrent.futures.wait(..., return_when=FIRST_COMPLETED)`, which **wakes up the moment the first future completes (success or failure)**. The failed slice is immediately re-partitioned and `executor.submit`'d at [request_handler.py:413-430](../main_node/request_handler.py#L413-L430) — the dispatcher layer never waits for everyone.

The real block is the `Connection.task_lock` on the worker side. The comment on `_submit_retry` itself ([request_handler.py:328-347](../main_node/request_handler.py#L328-L347)) already spells it out:

> When the survivor's `task_lock` is already held by an in-flight original slice, the retry will block inside `run_worker_task_slice` until that lock releases.

In other words: each worker can only run one slice at a time. After the retry is re-distributed across the N-1 survivors proportional to their original GFLOPS, all N-1 of them are still running their original slices, so every retry queues up — **the appearance is "wait for everyone to finish," the mechanism is N-1 independent per-worker waits happening concurrently**.

The multi-client case is the same story: two independent `ClientRequestHandler` instances both call into the dispatcher, and neither one can see which workers the other is currently using.

## 3. Why "just add a queue" doesn't fix it

This is the counterintuitive point this document wants to highlight. If we only add a global FIFO queue in front of main_node to receive client requests, then:

- Multi-client no longer fights over workers (the queue dequeues in order, so each dispatch sees an exclusive worker set)
- Failover is still slow — retry slices are still pushed to specific survivors, still blocked on `task_lock`

To solve both at once, **the correct shape is "global stealable queue + worker-side pull"**:

| Change | Effect |
|--------|--------|
| main_node owns a global slice queue (both original and retry slices) | multi-client requests fan out naturally; retries no longer need to pre-select a survivor |
| worker side switches from "being pushed" to "I'm idle, give me the next one" | retries land on whichever worker is currently freest, no need for `_build_retry_assignments` to re-distribute by GFLOPS |
| `_run_assignments_with_failover` collapses to "submit + wait for result aggregation" with no worker selection | dispatcher simplifies dramatically |

By-product: a large chunk of the `original_weights`-weighted partitioning logic in `_build_retry_assignments` (around [request_handler.py:230-280](../main_node/request_handler.py#L230-L280)) can be deleted.

## 4. Why this was not implemented in this cycle

- The change moves the worker side from a push model to a pull model, which requires modifying the wire protocol, the worker main loop, and the meaning of heartbeat. That is beyond the engineering budget of this assignment.
- The current push model already covers every evaluation scenario for functional correctness; the performance loss is only significant under high-frequency multi-client submission or high worker failure rate, neither of which is the focus of this assignment's evaluation (single-client end-to-end correctness + heterogeneous backend scheduling).
- A real fix also requires deciding whether a worker should ever execute multiple slices concurrently (which involves GPU/CPU resource-partitioning policy). That is an independent design decision and should not be bundled with "add a queue."

## 5. Related code locations

| Concern | File | Line |
|---------|------|------|
| Per-request executor, no cross-request view | [main_node/request_handler.py](../main_node/request_handler.py) | 282-438 |
| `FIRST_COMPLETED` wait loop | [main_node/request_handler.py](../main_node/request_handler.py) | 369-372 |
| Retry submit blocks on worker `task_lock` | [main_node/request_handler.py](../main_node/request_handler.py) | 328-347 |
| Retry slices re-partitioned by original GFLOPS | [main_node/request_handler.py](../main_node/request_handler.py) | 230-280 |
| Multi-client accept loop (entry side is already multi-client clean) | [main_node/connection_service.py](../main_node/connection_service.py) | 208-250 |
| Per-client service thread | [main_node/connection_service.py](../main_node/connection_service.py) | 163-170 |

## 6. Suggested minimum migration steps

1. Add a `GlobalSliceQueue` in main_node, enqueueing `(request_id, slice)` with a priority field so retry slices can jump the queue.
2. Add wire-protocol messages `WORKER_REQUEST_WORK` / `WORKER_NO_WORK_AVAILABLE`.
3. Change the worker main loop to: finish a slice -> send `WORKER_REQUEST_WORK` -> block waiting for the next one -> execute on receipt.
4. Collapse `_run_assignments_with_failover` to: enqueue every slice -> wait for all results keyed by `task_id` to come back -> aggregate.
5. Keep heartbeat as a liveness signal only; it no longer carries the implicit "ready to accept new work" semantics.

If this direction is pursued in the future, it is worth introducing an end-to-end benchmark at the same time: simulate 3 clients × continuous submission and compare tail latency (p95/p99) and per-worker utilization curves before and after the change.
