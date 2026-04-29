# Full protocol walkthrough: Client ↔ Main ↔ Worker — 2026-04-19

SuperWeb-Cluster three-party protocol panoramic guide (Client ↔ Main ↔ Worker).
Timestamp: 2026-04-19. Originally drafted alongside several pre-implementation
plans that have since shipped and been removed; see [`archived_plans.md`](archived_plans.md)
for what those plans became. For the current bootstrap behavior tree, message
catalog, and CLI options, see [`technical-detail.md`](technical-detail.md).

How to read this: this document does not replace the spec of any single protocol. It draws out how the discovery / control / data planes interleave inside a real request, so a newcomer does not have to read the code three times to answer "right now, which message on which line is the client waiting for?".

> **Port-number note.** This walkthrough was authored when the runtime
> defaults were `:31000` (control) and `:45000` (data). The current canonical
> defaults defined in [`core/constants.py`](../core/constants.py) are
> `:52020` and `:52021`. The old numbers below are kept verbatim for
> historical accuracy of the dated walkthrough; mentally substitute the
> current values when reading against today's code.


## 0. Topology overview

```text
                   UDP 5353 (mDNS discovery plane)
                  ┌───────────────────────┐
                  │                       │
                  ▼                       ▼
           ┌──────────────┐        ┌─────────────┐
           │  Client 1    │        │  Worker 1   │
           └──────┬───────┘        └──────┬──────┘
                  │ TCP                   │ TCP
                  │ control (protobuf)    │ control (protobuf)
                  │ data    (SWAD frame)  │ data    (SWAD frame)
                  ▼                       ▼
               ┌──────────────────────────────────┐
               │             Main Node            │
               │  ┌───────────────────────────┐   │
               │  │ control server  (31000)   │   │
               │  │ data server     (45000)   │   │
               │  │ discovery announcer (UDP) │   │
               │  │ ArtifactManager           │   │
               │  │ Registry / Dispatcher     │   │
               │  └───────────────────────────┘   │
               └──────────────────────────────────┘
                          ▲              ▲
                          │ TCP control  │ TCP control
                          │ TCP data     │ TCP data
                   ┌──────┴──────┐┌──────┴──────┐
                   │  Worker 2   ││  Worker 3   │
                   └─────────────┘└─────────────┘
```

  - Main is open to both clients and workers. It is the only node in the topology that simultaneously plays the roles of "control server" and "data server".
  - Client only knows about main; it never talks directly to a worker.
  - Worker also only knows about main; it does not talk laterally to other workers.
  - All TCP connections are **actively initiated by the outer-ring nodes**; main never connects back outward.


## 1. Three-plane quick-reference table

```text
 ┌──────────────┬────────────────────┬────────────────────┬────────────────────┐
 │              │ discovery plane    │ control plane      │ data plane         │
 ├──────────────┼────────────────────┼────────────────────┼────────────────────┤
 │ transport    │ UDP / mDNS 5353    │ TCP (long-lived,   │ TCP (short-lived,  │
 │              │                    │   31000)           │   45000)           │
 │ encoding     │ DNS wire format    │ length-prefixed    │ fixed header +     │
 │              │ (PTR/SRV/TXT/A)    │ protobuf           │ variable tail      │
 │              │                    │                    │ (SWAD MAGIC)       │
 │ initiator    │ any node looking   │ client / worker    │ client / worker    │
 │              │ for main           │                    │                    │
 │ duration     │ a single ms-scale  │ entire session     │ one artifact       │
 │              │ Q&A                │ lifetime           │ exchange           │
 │ typical use  │ resolve main's     │ register / heart-  │ push weight /      │
 │              │ ip:port            │ beat / request     │ pull result        │
 │ header type  │ DNS header         │ MessageKind (enum) │ u8 message_type    │
 │ corresponding│ discovery.py       │ control_plane_     │ data_plane_codec   │
 │ codec        │                    │   codec.py         │   .py              │
 └──────────────┴────────────────────┴────────────────────┴────────────────────┘
```

The control-plane TCP port (31000) carries **all** protobuf messages, both client and worker; message types are distinguished by the protobuf `MessageKind` rather than by separate ports.
The data-plane TCP port (45000) carries only artifact-related SWAD frames; once the control plane receives a request, it tells the peer the host:port to dial, so artifacts are never multiplexed onto the control socket.


## 2. Control-plane message dictionary (protobuf over TCP)

All control-plane messages are defined in `wire/proto/superweb_cluster_runtime.proto` and discriminated by `oneof kind`. On the wire one message is:

```text
    [ uint32 length ][ RuntimeMessage (protobuf) ]
    RuntimeMessage.kind = <specific message>
```

`MessageKind` enum (see `wire/internal_protocol/control_plane_codec.py`):

```text
 ┌────────────────────┬──────────┬─────────┬─────────────────────────────┐
 │ MessageKind         │ Sender    │ Receiver │ Main fields                │
 ├────────────────────┼──────────┼─────────┼─────────────────────────────┤
 │                                EXTERNAL (client ↔ main)               │
 ├────────────────────┼──────────┼─────────┼─────────────────────────────┤
 │ CLIENT_JOIN         │ client    │ main     │ client_name                │
 │ CLIENT_RESPONSE*    │ main      │ client   │ status_code, client_id,    │
 │  (for join)         │           │          │ ...                        │
 │ CLIENT_REQUEST      │ client    │ main     │ request_id, method,        │
 │                     │           │          │ size, object_id,           │
 │                     │           │          │ iteration_count,           │
 │                     │           │          │ request_payload (oneof):   │
 │                     │           │          │   - GemvRequestPayload     │
 │                     │           │          │   - Conv2dRequestPayload   │
 │                     │           │          │     - upload_size_bytes,   │
 │                     │           │          │       upload_checksum      │
 │ CLIENT_REQUEST_OK   │ main      │ client   │ task_id, method, size,     │
 │                     │           │          │ accepted_timestamp_ms,     │
 │                     │           │          │ --- data-plane fields ---  │
 │                     │           │          │ upload_id, download_id,    │
 │                     │           │          │ data_endpoint_host,        │
 │                     │           │          │ data_endpoint_port         │
 │ CLIENT_RESPONSE     │ main      │ client   │ status_code, task_id,      │
 │  (for request)      │           │          │ elapsed_ms,                │
 │                     │           │          │ response_payload (oneof),  │
 │                     │           │          │ result_artifact,           │
 │                     │           │          │ timing (ResponseTiming)    │
 │ CLIENT_INFO_REQUEST │ client    │ main     │ client_id, timestamp_ms    │
 │ CLIENT_INFO_REPLY   │ main      │ client   │ has_active_tasks,          │
 │                     │           │          │ active_task_ids, ...       │
 │ HEARTBEAT           │ main      │ client   │ unix_time_ms (server clock)│
 │ HEARTBEAT_OK        │ client    │ main     │ heartbeat_unix_time_ms     │
 ├────────────────────┼──────────┼─────────┼─────────────────────────────┤
 │                                INTERNAL (main ↔ worker)               │
 ├────────────────────┼──────────┼─────────┼─────────────────────────────┤
 │ REGISTER_WORKER     │ worker    │ main     │ node_name, hardware,       │
 │                     │           │          │ performance                │
 │ REGISTER_OK         │ main      │ worker   │ main_node_ip, port,        │
 │                     │           │          │ assigned node_id           │
 │ HEARTBEAT           │ main      │ worker   │ unix_time_ms               │
 │ HEARTBEAT_OK        │ worker    │ main     │ heartbeat_unix_time_ms,    │
 │                     │           │          │ active_task_ids,           │
 │                     │           │          │ node_status,               │
 │                     │           │          │ completed_task_count       │
 │ TASK_ASSIGN         │ main      │ worker   │ task_id, request_id,       │
 │                     │           │          │ method, size,              │
 │                     │           │          │ transfer_mode,             │
 │                     │           │          │ artifact_id,               │
 │                     │           │          │ task_payload (oneof):      │
 │                     │           │          │   - GemvTaskPayload        │
 │                     │           │          │     (row slice + vector)   │
 │                     │           │          │   - Conv2dTaskPayload      │
 │                     │           │          │     (output-channel slice, │
 │                     │           │          │      weight_artifact ←     │
 │                     │           │          │      points to main's      │
 │                     │           │          │      data plane)           │
 │ TASK_ACCEPT         │ worker    │ main     │ task_id, status_code       │
 │ TASK_FAIL           │ worker    │ main     │ task_id, error_message     │
 │ TASK_RESULT         │ worker    │ main     │ task_id, status_code,      │
 │                     │           │          │ result_payload (oneof),    │
 │                     │           │          │ result_artifact            │
 │ ARTIFACT_RELEASE    │ main      │ worker   │ task_id, artifact_id       │
 │ WORKER_UPDATE       │ worker    │ main     │ performance (refreshed     │
 │                     │           │          │ when idle)                 │
 └────────────────────┴──────────┴─────────┴─────────────────────────────┘
```

\* `CLIENT_RESPONSE (for join)` and `CLIENT_RESPONSE (for request)` are the same `MessageKind`; they are distinguished by fields other than `status_code`. The one for join is just a simple acknowledgement (no `response_payload`).


## 3. Data-plane message dictionary (SWAD frames, TCP)

All data-plane messages share the same header prefix:

```text
    [ 4s MAGIC="SWAD" ][ u8 version=1 ][ u8 message_type ][ ...type-specific... ]
```

About `MAGIC="SWAD"`:
  - Meaning: an acronym for "SuperWeb Artifact Data".
  - Purpose: lets the data-plane receiver decide at a glance, by reading only the first 4 bytes, "this is our protocol frame", quickly discarding:
    - HTTP/TLS packets that wandered in via a wrong port (they will not start with `"SWAD"`)
    - half-packets / garbled bytes
    - old clients left over from a future protocol generation
  - Why 4 bytes: this is the standard convention for binary protocols (compare PNG's `"\x89PNG"`, ZIP's `"PK\x03\x04"`, ELF's `"\x7FELF"`). Enough to disambiguate but not wasteful — a `CHUNK` frame may be sent tens of thousands of times for a large file, and every extra byte in the magic compounds into overhead.
  - Why not spell out the full name: the magic is not for humans to read, it is for parsers to peek at; if you want self-documenting code, adding a comment is cheaper than spending bytes.

Message types (as of 2026-04-19):

```text
 ┌──────────────────────┬─────┬──────────┬────────────┬────────────────────────┐
 │ message_type          │ Val │ Sender    │ Receiver   │ Main fields            │
 ├──────────────────────┼─────┼──────────┼────────────┼────────────────────────┤
 │ DOWNLOAD_REQUEST      │  1  │ the puller│ the owner   │ artifact_id            │
 │   (formerly REQUEST)  │     │ (client   │ (main /     │                        │
 │                       │     │  or worker│  worker)    │                        │
 │                       │     │  or main) │             │                        │
 │ INIT                  │  2  │ owner     │ puller      │ size_bytes, chunk_size,│
 │                       │     │           │             │ checksum, content_type │
 │ CHUNK                 │  3  │ sender    │ receiver    │ offset, data           │
 │ END                   │  4  │ sender    │ receiver    │ size_bytes (length     │
 │                       │     │           │             │ check)                 │
 │ ERROR                 │  5  │ either    │ peer        │ message                │
 │ DELIVER               │  6  │ client    │ main        │ upload_id, size_bytes, │
 │  (added 2026-04-19,   │     │           │             │ checksum, content_type │
 │   push-upload entry)  │     │           │             │                        │
 └──────────────────────┴─────┴──────────┴────────────┴────────────────────────┘
```

The two typical "single connections" on the data plane:

```text
 ┌─────────────────────────────────────────────────────────────────────────┐
 │ A. Pure pull (existing mode: worker pulls weight, client pulls result)  │
 │                                                                          │
 │     puller  --DOWNLOAD_REQUEST(artifact_id)-->  owner                   │
 │     puller  <----INIT(size,checksum,...)-----   owner                   │
 │     puller  <----CHUNK(offset,data) x N------   owner                   │
 │     puller  <----END(size)-------------------   owner                   │
 │     (connection closes)                                                 │
 └─────────────────────────────────────────────────────────────────────────┘
```

```text
 ┌─────────────────────────────────────────────────────────────────────────┐
 │ B. Push-then-pull (2026-04-19 new path, only for client->main conv2d    │
 │    full)                                                                │
 │                                                                          │
 │     client  --DELIVER(upload_id,size,checksum)-->  main                 │
 │     client  --CHUNK(offset,data) x N------------>  main                 │
 │     client  --END(size)-------------------------->  main                │
 │                    (same TCP socket stays open)                         │
 │     client  --DOWNLOAD_REQUEST(download_id)----->  main                 │
 │     client  <----INIT(size,checksum,...)-------- main                   │
 │     client  <----CHUNK x N----------------------- main                  │
 │     client  <----END----------------------------- main                  │
 │     (connection closes)                                                 │
 └─────────────────────────────────────────────────────────────────────────┘
```

Note: the main ↔ worker data plane is always mode A (pull model); there is no place for `DELIVER`.


## 4. Discovery-plane message dictionary (UDP / mDNS)

Short enough to list separately:

```text
 ┌────────────┬──────────┬────────┬────────────────────────────────────────┐
 │ Message     │ Sender    │ Trans- │ Payload                               │
 │             │           │ port   │                                       │
 ├────────────┼──────────┼────────┼────────────────────────────────────────┤
 │ DISCOVER    │ client /  │ UDP    │ mDNS PTR query: MDNS_SERVICE_TYPE     │
 │             │ worker    │ mcast  │                                       │
 │ ANNOUNCE    │ main      │ UDP    │ mDNS response: PTR + SRV(port) +      │
 │             │           │ mcast/ │ TXT(role,node) + A(host)              │
 │             │           │ ucast  │                                       │
 └────────────┴──────────┴────────┴────────────────────────────────────────┘
```

After one `DISCOVER` / `ANNOUNCE` round trip, the initiator obtains
    `AnnouncePayload(host, port, node_name)`
and from then on every interaction goes over TCP. The discovery plane drops out at this point, unless the TCP connection breaks and rediscovery is needed.


## 5. Phase I: startup negotiation (background phase, no client involved)

In time order, left-to-right across worker / main:

```text
 -------------+-----------------------------------------+-------------
  WORKER 1    │                MAIN                     │   external
 -------------+-----------------------------------------+-------------
  startup     │                                         │
  UDP send DISCOVER ------------------> UDP recv        │
              │ <------- UDP send ANNOUNCE -- (host,port)│
  TCP connect ----------------------->  TCP accept      │
  REGISTER_WORKER(name,hardware,perf) ------------------>│
              │ <------ REGISTER_OK(node_id) ----------│
              │                                         │
  <long-lived connection kept>                          │
              │ <HEARTBEAT every N seconds>             │
              │ <-- HEARTBEAT_OK(active_task_ids) --   │
  When idle, periodically:                              │
  WORKER_UPDATE(performance) ----------------------> │
 -------------+-----------------------------------------+-------------
```

The discovery plane is used only on **first acquisition of main's address**. After that the worker keeps this TCP control connection as a long-lived link, only re-running `DISCOVER` if it errors out.
Heartbeats are bidirectional: `main → worker` is active liveness probing, while `worker → main` (piggybacked on `HEARTBEAT_OK`) simultaneously reports `active_task_ids` / `node_status`.


## 6. Phase II: client joins

```text
 -------------+---------------------------------------------+-------------
  CLIENT 1    │                    MAIN                     │
 -------------+---------------------------------------------+-------------
  startup     │                                             │
  UDP DISCOVER -----------------------------> UDP recv      │
              │ <------- UDP ANNOUNCE ---- (host, port)    │
  TCP connect ------------------------------> TCP accept    │
  CLIENT_JOIN(client_name) -------------------------------->│
              │ <-- CLIENT_RESPONSE(status=OK, client_id) -│
              │ (from now on the client may issue any number│
              │  of CLIENT_REQUEST)                         │
 -------------+---------------------------------------------+-------------
```

After `CLIENT_JOIN`, main starts sending `HEARTBEAT` on this connection, and the client replies with `HEARTBEAT_OK`. Within its own request flow the client also uses `CLIENT_INFO_REQUEST` / `CLIENT_INFO_REPLY` to ask "is my task still alive?", to avoid silent death of the control channel.


## 7. Phase III: GEMV request (small payload, does not exercise the data plane)

GEMV's vector and output are both small (KB scale); the entire exchange fits in inline protobuf fields on the control plane, and the data plane stays idle the whole time.

```text
 -------------+---------------------+----------------------+-------------
  CLIENT 1    │        MAIN         │    WORKER n (n of)   │
 -------------+---------------------+----------------------+-------------
  CLIENT_REQUEST                                                          │
  (method=gemv, vector_data inline) -->                                  │
              │ allocate_data_plane_endpoints                            │
              │   - sees method=gemv -> upload_id=""/download_id=""      │
              │ <- CLIENT_REQUEST_OK(task_id, upload_id="",              │
              │        download_id="", data_endpoint_* still filled but  │
              │        client will not connect)                          │
              │                                                           │
              │ partition: request split into N row-slices, one per      │
              │ worker                                                    │
              │ TASK_ASSIGN(slice_i, row_start, row_end,                  │
              │             vector_data inline) ------> worker i         │
              │ <-- TASK_ACCEPT --------------------------               │
              │ <-- TASK_RESULT(output_vector inline) - (N times)        │
              │ aggregator stitches the N row-slices                      │
              │ <- CLIENT_RESPONSE(gemv_payload.output_vector inline)    │
              │      elapsed_ms, timing (dispatch/task_window/aggregate) │
 -------------+---------------------+----------------------+-------------
```

There are no SWAD frames on this path, and no `artifact_id`.


## 8. Phase IV: Conv2D full (full artifact output, the most complex path)

This path simultaneously exercises:
  - control plane (client ↔ main, main ↔ worker)
  - data plane (client → main pushes weight; worker → main pulls weight; client ← main pulls result)

Field legend:
  `ctrl` = control-plane frame, `data` = data-plane frame, `(*)` marks a message that gets multiplied by N when sliced.

```text
 --------------+----------------------+----------------------+-------------
  CLIENT 1     │        MAIN          │      WORKER n        │
 --------------+----------------------+----------------------+-------------
  (precondition: client locally generate_default_weight_bytes + sha256)  │
  ctrl CLIENT_REQUEST                                                    │
   method=conv2d                                                          │
   stats_only=false                                                       │
   upload_size_bytes=32768                                                │
   upload_checksum="abc..."                                               │
               ----------------->                                        │
               │ allocate_data_plane_endpoints                           │
               │   - register_upload_slot(upload_id, 32768, "abc...")    │
               │   - download_id = "req-7-download-<nonce>"              │
               │   - Future<path>                                        │
               │ <- ctrl CLIENT_REQUEST_OK                               │
               │       task_id, upload_id, download_id,                  │
               │       data_endpoint_host=10.0.0.5, _port=45000          │
                                                                          │
  (control socket waits for CLIENT_RESPONSE)                             │
  (opens a new TCP to 10.0.0.5:45000, background thread pushes weight)   │
                                                                          │
  data DELIVER(upload_id=..., size=32768, checksum="abc...")             │
               ----------------->                                        │
  data CHUNK(0, 16384) (*)                                               │
               ----------------->                                        │
  data CHUNK(16384, 16384)                                               │
               ----------------->                                        │
  data END(32768)                                                        │
               -----------------> main _consume_upload_slot              │
               │     - verify size, verify sha256                        │
               │     - Future.set_result(<temp_path>)                    │
                                                                          │
               │ request_handler.upload_future.result()                  │
               │  -> obtains temp_path                                   │
               │ register_existing_file(temp_path,                       │
               │     artifact_id = "req-7-weights")  (for worker pull)   │
               │                                                          │
               │ partition: split by output-channel into N slices        │
               │ ctrl TASK_ASSIGN(task_id_i,                             │
               │     method=conv2d,                                      │
               │     start_oc/end_oc=...                                 │
               │     transfer_mode=ARTIFACT_REQUIRED                     │
               │     artifact_id="req-7-weights",                        │
               │     task_payload.Conv2dTaskPayload{                     │
               │       weight_artifact=ArtifactDescriptor{               │
               │         artifact_id="req-7-weights",                    │
               │         transfer_host=main_host,                        │
               │         transfer_port=main_data_port,                   │
               │         size_bytes=32768, checksum="abc..."             │
               │       }                                                 │
               │     }) ------------------------> worker_n               │
               │                                  (repeated per worker)  │
                                                                          │
                                    (worker side: internal data-plane    │
                                     pull of weight)                     │
                                    data DOWNLOAD_REQUEST(               │
                                      artifact_id="req-7-weights")       │
                                      -----------------> main            │
                                    <-- data INIT/CHUNK x M/END --       │
                                    worker writes to disk                │
                                                                          │
               │ <-- ctrl TASK_ACCEPT --                                 │
               │                                                          │
                                    worker locally executes conv2d slice │
                                    writes its slice result into a local │
                                    artifact (an embedded data-plane     │
                                    server exposes that artifact)        │
               │ <-- ctrl TASK_RESULT(                                   │
               │     result_payload.Conv2dResultPayload{                 │
               │       start_oc, end_oc, output_h, output_w,             │
               │       result_artifact_id=<worker's artifact id>         │
               │     },                                                   │
               │     result_artifact=ArtifactDescriptor{points to worker})│
                                                                          │
               │ main pulls from worker's data plane per descriptor:     │
               │ -- data DOWNLOAD_REQUEST(artifact_id) ---> worker       │
               │ <-- data INIT/CHUNK x M/END ----          worker        │
               │                                                          │
               │ -- ctrl ARTIFACT_RELEASE(artifact_id) --> worker        │
               │   (tells worker it may delete locally now)              │
               │                                                          │
               │ aggregator stitches N output slices into the final      │
               │ result                                                   │
               │ ArtifactManager.publish_bytes(result,                   │
               │   artifact_id = download_id)                            │
                                                                          │
               │ <- ctrl CLIENT_RESPONSE                                 │
               │     method=conv2d,                                      │
               │     result_artifact=ArtifactDescriptor{                 │
               │       artifact_id = download_id,                        │
               │       transfer_host=main_host,                          │
               │       transfer_port=45000, ...                          │
               │     }                                                   │
                                                                          │
  (control-plane message in hand; now reuse the data socket opened in    │
   step 2)                                                                │
  data DOWNLOAD_REQUEST(download_id)                                     │
               ----------------->                                        │
               <-- data INIT(size, checksum, chunk_size)                 │
               <-- data CHUNK x K                                        │
               <-- data END(size)                                        │
  (client writes to disk, closes data socket)                            │
 --------------+----------------------+----------------------+-------------
```

Synchronization points:
  - The main thread waits on `upload_future.result()` before it can dispatch, so the client's `DELIVER` must finish (`END`) at main before main begins partitioning.
  - Main only sends `CLIENT_RESPONSE` after gathering all workers' `TASK_RESULT`. This guarantees that by the time the client issues `DOWNLOAD_REQUEST(download_id)`, the artifact has already been registered in main's `ArtifactManager`.


## 9. Phase V: Conv2D stats_only (no downstream artifact, upstream unchanged)

The differences from Phase IV are confined to two places:

```text
  ctrl CLIENT_REQUEST_OK.download_id = ""        (main does not pre-allocate
                                                  a result id)
  ctrl CLIENT_RESPONSE.result_artifact = None    (result is folded back into
                                                  protobuf)
       response_payload.Conv2dResponsePayload {
         stats_element_count, stats_sum, stats_sum_squares,
         stats_samples[..]
       }
```

The upstream `DELIVER` / `CHUNK` / `END` flow is identical to Phase IV. The worker-side `TASK_ASSIGN` / `TASK_RESULT` behavior is also identical, because the worker is unaware of `client_response_mode`. It is only when assembling `CLIENT_RESPONSE` that main chooses the stats branch instead of publishing a result artifact.


## 10. Which message goes on which plane: one-line mnemonics

Control plane (always traversed):
  - identity management (`JOIN`/`REGISTER`/`OK`)
  - heartbeat and liveness detection (`HEARTBEAT`/`HEARTBEAT_OK`/`CLIENT_INFO_*`)
  - request negotiation (`CLIENT_REQUEST` / `CLIENT_REQUEST_OK` / `TASK_ASSIGN` / ...)
  - direct return of small payloads (gemv output, stats-only result)
  - task lifecycle events (`TASK_ACCEPT`/`FAIL`/`RESULT`, `ARTIFACT_RELEASE`)

Data plane (trigger conditions):
  - `client → main`: a conv2d request with `upload_size_bytes > 0` (entry via `DELIVER`)
  - `main → worker` reverse: worker pulls conv2d weight from main (`TASK_ASSIGN.weight_artifact` points to main)
  - `worker → main`: main pulls worker's result back per `TASK_RESULT.result_artifact`
  - `main → client`: the result artifact for conv2d full (reuses the same socket as above)

Discovery plane (only on cold start):
  - any node looking for main (client / worker) on its first connection
  - on TCP reconnect, first see if discovery can run again; otherwise fall back to the explicit address from configuration


## 11. Common misconceptions

Q1. "Why doesn't main just stuff the result bytes into the control socket?"
    Because the control plane is length-prefixed protobuf — the entire message must be read in one go before it can be decoded. A 10 MB result would block heartbeats and other request acks. So any payload larger than a few KB goes onto the SWAD data plane.

Q2. "Why doesn't the client just P2P-push weight directly to a worker?"
    Because the client has no visibility into workers at all (that line does not exist in the topology); workers are an internal implementation detail of main. The client uniformly only talks to main, simplifying three things: permissions, discovery, and firewalls.

Q3. "Why are `DELIVER` and `DOWNLOAD_REQUEST` two different frames?"
    The directions are reversed (`DELIVER` initiates a push, `DOWNLOAD_REQUEST` initiates a pull), and the fields differ (`DELIVER` must declare `size`/`checksum`, `DOWNLOAD_REQUEST` only carries `artifact_id`). Sharing one frame would prevent main's dispatcher from telling intent.

Q4. "Why don't workers need a data plane between themselves?"
    The current operators (gemv, conv2d) are map-only; each slice is independent. With no shuffle, there is no need for worker-to-worker connections.

Q5. "What if the network does not support multicast for mDNS?"
    Client / worker can skip `DISCOVER` and use `--main-host host:port` to specify main's TCP endpoint directly. The discovery plane is essentially just "first-time resolution of host:port".


## 12. Quick self-check list (for whoever is debugging)

  - [ ] Cannot connect to main:        first tcpdump UDP `:5353` and check whether `ANNOUNCE` is present
  - [ ] `CLIENT_JOIN` times out:        is the main control server actually on `:31000`?
  - [ ] `REQUEST_OK` is missing `upload_id`: cross-check the `method`/`stats_only`/`upload_size_bytes` triple against the §3 decision table
  - [ ] After `DELIVER` the future never returns: is the main data server (`:45000`) actually listening, and do the upload slot's `size`/`checksum` line up?
  - [ ] Worker fails to pull weight:    is `TASK_ASSIGN.weight_artifact.transfer_host` the interface of main that is reachable from the worker side?
  - [ ] Client fails to pull result:    is `CLIENT_RESPONSE.result_artifact` non-`None`, and is `download_id` equal to `result_artifact.artifact_id`?
