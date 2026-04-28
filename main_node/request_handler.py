"""Validate client requests, dispatch worker slices, and assemble one response.

Use this module after a client has already joined the main node and sent one
structured request. It owns request validation, worker assignment selection,
artifact decisions, and final response construction.
"""

from __future__ import annotations

import concurrent.futures
import os
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from adapters.audit_log import write_audit_event, write_diag_event
from core.constants import (
    CONV2D_CLIENT_RESPONSE_STATS_ONLY,
    CONV2D_STATS_MAX_SAMPLES,
    INLINE_RESPONSE_ENVELOPE_MARGIN,
    MAIN_NODE_NAME,
    METHOD_FREE_CONTENT,
    METHOD_GEMM,
    METHOD_GEMV,
    METHOD_CONV2D,
    STATUS_BAD_REQUEST,
    STATUS_INTERNAL_ERROR,
    STATUS_NOT_FOUND,
    STATUS_OK,
)
from core.tracing import trace_function
from core.work_partition import partition_contiguous_range
from compute_node.input_matrix.gemv import build_input_matrix_spec as build_gemv_input_matrix_spec
from compute_node.input_matrix.gemm import build_spec as gemm_build_spec
from compute_node.compute_methods.conv2d.executor import load_named_workload_spec
from main_node.dispatcher import WorkerTaskSlice
from wire.internal_protocol.transport import (
    GemmResponsePayload,
    GemvResponsePayload,
    Conv2dResponsePayload,
    ResponseTiming,
    build_client_response,
)


@dataclass(slots=True)
class DataPlaneAllocation:
    """Pre-registered upload/download IDs handed to the client in REQUEST_OK.

    Populated by ``ClientRequestHandler.allocate_data_plane_endpoints`` after
    request validation and before the main node replies with CLIENT_REQUEST_OK.
    Carries identifiers that the client echoes back on its pre-opened data
    socket: ``upload_id`` as the capability token on the DELIVER frame, and
    ``download_id`` as the artifact id the client will pull after the result
    is produced. ``upload_future`` resolves to the local weight path once the
    upload completes, or raises when the upload is rejected.
    """

    upload_id: str = ""
    download_id: str = ""
    data_endpoint_host: str = ""
    data_endpoint_port: int = 0
    upload_future: concurrent.futures.Future | None = None


class ClientRequestHandler:
    """Own the main-node request lifecycle from validation through aggregation."""

    def __init__(
        self,
        *,
        config,
        registry,
        dispatcher,
        aggregator,
        gemv_spec,
        conv2d_dataset_dir,
        task_exchange,
        artifact_manager,
        cluster_counts,
        logger=None,
    ) -> None:
        """Wire together the services needed to handle one client request end to end.

        Args: config runtime settings, registry/dispatcher/aggregator cluster services, gemv_spec and conv2d_dataset_dir workload metadata, task_exchange/artifact_manager transfer helpers, cluster_counts callable for reply metadata, logger audit sink for operator progress messages.
        Returns: None after the handler stores references to its shared collaborators.
        """
        self.config = config
        self.registry = registry
        self.dispatcher = dispatcher
        self.aggregator = aggregator
        self.gemv_spec = gemv_spec
        self.conv2d_dataset_dir = conv2d_dataset_dir
        self.task_exchange = task_exchange
        self.run_worker_task_slice = task_exchange.run_worker_task_slice
        self.artifact_manager = artifact_manager
        self._cluster_counts = cluster_counts
        self.logger = logger

    def _safe_float(self, value) -> float:
        """Convert one maybe-missing numeric value into a stable float."""

        try:
            return float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _build_system_overview_text(self) -> str:
        """Render one human-readable system overview for free-content replies."""

        worker_count, client_count = self._cluster_counts()
        total_effective_gflops = self._safe_float(self.registry.total_registered_gflops())
        method_totals = self.registry.total_registered_gflops_by_method()
        if not isinstance(method_totals, dict):
            method_totals = {}
        main_node_host = getattr(self.artifact_manager, "public_host", "127.0.0.1") or "127.0.0.1"
        supported_methods = ", ".join((METHOD_GEMV, METHOD_CONV2D, METHOD_GEMM, METHOD_FREE_CONTENT))
        return "\n".join(
            (
                "superweb-cluster system overview",
                f"main_node_endpoint: {main_node_host}:{self.config.tcp_port}",
                f"worker_count: {worker_count}",
                f"client_count: {client_count}",
                f"total_effective_gflops: {total_effective_gflops:.3f}",
                f"gemv_effective_gflops: {self._safe_float(method_totals.get(METHOD_GEMV, 0.0)):.3f}",
                f"conv2d_effective_gflops: {self._safe_float(method_totals.get(METHOD_CONV2D, 0.0)):.3f}",
                f"gemm_effective_gflops: {self._safe_float(method_totals.get(METHOD_GEMM, 0.0)):.3f}",
                f"supported_methods: {supported_methods}",
            )
        )

    def allocate_data_plane_endpoints(self, request) -> DataPlaneAllocation:
        """Pre-register upload and download IDs for one accepted client request.

        Use this after the session service has assigned a ``task_id`` on the
        request but before sending ``CLIENT_REQUEST_OK``. Decides deterministically
        whether the client needs an upload slot (``upload_size_bytes > 0`` in the
        conv2d payload) and whether the main-node result will be delivered over
        the data plane (conv2d with ``client_response_mode != STATS_ONLY``). The
        client is told these ids up front so it can pre-open its data socket.

        Args:
            request: Decoded ``CLIENT_REQUEST`` envelope with task_id already
                assigned and method-specific payload populated.

        Returns:
            A ``DataPlaneAllocation`` describing the upload slot (if any) and
            the reserved download id. Empty fields mean the client does not
            need to open a data socket for that direction.
        """
        allocation = DataPlaneAllocation(
            data_endpoint_host=self.artifact_manager.public_host,
            data_endpoint_port=self.artifact_manager.port,
        )
        if request.method != METHOD_CONV2D:
            return allocation

        conv2d_payload = request.conv2d_payload
        upload_size_bytes = int(getattr(conv2d_payload, "upload_size_bytes", 0) or 0) if conv2d_payload is not None else 0
        upload_checksum = str(getattr(conv2d_payload, "upload_checksum", "") or "") if conv2d_payload is not None else ""
        client_response_mode = int(getattr(conv2d_payload, "client_response_mode", 0) or 0) if conv2d_payload is not None else 0
        stats_only = client_response_mode == CONV2D_CLIENT_RESPONSE_STATS_ONLY

        if upload_size_bytes > 0:
            allocation.upload_id = f"{request.request_id}-upload-{uuid.uuid4().hex[:8]}"
            allocation.upload_future = self.artifact_manager.register_upload_slot(
                upload_id=allocation.upload_id,
                expected_size=upload_size_bytes,
                expected_checksum=upload_checksum,
                expected_content_type="application/x-superweb-conv2d-weight",
                destination_suffix=".weight.bin",
            )

        if not stats_only:
            allocation.download_id = f"{request.request_id}-download-{uuid.uuid4().hex[:8]}"

        return allocation

    def _cleanup_local_task_results(self, task_results) -> None:
        """Delete any temporary local result files attached to worker task results.

        Use this after the main node has merged worker outputs so temporary
        fetch-to-file artifacts do not accumulate under the artifact workspace.

        Args:
            task_results: Iterable of worker ``TaskResult`` instances.

        Returns:
            ``None`` after all temporary local result files are removed.
        """
        for task_result in task_results:
            if task_result.local_result_path:
                try:
                    os.unlink(task_result.local_result_path)
                except FileNotFoundError:
                    pass

    def _build_retry_assignments(
        self,
        request,
        failed_slice: WorkerTaskSlice,
        surviving_connections,
        original_weights: dict[str, float],
        retry_counter: list[int],
    ) -> list[WorkerTaskSlice]:
        """Re-partition a failed slice's range across surviving workers.

        Why: a single worker dying must not sink the entire client request — the
        dispatcher already partitioned the request proportional to effective
        GFLOPS, so the fairest redispatch is to split the failed range across
        the remaining workers using those same weights. New task/artifact ids
        are minted per retry so the wire protocol and connection mailboxes can
        route messages for original and retry slices to the same worker without
        id collisions.
        """
        if failed_slice.method == METHOD_GEMV:
            range_start, range_end = failed_slice.row_start, failed_slice.row_end
        elif failed_slice.method == METHOD_GEMM:
            range_start, range_end = failed_slice.m_start, failed_slice.m_end
        else:
            range_start, range_end = failed_slice.start_oc, failed_slice.end_oc
        if range_end <= range_start:
            return []

        weights = [float(original_weights.get(conn.peer_id, 0.0)) for conn in surviving_connections]
        if sum(weights) <= 0.0:
            weights = [1.0] * len(surviving_connections)

        partitions = partition_contiguous_range(range_start, range_end, weights)

        retry_slices: list[WorkerTaskSlice] = []
        for partition, connection in zip(partitions, surviving_connections):
            if partition.end <= partition.start:
                continue
            retry_index = retry_counter[0]
            retry_counter[0] += 1
            retry_task_id = f"{request.request_id}-r{retry_index}"
            if failed_slice.method == METHOD_GEMV:
                retry_slices.append(
                    WorkerTaskSlice(
                        connection=connection,
                        task_id=retry_task_id,
                        artifact_id=f"{retry_task_id}:{connection.runtime_id}",
                        row_start=partition.start,
                        row_end=partition.end,
                        effective_gflops=float(original_weights.get(connection.peer_id, 0.0)),
                        method=METHOD_GEMV,
                    )
                )
            elif failed_slice.method == METHOD_GEMM:
                retry_slices.append(
                    WorkerTaskSlice(
                        connection=connection,
                        task_id=retry_task_id,
                        artifact_id=f"{retry_task_id}:{connection.runtime_id}",
                        row_start=0,
                        row_end=0,
                        m_start=partition.start,
                        m_end=partition.end,
                        effective_gflops=float(original_weights.get(connection.peer_id, 0.0)),
                        method=METHOD_GEMM,
                    )
                )
            else:
                retry_slices.append(
                    WorkerTaskSlice(
                        connection=connection,
                        task_id=retry_task_id,
                        artifact_id=f"{retry_task_id}:{connection.runtime_id}:0",
                        row_start=0,
                        row_end=0,
                        start_oc=partition.start,
                        end_oc=partition.end,
                        effective_gflops=float(original_weights.get(connection.peer_id, 0.0)),
                        method=METHOD_CONV2D,
                    )
                )
        return retry_slices

    def _run_assignments_with_failover(self, request, assignments: list[WorkerTaskSlice]):
        """Execute worker assignments with mid-task failover.

        Submits every assignment to a thread pool. When a worker slice raises,
        the failed worker is marked dead and its range is re-sliced across the
        still-healthy workers proportional to their original effective GFLOPS.
        The request only fails when no healthy workers remain.

        Returns a list of ``(TaskResult, WorkerTiming)`` outcomes whose union of
        ranges tiles the original request range, ready for the aggregator.
        """
        if not assignments:
            return []

        original_weights = {
            assignment.connection.peer_id: assignment.effective_gflops
            for assignment in assignments
        }

        def _peer_id(assignment: WorkerTaskSlice) -> str:
            return assignment.connection.peer_id

        def _unique_survivors(failed_peer_ids: set[str]):
            """Return one connection per live peer from the assignment pool."""
            unique: list = []
            seen: set[str] = set()
            for assignment in assignments:
                peer_id = _peer_id(assignment)
                if peer_id in failed_peer_ids or peer_id in seen:
                    continue
                seen.add(peer_id)
                unique.append(assignment.connection)
            return unique

        outcomes: list = []
        failed_peer_ids: set[str] = set()
        retry_counter = [0]
        pending: dict = {}

        def _slice_scope(slice_: WorkerTaskSlice) -> str:
            if request.method == METHOD_CONV2D:
                return f"oc={slice_.start_oc}:{slice_.end_oc}"
            if request.method == METHOD_GEMM:
                return f"m_rows={slice_.m_start}:{slice_.m_end}"
            return f"rows={slice_.row_start}:{slice_.row_end}"

        def _submit_retry(slice_: WorkerTaskSlice):
            """Submit one retry slice and surface any queue-wait to operators.

            When the survivor's ``task_lock`` is already held by an in-flight
            original slice, the retry will block inside ``run_worker_task_slice``
            until that lock releases. Emit one audit line at submit time so the
            log shows *why* there is a gap between ``redispatch`` and the next
            ``TASK_ASSIGN`` line for this retry task.
            """
            target_lock = getattr(slice_.connection, "task_lock", None)
            if target_lock is not None and getattr(target_lock, "locked", lambda: False)():
                write_audit_event(
                    f"worker failover retry queued request_id={request.request_id} "
                    f"retry_task_id={slice_.task_id} "
                    f"target_runtime_id={slice_.connection.runtime_id} "
                    f"reason=connection busy with prior slice on this worker",
                    stdout=True,
                    logger=self.logger,
                )
            return executor.submit(self.run_worker_task_slice, request, slice_)

        max_workers = max(len(assignments), 1) * 2
        dispatch_started_at = time.monotonic()
        total_dispatched = len(assignments)
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="task-dispatch") as executor:
            for assignment in assignments:
                future = executor.submit(self.run_worker_task_slice, request, assignment)
                pending[future] = assignment
            distinct_initial_workers = {
                assignment.connection.runtime_id for assignment in assignments
            }
            write_diag_event(
                f"[DIAG] all initial assignments submitted request_id={request.request_id} "
                f"slice_count={len(assignments)} "
                f"dispatched_workers={len(distinct_initial_workers)} "
                f"cluster_workers={self.registry.count_workers()} "
                f"submit_elapsed_ms={max(0, int((time.monotonic() - dispatch_started_at) * 1000))}",
                logger=self.logger,
            )

            while pending:
                done, _ = concurrent.futures.wait(
                    pending,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    failed_assignment = pending.pop(future)
                    try:
                        task_result, timing = future.result()
                        outcomes.append((task_result, timing))
                        artifact_bytes = (
                            task_result.result_artifact.size_bytes
                            if task_result.result_artifact is not None
                            else 0
                        )
                        write_diag_event(
                            f"[DIAG] worker outcome collected request_id={request.request_id} "
                            f"received={len(outcomes)}/{total_dispatched} "
                            f"remaining={len(pending)} "
                            f"task_id={failed_assignment.task_id} "
                            f"runtime_id={failed_assignment.connection.runtime_id} "
                            f"wall_ms={timing.wall_ms} artifact_fetch_ms={timing.artifact_fetch_ms} "
                            f"computation_ms={timing.computation_ms} peripheral_ms={timing.peripheral_ms} "
                            f"artifact_bytes={artifact_bytes}",
                            logger=self.logger,
                        )
                        continue
                    except (OSError, ValueError, ConnectionError, RuntimeError) as exc:
                        peer_id = _peer_id(failed_assignment)
                        failed_peer_ids.add(peer_id)
                        write_audit_event(
                            f"worker failover triggered request_id={request.request_id} "
                            f"failed_runtime_id={failed_assignment.connection.runtime_id} "
                            f"failed_task_id={failed_assignment.task_id} "
                            f"method={request.method} error={exc}",
                            stdout=True,
                            logger=self.logger,
                        )
                        survivors = _unique_survivors(failed_peer_ids)
                        if not survivors:
                            raise RuntimeError(
                                f"request {request.request_id} failed: all workers failed during "
                                f"task execution, no surviving worker to redispatch to "
                                f"(last error: {exc})"
                            ) from exc
                        retry_slices = self._build_retry_assignments(
                            request,
                            failed_assignment,
                            survivors,
                            original_weights,
                            retry_counter,
                        )
                        for retry_slice in retry_slices:
                            write_audit_event(
                                f"worker failover redispatch request_id={request.request_id} "
                                f"retry_task_id={retry_slice.task_id} "
                                f"target_runtime_id={retry_slice.connection.runtime_id} "
                                f"{_slice_scope(retry_slice)}",
                                logger=self.logger,
                            )
                            future = _submit_retry(retry_slice)
                            pending[future] = retry_slice
                            total_dispatched += 1

        write_diag_event(
            f"[DIAG] all worker outcomes collected request_id={request.request_id} "
            f"outcome_count={len(outcomes)}/{total_dispatched} "
            f"total_elapsed_ms={max(0, int((time.monotonic() - dispatch_started_at) * 1000))}",
            logger=self.logger,
        )
        return outcomes

    @trace_function
    def build_client_response_for_request(
        self,
        request,
        *,
        allocation: DataPlaneAllocation | None = None,
    ):
        """Use this after receiving a validated CLIENT_REQUEST from one client session.

        Args: request decoded client request envelope with method-specific payload,
            allocation optional data-plane allocation previously registered by
            ``allocate_data_plane_endpoints`` and announced in ``CLIENT_REQUEST_OK``.
        Returns: A ready-to-send ClientResponse envelope, including inline bytes or an artifact descriptor.
        """
        task_id = request.request_id
        started_at = time.monotonic()
        if allocation is None:
            allocation = DataPlaneAllocation()
        try:
            return self._build_client_response_body(request, allocation, task_id, started_at)
        finally:
            if allocation.upload_id and allocation.upload_future is not None:
                self.artifact_manager.cancel_upload_slot(allocation.upload_id)

    def _build_client_response_body(self, request, allocation, task_id, started_at):
        """Render the response for one validated client request.

        Split out from ``build_client_response_for_request`` so the outer call
        site can reliably cancel any orphan upload slot in a single finally,
        regardless of which early-return path below fires.
        """

        def build_response(*, status_code: int, **kwargs):
            kwargs.setdefault("size", request.size)
            return build_client_response(
                request_id=task_id,
                status_code=status_code,
                task_id=task_id,
                elapsed_ms=max(0, int((time.monotonic() - started_at) * 1000)),
                **kwargs,
            )

        worker_count, client_count = self._cluster_counts()

        if request.method == METHOD_FREE_CONTENT:
            overview_bytes = self._build_system_overview_text().encode("utf-8")
            return build_response(
                status_code=STATUS_OK,
                method=request.method,
                object_id=request.object_id,
                stream_id=request.stream_id,
                output_vector=overview_bytes,
                output_length=len(overview_bytes),
                worker_count=worker_count,
                client_count=client_count,
                client_id="",
                iteration_count=request.iteration_count,
            )

        if request.method not in (METHOD_GEMV, METHOD_CONV2D, METHOD_GEMM):
            return build_response(
                status_code=STATUS_BAD_REQUEST,
                method=request.method,
                object_id=request.object_id,
                stream_id=request.stream_id,
                error_message=f"unsupported method: {request.method}",
                worker_count=worker_count,
                client_count=client_count,
                client_id="",
                iteration_count=request.iteration_count,
            )

        gemv_spec = build_gemv_input_matrix_spec(default_variant=request.size or "large")
        client_weight_path = None

        if request.method == METHOD_GEMV and request.vector_length != gemv_spec.cols:
            return build_response(
                status_code=STATUS_BAD_REQUEST,
                method=request.method,
                object_id=request.object_id,
                stream_id=request.stream_id,
                error_message=(
                    f"vector_length={request.vector_length} does not match "
                    f"expected {gemv_spec.cols}"
                ),
                worker_count=worker_count,
                client_count=client_count,
                client_id="",
                iteration_count=request.iteration_count,
            )

        if request.method == METHOD_GEMV and len(request.vector_data) != request.vector_length * 4:
            return build_response(
                status_code=STATUS_BAD_REQUEST,
                method=request.method,
                object_id=request.object_id,
                stream_id=request.stream_id,
                error_message="vector_data byte length does not match vector_length",
                worker_count=worker_count,
                client_count=client_count,
                client_id="",
                iteration_count=request.iteration_count,
            )
        if request.iteration_count <= 0:
            return build_response(
                status_code=STATUS_BAD_REQUEST,
                method=request.method,
                object_id=request.object_id,
                stream_id=request.stream_id,
                error_message="iteration_count must be positive",
                worker_count=worker_count,
                client_count=client_count,
                client_id="",
                iteration_count=1,
            )

        workers = self.registry.list_workers()
        if request.method == METHOD_GEMV:
            assignments = self.dispatcher.dispatch_gemv(
                request_id=request.request_id,
                rows=gemv_spec.rows,
                workers=workers,
                worker_hardware=self.registry.list_worker_hardware(METHOD_GEMV),
            )
        elif request.method == METHOD_GEMM:
            try:
                gemm_spec = gemm_build_spec(default_variant=request.size or "large")
            except ValueError as exc:
                return build_response(
                    status_code=STATUS_BAD_REQUEST,
                    method=request.method,
                    object_id=request.object_id,
                    stream_id=request.stream_id,
                    error_message=str(exc),
                    worker_count=worker_count,
                    client_count=client_count,
                    client_id="",
                    iteration_count=request.iteration_count,
                )
            assignments = self.dispatcher.dispatch_gemm(
                request_id=request.request_id,
                rows=gemm_spec.m,
                workers=workers,
                worker_hardware=self.registry.list_worker_hardware(METHOD_GEMM),
            )
        else:
            try:
                spec, _variant = load_named_workload_spec(request.object_id, size=request.size)
            except ValueError as exc:
                return build_response(
                    status_code=STATUS_BAD_REQUEST,
                    method=request.method,
                    object_id=request.object_id,
                    stream_id=request.stream_id,
                    error_message=str(exc),
                    worker_count=worker_count,
                    client_count=client_count,
                    client_id="",
                    iteration_count=request.iteration_count,
                )

            if allocation.upload_future is not None:
                upload_wait_started_at = time.monotonic()
                try:
                    weight_path = allocation.upload_future.result(
                        timeout=float(self.config.compute_artifact_ttl_seconds),
                    )
                    write_diag_event(
                        f"[DIAG] client weight upload received task_id={task_id} "
                        f"upload_id={allocation.upload_id} "
                        f"local_path={weight_path} "
                        f"wait_ms={max(0, int((time.monotonic() - upload_wait_started_at) * 1000))}",
                        logger=self.logger,
                    )
                except concurrent.futures.TimeoutError as exc:
                    self.artifact_manager.cancel_upload_slot(allocation.upload_id)
                    return build_response(
                        status_code=STATUS_INTERNAL_ERROR,
                        method=request.method,
                        object_id=request.object_id,
                        stream_id=request.stream_id,
                        error_message=f"timed out waiting for client weight upload: {exc}",
                        worker_count=worker_count,
                        client_count=client_count,
                        client_id="",
                        iteration_count=request.iteration_count,
                    )
                except (OSError, RuntimeError, ValueError, concurrent.futures.CancelledError) as exc:
                    return build_response(
                        status_code=STATUS_INTERNAL_ERROR,
                        method=request.method,
                        object_id=request.object_id,
                        stream_id=request.stream_id,
                        error_message=f"failed to receive client weight upload: {exc}",
                        worker_count=worker_count,
                        client_count=client_count,
                        client_id="",
                        iteration_count=request.iteration_count,
                    )
                client_weight_path = weight_path
                self.task_exchange.register_client_weight_path(request.request_id, weight_path)
            else:
                try:
                    self.task_exchange.ensure_conv2d_dataset_ready()
                except (OSError, subprocess.CalledProcessError) as exc:
                    return build_response(
                        status_code=STATUS_INTERNAL_ERROR,
                        method=request.method,
                        object_id=request.object_id,
                        stream_id=request.stream_id,
                        error_message=f"failed to prepare conv2d dataset: {exc}",
                        worker_count=worker_count,
                        client_count=client_count,
                        client_id="",
                        iteration_count=request.iteration_count,
                    )

            assignments = self.dispatcher.dispatch_conv2d(
                request_id=request.request_id,
                output_channels=spec.c_out,
                workers=workers,
                worker_hardware=self.registry.list_worker_hardware(METHOD_CONV2D),
            )
        if not assignments:
            if client_weight_path is not None:
                self.task_exchange.unregister_client_weight_path(request.request_id)
                try:
                    client_weight_path.unlink(missing_ok=True)
                except OSError:
                    pass
            return build_response(
                status_code=STATUS_NOT_FOUND,
                method=request.method,
                object_id=request.object_id,
                stream_id=request.stream_id,
                error_message="no registered compute workers are currently available",
                worker_count=worker_count,
                client_count=client_count,
                client_id="",
                iteration_count=request.iteration_count,
            )

        total_effective_gflops = sum(
            float(a.effective_gflops or 0.0) for a in assignments
        )
        assignment_descriptions: list[str] = []
        for assignment in assignments:
            if request.method == METHOD_CONV2D:
                scope = f"oc={assignment.start_oc}:{assignment.end_oc} span={assignment.end_oc - assignment.start_oc}"
            elif request.method == METHOD_GEMM:
                scope = f"m_rows={assignment.m_start}:{assignment.m_end} span={assignment.m_end - assignment.m_start}"
            else:
                scope = f"rows={assignment.row_start}:{assignment.row_end} span={assignment.row_end - assignment.row_start}"
            weight_frac = (
                float(assignment.effective_gflops or 0.0) / total_effective_gflops
                if total_effective_gflops > 0
                else 0.0
            )
            assignment_descriptions.append(
                f"[runtime_id={assignment.connection.runtime_id} "
                f"gflops={float(assignment.effective_gflops or 0.0):.3f} "
                f"frac={weight_frac:.3f} {scope}]"
            )
        distinct_worker_runtime_ids = {
            assignment.connection.runtime_id for assignment in assignments
        }
        write_diag_event(
            f"[DIAG] request accepted task_id={task_id} method={request.method} "
            f"size={request.size or 'large'} "
            f"cluster_workers={self.registry.count_workers()} "
            f"dispatched_workers={len(distinct_worker_runtime_ids)} "
            f"slice_count={len(assignments)} "
            f"total_effective_gflops={total_effective_gflops:.3f} "
            f"assignments={' '.join(assignment_descriptions)}",
            logger=self.logger,
        )

        try:
            completion_conv2d_stats_only = False
            dispatch_done_at = time.monotonic()
            if request.method == METHOD_GEMV:
                task_window_started_at = time.monotonic()
                outcomes = self._run_assignments_with_failover(request, assignments)
                task_window_done_at = time.monotonic()
                task_results = [outcome[0] for outcome in outcomes]
                worker_timings = tuple(outcome[1] for outcome in outcomes)
                write_audit_event(
                    f"aggregating result task_id={task_id} method={request.method} size={request.size or 'large'} worker_slices={len(task_results)}",
                    stdout=True,
                    logger=self.logger,
                )
                output_vector = self.aggregator.collect_gemv_result(
                    rows=gemv_spec.rows,
                    results=task_results,
                )
                output_length = gemv_spec.rows
                result_artifact = None
                if len(output_vector) + INLINE_RESPONSE_ENVELOPE_MARGIN > self.config.max_message_size:
                    result_artifact = self.artifact_manager.publish_bytes(
                        output_vector,
                        producer_node_id=MAIN_NODE_NAME,
                        content_type="application/octet-stream",
                        artifact_id=request.request_id,
                    )
                    output_vector = b""
                response_payload = GemvResponsePayload(
                    output_length=output_length,
                    output_vector=output_vector,
                )
            elif request.method == METHOD_GEMM:
                task_window_started_at = time.monotonic()
                outcomes = self._run_assignments_with_failover(request, assignments)
                task_window_done_at = time.monotonic()
                task_results = [outcome[0] for outcome in outcomes]
                worker_timings = tuple(outcome[1] for outcome in outcomes)
                write_audit_event(
                    f"aggregating result task_id={task_id} method={request.method} size={request.size or 'large'} worker_slices={len(task_results)}",
                    stdout=True,
                    logger=self.logger,
                )
                output_vector = self.aggregator.collect_gemm_result(
                    m=gemm_spec.m,
                    n=gemm_spec.n,
                    results=task_results,
                )
                output_length = gemm_spec.m * gemm_spec.n
                if self.artifact_manager is None:
                    raise RuntimeError("artifact manager is required to publish GEMM results")
                # GEMM results always go through the data plane, never inline:
                # even the smallest variant (m=n=k=1024) produces a 4 MiB C
                # matrix that matches the max_message_size budget exactly, so
                # inline encoding always risks tripping the receiver's size
                # guard once the envelope fields are serialised. Mirrors
                # conv2d's unconditional-artifact behaviour.
                result_artifact = self.artifact_manager.publish_bytes(
                    output_vector,
                    producer_node_id=MAIN_NODE_NAME,
                    content_type="application/octet-stream",
                    artifact_id=request.request_id,
                )
                response_payload = GemmResponsePayload(
                    output_length=output_length,
                    output_vector=b"",
                )
            else:
                spec, _variant = load_named_workload_spec(request.object_id, size=request.size)
                output_length = spec.output_h * spec.output_w * spec.c_out
                output_vector = b""
                result_artifact = None
                conv_mode = 0
                stats_max_samples = 0
                payload = request.conv2d_payload
                if payload is not None:
                    conv_mode = int(payload.client_response_mode)
                    stats_max_samples = int(payload.stats_max_samples)
                if self.artifact_manager is None:
                    raise RuntimeError("artifact manager is required for conv2d responses")
                stats_only_request = conv_mode == CONV2D_CLIENT_RESPONSE_STATS_ONLY
                artifact_path = (
                    None
                    if stats_only_request
                    else self.artifact_manager.root_dir / f"{request.request_id}.bin"
                )
                task_window_started_at = time.monotonic()
                outcomes = self._run_assignments_with_failover(request, assignments)
                task_window_done_at = time.monotonic()
                task_results = [outcome[0] for outcome in outcomes]
                worker_timings = tuple(outcome[1] for outcome in outcomes)
                if stats_only_request:
                    write_audit_event(
                        f"aggregating result task_id={task_id} method={request.method} size={request.size or 'large'} worker_slices={len(task_results)} "
                        f"mode=stats_only",
                        stdout=True,
                        logger=self.logger,
                    )
                    try:
                        effective_max_samples = stats_max_samples if stats_max_samples > 0 else CONV2D_STATS_MAX_SAMPLES
                        stats_count, stats_sum, stats_sum_squares, stats_samples = self.aggregator.aggregate_conv2d_stats(
                            results=task_results,
                            total_cout=spec.c_out,
                            out_h=spec.output_h,
                            out_w=spec.output_w,
                            max_samples=effective_max_samples,
                        )
                    finally:
                        self._cleanup_local_task_results(task_results)
                    if stats_count != output_length:
                        raise ValueError(
                            f"conv2d aggregated stats element count mismatch: stats_element_count={stats_count} "
                            f"expected output_length={output_length}"
                        )
                    completion_conv2d_stats_only = True
                    response_payload = Conv2dResponsePayload(
                        output_length=output_length,
                        output_vector=b"",
                        result_artifact_id="",
                        stats_element_count=stats_count,
                        stats_sum=stats_sum,
                        stats_sum_squares=stats_sum_squares,
                        stats_samples=stats_samples,
                    )
                else:
                    write_audit_event(
                        f"aggregating result task_id={task_id} method={request.method} size={request.size or 'large'} worker_slices={len(task_results)} "
                        f"output_path={artifact_path}",
                        stdout=True,
                        logger=self.logger,
                    )
                    try:
                        self.aggregator.collect_conv2d_result_to_file(
                            out_h=spec.output_h,
                            out_w=spec.output_w,
                            total_cout=spec.c_out,
                            results=task_results,
                            output_path=artifact_path,
                        )
                    finally:
                        self._cleanup_local_task_results(task_results)
                    result_artifact = self.artifact_manager.register_existing_file(
                        artifact_path,
                        producer_node_id=MAIN_NODE_NAME,
                        content_type="application/octet-stream",
                        artifact_id=allocation.download_id or request.request_id,
                    )
                    response_payload = Conv2dResponsePayload(
                        output_length=output_length,
                        output_vector=output_vector,
                        result_artifact_id=result_artifact.artifact_id if result_artifact is not None else "",
                    )
            aggregate_done_at = time.monotonic()
            write_diag_event(
                f"[DIAG] aggregator finished task_id={task_id} method={request.method} "
                f"aggregate_ms={max(0, int((aggregate_done_at - task_window_done_at) * 1000))} "
                f"output_length={output_length} "
                f"result_artifact={'yes' if result_artifact is not None else 'no'}",
                logger=self.logger,
            )
            timing = ResponseTiming(
                dispatch_ms=max(0, int((dispatch_done_at - started_at) * 1000)),
                task_window_ms=max(0, int((task_window_done_at - task_window_started_at) * 1000)),
                aggregate_ms=max(0, int((aggregate_done_at - task_window_done_at) * 1000)),
                workers=worker_timings,
            )
            completion_parts = [
                f"task complete task_id={task_id}",
                f"method={request.method}",
                f"size={request.size or 'large'}",
                f"elapsed_ms={max(0, int((time.monotonic() - started_at) * 1000))}",
                f"dispatch_ms={timing.dispatch_ms}",
                f"task_window_ms={timing.task_window_ms}",
                f"aggregate_ms={timing.aggregate_ms}",
                f"output_length={output_length}",
            ]
            if result_artifact is not None:
                completion_parts.append(f"artifact_id={result_artifact.artifact_id}")
                completion_parts.append(f"artifact_bytes={result_artifact.size_bytes}")
            elif completion_conv2d_stats_only:
                completion_parts.append(
                    f"conv2d_stats_only=1 stats_element_count={response_payload.stats_element_count} "
                    f"stats_sample_count={len(response_payload.stats_samples)}"
                )
            else:
                completion_parts.append(f"inline_bytes={len(response_payload.output_vector)}")
            write_audit_event(
                " ".join(completion_parts),
                stdout=True,
                logger=self.logger,
            )
            worker_count, client_count = self._cluster_counts()
            return build_response(
                status_code=STATUS_OK,
                method=request.method,
                object_id=request.object_id,
                stream_id=request.stream_id,
                worker_count=worker_count,
                client_count=client_count,
                client_id="",
                iteration_count=request.iteration_count,
                response_payload=response_payload,
                result_artifact=result_artifact,
                timing=timing,
            )
        except (OSError, ValueError, ConnectionError, RuntimeError) as exc:
            worker_count, client_count = self._cluster_counts()
            return build_response(
                status_code=STATUS_INTERNAL_ERROR,
                method=request.method,
                object_id=request.object_id,
                stream_id=request.stream_id,
                error_message=str(exc),
                worker_count=worker_count,
                client_count=client_count,
                client_id="",
                iteration_count=request.iteration_count,
            )
        finally:
            if client_weight_path is not None:
                self.task_exchange.unregister_client_weight_path(request.request_id)
                try:
                    client_weight_path.unlink(missing_ok=True)
                except OSError:
                    pass
