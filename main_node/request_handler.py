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

from adapters.audit_log import write_audit_event
from app.constants import (
    CONV2D_CLIENT_RESPONSE_STATS_ONLY,
    CONV2D_STATS_MAX_SAMPLES,
    MAIN_NODE_NAME,
    METHOD_FREE_CONTENT,
    METHOD_GEMV,
    METHOD_CONV2D,
    STATUS_BAD_REQUEST,
    STATUS_INTERNAL_ERROR,
    STATUS_NOT_FOUND,
    STATUS_OK,
)
from app.trace_utils import trace_function
from compute_node.input_matrix.gemv import build_input_matrix_spec as build_gemv_input_matrix_spec
from compute_node.compute_methods.conv2d.executor import load_named_workload_spec
from wire.internal_protocol.runtime_transport import (
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
        supported_methods = ", ".join((METHOD_GEMV, METHOD_CONV2D, METHOD_FREE_CONTENT))
        return "\n".join(
            (
                "superweb-cluster system overview",
                f"main_node_endpoint: {main_node_host}:{self.config.tcp_port}",
                f"worker_count: {worker_count}",
                f"client_count: {client_count}",
                f"total_effective_gflops: {total_effective_gflops:.3f}",
                f"gemv_effective_gflops: {self._safe_float(method_totals.get(METHOD_GEMV, 0.0)):.3f}",
                f"conv2d_effective_gflops: {self._safe_float(method_totals.get(METHOD_CONV2D, 0.0)):.3f}",
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

        if request.method not in (METHOD_GEMV, METHOD_CONV2D):
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
                try:
                    weight_path = allocation.upload_future.result(
                        timeout=float(self.config.compute_artifact_ttl_seconds),
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

        try:
            completion_conv2d_stats_only = False
            dispatch_done_at = time.monotonic()
            if request.method == METHOD_GEMV:
                task_window_started_at = time.monotonic()
                with ThreadPoolExecutor(max_workers=len(assignments), thread_name_prefix="task-dispatch") as executor:
                    outcomes = list(executor.map(lambda item: self.run_worker_task_slice(request, item), assignments))
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
                if len(output_vector) > self.config.max_message_size:
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
                with ThreadPoolExecutor(max_workers=len(assignments), thread_name_prefix="task-dispatch") as executor:
                    outcomes = list(executor.map(lambda item: self.run_worker_task_slice(request, item), assignments))
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
