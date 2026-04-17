"""Validate client requests, dispatch worker slices, and assemble one response.

Use this module after a client has already joined the main node and sent one
structured request. It owns request validation, worker assignment selection,
artifact decisions, and final response construction.
"""

from __future__ import annotations

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

from app.constants import (
    MAIN_NODE_NAME,
    METHOD_FIXED_MATRIX_VECTOR_MULTIPLICATION,
    METHOD_SPATIAL_CONVOLUTION,
    STATUS_BAD_REQUEST,
    STATUS_INTERNAL_ERROR,
    STATUS_NOT_FOUND,
    STATUS_OK,
)
from app.trace_utils import trace_function
from compute_node.compute_methods.spatial_convolution.executor import load_named_workload_spec
from wire.internal_protocol.runtime_transport import (
    FixedMatrixVectorResponsePayload,
    SpatialConvolutionResponsePayload,
    build_client_response,
)


class ClientRequestHandler:
    """Own the main-node request lifecycle from validation through aggregation."""

    def __init__(
        self,
        *,
        config,
        registry,
        dispatcher,
        aggregator,
        fixed_matvec_spec,
        spatial_dataset_dir,
        task_exchange,
        artifact_manager,
        cluster_counts,
    ) -> None:
        """Wire together the services needed to handle one client request end to end.

        Args: config runtime settings, registry/dispatcher/aggregator cluster services, fixed_matvec_spec and spatial_dataset_dir workload metadata, task_exchange/artifact_manager transfer helpers, cluster_counts callable for reply metadata.
        Returns: None after the handler stores references to its shared collaborators.
        """
        self.config = config
        self.registry = registry
        self.dispatcher = dispatcher
        self.aggregator = aggregator
        self.fixed_matvec_spec = fixed_matvec_spec
        self.spatial_dataset_dir = spatial_dataset_dir
        self.task_exchange = task_exchange
        self.run_worker_task_slice = task_exchange.run_worker_task_slice
        self.artifact_manager = artifact_manager
        self._cluster_counts = cluster_counts

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
    def build_client_response_for_request(self, request):
        """Use this after receiving a validated CLIENT_REQUEST from one client session.

        Args: request decoded client request envelope with method-specific payload.
        Returns: A ready-to-send ClientResponse envelope, including inline bytes or an artifact descriptor.
        """
        worker_count, client_count = self._cluster_counts()

        if request.method not in (METHOD_FIXED_MATRIX_VECTOR_MULTIPLICATION, METHOD_SPATIAL_CONVOLUTION):
            return build_client_response(
                request_id=request.request_id,
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

        if request.method == METHOD_FIXED_MATRIX_VECTOR_MULTIPLICATION and request.vector_length != self.fixed_matvec_spec.cols:
            return build_client_response(
                request_id=request.request_id,
                status_code=STATUS_BAD_REQUEST,
                method=request.method,
                object_id=request.object_id,
                stream_id=request.stream_id,
                error_message=(
                    f"vector_length={request.vector_length} does not match "
                    f"expected {self.fixed_matvec_spec.cols}"
                ),
                worker_count=worker_count,
                client_count=client_count,
                client_id="",
                iteration_count=request.iteration_count,
            )

        if request.method == METHOD_FIXED_MATRIX_VECTOR_MULTIPLICATION and len(request.vector_data) != request.vector_length * 4:
            return build_client_response(
                request_id=request.request_id,
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
            return build_client_response(
                request_id=request.request_id,
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
        if request.method == METHOD_FIXED_MATRIX_VECTOR_MULTIPLICATION:
            assignments = self.dispatcher.dispatch_fixed_matrix_vector_multiplication(
                request_id=request.request_id,
                rows=self.fixed_matvec_spec.rows,
                workers=workers,
                worker_hardware=self.registry.list_worker_hardware(METHOD_FIXED_MATRIX_VECTOR_MULTIPLICATION),
            )
        else:
            try:
                spec, _variant = load_named_workload_spec(request.object_id)
            except ValueError as exc:
                return build_client_response(
                    request_id=request.request_id,
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

            max_channels_per_task = self.task_exchange.max_spatial_channels_per_task(spec)
            if max_channels_per_task <= 0:
                return build_client_response(
                    request_id=request.request_id,
                    status_code=STATUS_BAD_REQUEST,
                    method=request.method,
                    object_id=request.object_id,
                    stream_id=request.stream_id,
                    error_message="spatial_convolution workload is too large for in-band task transport",
                    worker_count=worker_count,
                    client_count=client_count,
                    client_id="",
                    iteration_count=request.iteration_count,
                )

            try:
                self.task_exchange.ensure_spatial_convolution_dataset_ready()
            except (OSError, subprocess.CalledProcessError) as exc:
                return build_client_response(
                    request_id=request.request_id,
                    status_code=STATUS_INTERNAL_ERROR,
                    method=request.method,
                    object_id=request.object_id,
                    stream_id=request.stream_id,
                    error_message=f"failed to prepare spatial_convolution dataset: {exc}",
                    worker_count=worker_count,
                    client_count=client_count,
                    client_id="",
                    iteration_count=request.iteration_count,
                )

            assignments = self.dispatcher.dispatch_spatial_convolution(
                request_id=request.request_id,
                output_channels=spec.c_out,
                workers=workers,
                worker_hardware=self.registry.list_worker_hardware(METHOD_SPATIAL_CONVOLUTION),
                max_channels_per_task=max_channels_per_task,
            )
        if not assignments:
            return build_client_response(
                request_id=request.request_id,
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
            if request.method == METHOD_FIXED_MATRIX_VECTOR_MULTIPLICATION:
                with ThreadPoolExecutor(max_workers=len(assignments), thread_name_prefix="task-dispatch") as executor:
                    task_results = list(executor.map(lambda item: self.run_worker_task_slice(request, item), assignments))
                output_vector = self.aggregator.collect_fixed_matrix_vector_result(
                    rows=self.fixed_matvec_spec.rows,
                    results=task_results,
                )
                output_length = self.fixed_matvec_spec.rows
                result_artifact = None
                if len(output_vector) > self.config.max_message_size:
                    result_artifact = self.artifact_manager.publish_bytes(
                        output_vector,
                        producer_node_id=MAIN_NODE_NAME,
                        content_type="application/octet-stream",
                        artifact_id=request.request_id,
                    )
                    output_vector = b""
                response_payload = FixedMatrixVectorResponsePayload(
                    output_length=output_length,
                    output_vector=output_vector,
                )
            else:
                spec, _variant = load_named_workload_spec(request.object_id)
                output_length = spec.output_h * spec.output_w * spec.c_out
                output_vector = b""
                result_artifact = None
                if self.artifact_manager is None:
                    raise RuntimeError("artifact manager is required for spatial_convolution responses")
                artifact_path = self.artifact_manager.root_dir / f"{request.request_id}.bin"
                with ThreadPoolExecutor(max_workers=len(assignments), thread_name_prefix="task-dispatch") as executor:
                    task_results = list(executor.map(lambda item: self.run_worker_task_slice(request, item), assignments))
                try:
                    self.aggregator.collect_spatial_convolution_result_to_file(
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
                    artifact_id=request.request_id,
                )
                response_payload = SpatialConvolutionResponsePayload(
                    output_length=output_length,
                    output_vector=output_vector,
                    result_artifact_id=result_artifact.artifact_id if result_artifact is not None else "",
                )
            worker_count, client_count = self._cluster_counts()
            return build_client_response(
                request_id=request.request_id,
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
            )
        except (OSError, ValueError, ConnectionError, RuntimeError) as exc:
            worker_count, client_count = self._cluster_counts()
            return build_client_response(
                request_id=request.request_id,
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
