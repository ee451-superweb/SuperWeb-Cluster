"""Convert between Python runtime-envelope objects and generated protobuf messages.

Use this module when runtime code needs the canonical protobuf serializer and
parser for control-plane messages exchanged between clients, workers, and the
main node.
"""

from __future__ import annotations

from core.types import (
    ComputeHardwarePerformance,
    ComputePerformanceSummary,
    HardwareProfile,
    MethodPerformanceSummary,
)
from wire.external_protocol.data_plane import ArtifactDescriptor
from wire.proto import superweb_cluster_runtime_pb2 as runtime_pb2


def _runtime():
    """Use this lazy import helper to avoid a transport/codec circular import.

    Args: no caller-supplied inputs.
    Returns: The ``wire.internal_protocol.transport`` module object.
    """
    from wire.internal_protocol import transport as runtime_messages

    return runtime_messages


def _to_pb_hardware_profile(payload: HardwareProfile) -> runtime_pb2.HardwareProfile:
    """Use this during protobuf encoding to convert a HardwareProfile into generated protobuf form.

    Args: payload internal HardwareProfile object.
    Returns: The generated protobuf HardwareProfile message.
    """
    return runtime_pb2.HardwareProfile(
        hostname=payload.hostname,
        local_ip=payload.local_ip,
        mac_address=payload.mac_address,
        system=payload.system,
        release=payload.release,
        machine=payload.machine,
        processor=payload.processor,
        logical_cpu_count=payload.logical_cpu_count,
        memory_bytes=payload.memory_bytes,
    )


def _from_pb_hardware_profile(payload: runtime_pb2.HardwareProfile) -> HardwareProfile:
    """Use this during protobuf parsing to convert a generated hardware profile into the internal model.

    Args: payload generated protobuf HardwareProfile message.
    Returns: The internal HardwareProfile object.
    """
    return HardwareProfile(
        hostname=payload.hostname,
        local_ip=payload.local_ip,
        mac_address=payload.mac_address,
        system=payload.system,
        release=payload.release,
        machine=payload.machine,
        processor=payload.processor,
        logical_cpu_count=payload.logical_cpu_count,
        memory_bytes=payload.memory_bytes,
    )


def _to_pb_compute_hardware_performance(
    payload: ComputeHardwarePerformance,
) -> runtime_pb2.ComputeHardwarePerformance:
    """Use this during protobuf encoding to convert one hardware-performance row.

    Args: payload internal ComputeHardwarePerformance row.
    Returns: The generated protobuf ComputeHardwarePerformance message.
    """
    return runtime_pb2.ComputeHardwarePerformance(
        hardware_type=payload.hardware_type,
        effective_gflops=payload.effective_gflops,
        rank=payload.rank,
    )


def _from_pb_compute_hardware_performance(
    payload: runtime_pb2.ComputeHardwarePerformance,
) -> ComputeHardwarePerformance:
    """Use this during protobuf parsing to convert one generated hardware-performance row.

    Args: payload generated protobuf ComputeHardwarePerformance message.
    Returns: The internal ComputeHardwarePerformance object.
    """
    return ComputeHardwarePerformance(
        hardware_type=payload.hardware_type,
        effective_gflops=payload.effective_gflops,
        rank=payload.rank,
    )


def _to_pb_method_performance_summary(payload: MethodPerformanceSummary) -> runtime_pb2.MethodPerformanceSummary:
    """Use this during protobuf encoding to convert one per-method performance summary.

    Args: payload internal MethodPerformanceSummary object.
    Returns: The generated protobuf MethodPerformanceSummary message.
    """
    message = runtime_pb2.MethodPerformanceSummary(
        method=payload.method,
        hardware_count=payload.hardware_count,
    )
    message.ranked_hardware.extend(_to_pb_compute_hardware_performance(item) for item in payload.ranked_hardware)
    return message


def _from_pb_method_performance_summary(payload: runtime_pb2.MethodPerformanceSummary) -> MethodPerformanceSummary:
    """Use this during protobuf parsing to convert one generated method performance summary.

    Args: payload generated protobuf MethodPerformanceSummary message.
    Returns: The internal MethodPerformanceSummary object.
    """
    return MethodPerformanceSummary(
        method=payload.method,
        hardware_count=payload.hardware_count,
        ranked_hardware=[_from_pb_compute_hardware_performance(item) for item in payload.ranked_hardware],
    )


def _to_pb_compute_performance_summary(payload: ComputePerformanceSummary) -> runtime_pb2.ComputePerformanceSummary:
    """Use this during protobuf encoding to convert a full abstract performance summary.

    Args: payload internal ComputePerformanceSummary object.
    Returns: The generated protobuf ComputePerformanceSummary message.
    """
    message = runtime_pb2.ComputePerformanceSummary(hardware_count=payload.hardware_count)
    message.ranked_hardware.extend(_to_pb_compute_hardware_performance(item) for item in payload.ranked_hardware)
    message.method_summaries.extend(_to_pb_method_performance_summary(item) for item in payload.method_summaries)
    return message


def _from_pb_compute_performance_summary(payload: runtime_pb2.ComputePerformanceSummary) -> ComputePerformanceSummary:
    """Use this during protobuf parsing to convert a generated performance summary.

    Args: payload generated protobuf ComputePerformanceSummary message.
    Returns: The internal ComputePerformanceSummary object.
    """
    return ComputePerformanceSummary(
        hardware_count=payload.hardware_count,
        ranked_hardware=[_from_pb_compute_hardware_performance(item) for item in payload.ranked_hardware],
        method_summaries=[_from_pb_method_performance_summary(item) for item in payload.method_summaries],
    )


def _to_pb_artifact_descriptor(payload: ArtifactDescriptor) -> runtime_pb2.ArtifactDescriptor:
    """Use this during protobuf encoding to convert an ArtifactDescriptor.

    Args: payload internal artifact descriptor.
    Returns: The generated protobuf ArtifactDescriptor message.
    """
    return runtime_pb2.ArtifactDescriptor(
        artifact_id=payload.artifact_id,
        content_type=payload.content_type,
        size_bytes=payload.size_bytes,
        checksum=payload.checksum,
        producer_node_id=payload.producer_node_id,
        transfer_host=payload.transfer_host,
        transfer_port=payload.transfer_port,
        chunk_size=payload.chunk_size,
        ready=payload.ready,
    )


def _from_pb_artifact_descriptor(payload: runtime_pb2.ArtifactDescriptor) -> ArtifactDescriptor:
    """Use this during protobuf parsing to convert a generated ArtifactDescriptor.

    Args: payload generated protobuf ArtifactDescriptor message.
    Returns: The internal ArtifactDescriptor object.
    """
    return ArtifactDescriptor(
        artifact_id=payload.artifact_id,
        content_type=payload.content_type,
        size_bytes=payload.size_bytes,
        checksum=payload.checksum,
        producer_node_id=payload.producer_node_id,
        transfer_host=payload.transfer_host,
        transfer_port=payload.transfer_port,
        chunk_size=payload.chunk_size,
        ready=payload.ready,
    )


def _to_pb_gemv_request_payload(payload) -> runtime_pb2.GemvRequestPayload:
    """Use this during protobuf encoding to convert an GEMV client-request payload.

    Args: payload internal GemvRequestPayload object.
    Returns: The generated protobuf request payload message.
    """
    return runtime_pb2.GemvRequestPayload(
        vector_length=payload.vector_length,
        vector_data=payload.vector_data,
    )


def _from_pb_gemv_request_payload(payload: runtime_pb2.GemvRequestPayload):
    """Use this during protobuf parsing to convert a generated GEMV request payload.

    Args: payload generated protobuf GemvRequestPayload message.
    Returns: The internal GemvRequestPayload object.
    """
    runtime = _runtime()
    return runtime.GemvRequestPayload(
        vector_length=payload.vector_length,
        vector_data=payload.vector_data,
    )


def _to_pb_conv2d_request_payload(payload) -> runtime_pb2.Conv2dRequestPayload:
    """Use this during protobuf encoding to convert a conv2d client-request payload.

    Args: payload internal Conv2dRequestPayload object.
    Returns: The generated protobuf request payload message.
    """
    return runtime_pb2.Conv2dRequestPayload(
        tensor_h=payload.tensor_h,
        tensor_w=payload.tensor_w,
        channels_in=payload.channels_in,
        channels_out=payload.channels_out,
        kernel_size=payload.kernel_size,
        padding=payload.padding,
        stride=payload.stride,
        client_response_mode=payload.client_response_mode,
        stats_max_samples=payload.stats_max_samples,
        upload_size_bytes=getattr(payload, "upload_size_bytes", 0),
        upload_checksum=getattr(payload, "upload_checksum", ""),
    )


def _from_pb_conv2d_request_payload(payload: runtime_pb2.Conv2dRequestPayload):
    """Use this during protobuf parsing to convert a generated conv2d request payload.

    Args: payload generated protobuf Conv2dRequestPayload message.
    Returns: The internal Conv2dRequestPayload object.
    """
    runtime = _runtime()
    return runtime.Conv2dRequestPayload(
        tensor_h=payload.tensor_h,
        tensor_w=payload.tensor_w,
        channels_in=payload.channels_in,
        channels_out=payload.channels_out,
        kernel_size=payload.kernel_size,
        padding=payload.padding,
        stride=payload.stride,
        client_response_mode=payload.client_response_mode,
        stats_max_samples=payload.stats_max_samples,
        upload_size_bytes=payload.upload_size_bytes,
        upload_checksum=payload.upload_checksum,
    )


def _to_pb_gemv_response_payload(payload) -> runtime_pb2.GemvResponsePayload:
    """Use this during protobuf encoding to convert an GEMV client-response payload.

    Args: payload internal GemvResponsePayload object.
    Returns: The generated protobuf response payload message.
    """
    return runtime_pb2.GemvResponsePayload(
        output_length=payload.output_length,
        output_vector=payload.output_vector,
    )


def _from_pb_gemv_response_payload(payload: runtime_pb2.GemvResponsePayload):
    """Use this during protobuf parsing to convert a generated GEMV response payload.

    Args: payload generated protobuf GemvResponsePayload message.
    Returns: The internal GemvResponsePayload object.
    """
    runtime = _runtime()
    return runtime.GemvResponsePayload(
        output_length=payload.output_length,
        output_vector=payload.output_vector,
    )


def _to_pb_conv2d_response_payload(payload) -> runtime_pb2.Conv2dResponsePayload:
    """Use this during protobuf encoding to convert a conv2d client-response payload.

    Args: payload internal Conv2dResponsePayload object.
    Returns: The generated protobuf response payload message.
    """
    return runtime_pb2.Conv2dResponsePayload(
        output_length=payload.output_length,
        output_vector=payload.output_vector,
        result_artifact_id=payload.result_artifact_id,
        stats_element_count=payload.stats_element_count,
        stats_sum=payload.stats_sum,
        stats_sum_squares=payload.stats_sum_squares,
        stats_samples=list(payload.stats_samples),
    )


def _from_pb_conv2d_response_payload(payload: runtime_pb2.Conv2dResponsePayload):
    """Use this during protobuf parsing to convert a generated conv2d response payload.

    Args: payload generated protobuf Conv2dResponsePayload message.
    Returns: The internal Conv2dResponsePayload object.
    """
    runtime = _runtime()
    return runtime.Conv2dResponsePayload(
        output_length=payload.output_length,
        output_vector=payload.output_vector,
        result_artifact_id=payload.result_artifact_id,
        stats_element_count=payload.stats_element_count,
        stats_sum=payload.stats_sum,
        stats_sum_squares=payload.stats_sum_squares,
        stats_samples=tuple(payload.stats_samples),
    )


def _to_pb_worker_timing(payload) -> runtime_pb2.WorkerTiming:
    """Encode one per-worker timing entry into protobuf form."""
    return runtime_pb2.WorkerTiming(
        node_id=payload.node_id,
        task_id=payload.task_id,
        slice=payload.slice,
        wall_ms=payload.wall_ms,
        artifact_fetch_ms=payload.artifact_fetch_ms,
        computation_ms=payload.computation_ms,
        peripheral_ms=payload.peripheral_ms,
    )


def _from_pb_worker_timing(payload: runtime_pb2.WorkerTiming):
    """Decode one per-worker timing entry from protobuf form."""
    runtime = _runtime()
    return runtime.WorkerTiming(
        node_id=payload.node_id,
        task_id=payload.task_id,
        slice=payload.slice,
        wall_ms=payload.wall_ms,
        artifact_fetch_ms=payload.artifact_fetch_ms,
        computation_ms=payload.computation_ms,
        peripheral_ms=payload.peripheral_ms,
    )


def _to_pb_response_timing(payload) -> runtime_pb2.ResponseTiming:
    """Encode the response-level timing breakdown into protobuf form."""
    encoded = runtime_pb2.ResponseTiming(
        dispatch_ms=payload.dispatch_ms,
        task_window_ms=payload.task_window_ms,
        aggregate_ms=payload.aggregate_ms,
    )
    encoded.workers.extend(_to_pb_worker_timing(worker) for worker in payload.workers)
    return encoded


def _from_pb_response_timing(payload: runtime_pb2.ResponseTiming):
    """Decode the response-level timing breakdown from protobuf form."""
    runtime = _runtime()
    return runtime.ResponseTiming(
        dispatch_ms=payload.dispatch_ms,
        task_window_ms=payload.task_window_ms,
        aggregate_ms=payload.aggregate_ms,
        workers=tuple(_from_pb_worker_timing(worker) for worker in payload.workers),
    )


def _to_pb_gemm_request_payload(payload) -> runtime_pb2.GemmRequestPayload:
    """Encode a GEMM client-request payload into protobuf form."""
    return runtime_pb2.GemmRequestPayload()


def _from_pb_gemm_request_payload(payload: runtime_pb2.GemmRequestPayload):
    """Decode a generated GEMM client-request payload into the internal model."""
    runtime = _runtime()
    return runtime.GemmRequestPayload()


def _to_pb_gemm_response_payload(payload) -> runtime_pb2.GemmResponsePayload:
    """Encode a GEMM client-response payload into protobuf form."""
    return runtime_pb2.GemmResponsePayload(
        output_length=payload.output_length,
        output_vector=payload.output_vector,
    )


def _from_pb_gemm_response_payload(payload: runtime_pb2.GemmResponsePayload):
    """Decode a generated GEMM client-response payload into the internal model."""
    runtime = _runtime()
    return runtime.GemmResponsePayload(
        output_length=payload.output_length,
        output_vector=payload.output_vector,
    )


def _to_pb_gemm_task_payload(payload) -> runtime_pb2.GemmTaskPayload:
    """Encode a GEMM task payload into protobuf form."""
    return runtime_pb2.GemmTaskPayload(
        m_start=payload.m_start,
        m_end=payload.m_end,
        m=payload.m,
        n=payload.n,
        k=payload.k,
    )


def _from_pb_gemm_task_payload(payload: runtime_pb2.GemmTaskPayload):
    """Decode a generated GEMM task payload into the internal model."""
    runtime = _runtime()
    return runtime.GemmTaskPayload(
        m_start=payload.m_start,
        m_end=payload.m_end,
        m=payload.m,
        n=payload.n,
        k=payload.k,
    )


def _to_pb_gemm_result_payload(payload) -> runtime_pb2.GemmResultPayload:
    """Encode a GEMM task-result payload into protobuf form."""
    return runtime_pb2.GemmResultPayload(
        m_start=payload.m_start,
        m_end=payload.m_end,
        output_length=payload.output_length,
        output_vector=payload.output_vector,
    )


def _from_pb_gemm_result_payload(payload: runtime_pb2.GemmResultPayload):
    """Decode a generated GEMM task-result payload into the internal model."""
    runtime = _runtime()
    return runtime.GemmResultPayload(
        m_start=payload.m_start,
        m_end=payload.m_end,
        output_length=payload.output_length,
        output_vector=payload.output_vector,
    )


def _to_pb_gemv_task_payload(payload) -> runtime_pb2.GemvTaskPayload:
    """Use this during protobuf encoding to convert an GEMV task payload.

    Args: payload internal GemvTaskPayload object.
    Returns: The generated protobuf task payload message.
    """
    return runtime_pb2.GemvTaskPayload(
        row_start=payload.row_start,
        row_end=payload.row_end,
        vector_length=payload.vector_length,
        vector_data=payload.vector_data,
    )


def _from_pb_gemv_task_payload(payload: runtime_pb2.GemvTaskPayload):
    """Use this during protobuf parsing to convert a generated GEMV task payload.

    Args: payload generated protobuf GemvTaskPayload message.
    Returns: The internal GemvTaskPayload object.
    """
    runtime = _runtime()
    return runtime.GemvTaskPayload(
        row_start=payload.row_start,
        row_end=payload.row_end,
        vector_length=payload.vector_length,
        vector_data=payload.vector_data,
    )


def _to_pb_conv2d_task_payload(payload) -> runtime_pb2.Conv2dTaskPayload:
    """Use this during protobuf encoding to convert a conv2d task payload.

    Args: payload internal Conv2dTaskPayload object.
    Returns: The generated protobuf task payload message.
    """
    message = runtime_pb2.Conv2dTaskPayload(
        start_oc=payload.start_oc,
        end_oc=payload.end_oc,
        tensor_h=payload.tensor_h,
        tensor_w=payload.tensor_w,
        channels_in=payload.channels_in,
        channels_out=payload.channels_out,
        kernel_size=payload.kernel_size,
        padding=payload.padding,
        stride=payload.stride,
        weight_data=payload.weight_data,
        client_response_mode=payload.client_response_mode,
        stats_max_samples=payload.stats_max_samples,
    )
    if getattr(payload, "weight_artifact", None) is not None:
        message.weight_artifact.CopyFrom(_to_pb_artifact_descriptor(payload.weight_artifact))
    return message


def _from_pb_conv2d_task_payload(payload: runtime_pb2.Conv2dTaskPayload):
    """Use this during protobuf parsing to convert a generated conv2d task payload.

    Args: payload generated protobuf Conv2dTaskPayload message.
    Returns: The internal Conv2dTaskPayload object.
    """
    runtime = _runtime()
    return runtime.Conv2dTaskPayload(
        start_oc=payload.start_oc,
        end_oc=payload.end_oc,
        tensor_h=payload.tensor_h,
        tensor_w=payload.tensor_w,
        channels_in=payload.channels_in,
        channels_out=payload.channels_out,
        kernel_size=payload.kernel_size,
        padding=payload.padding,
        stride=payload.stride,
        weight_data=payload.weight_data,
        client_response_mode=payload.client_response_mode,
        stats_max_samples=payload.stats_max_samples,
        weight_artifact=(
            _from_pb_artifact_descriptor(payload.weight_artifact)
            if payload.HasField("weight_artifact")
            else None
        ),
    )


def _to_pb_gemv_result_payload(payload) -> runtime_pb2.GemvResultPayload:
    """Use this during protobuf encoding to convert an GEMV task-result payload.

    Args: payload internal GemvResultPayload object.
    Returns: The generated protobuf result payload message.
    """
    return runtime_pb2.GemvResultPayload(
        row_start=payload.row_start,
        row_end=payload.row_end,
        output_length=payload.output_length,
        output_vector=payload.output_vector,
    )


def _from_pb_gemv_result_payload(payload: runtime_pb2.GemvResultPayload):
    """Use this during protobuf parsing to convert a generated GEMV result payload.

    Args: payload generated protobuf GemvResultPayload message.
    Returns: The internal GemvResultPayload object.
    """
    runtime = _runtime()
    return runtime.GemvResultPayload(
        row_start=payload.row_start,
        row_end=payload.row_end,
        output_length=payload.output_length,
        output_vector=payload.output_vector,
    )


def _to_pb_conv2d_result_payload(payload) -> runtime_pb2.Conv2dResultPayload:
    """Use this during protobuf encoding to convert a conv2d task-result payload.

    Args: payload internal Conv2dResultPayload object.
    Returns: The generated protobuf result payload message.
    """
    return runtime_pb2.Conv2dResultPayload(
        start_oc=payload.start_oc,
        end_oc=payload.end_oc,
        output_h=payload.output_h,
        output_w=payload.output_w,
        output_length=payload.output_length,
        output_vector=payload.output_vector,
        result_artifact_id=payload.result_artifact_id,
        stats_element_count=payload.stats_element_count,
        stats_sum=payload.stats_sum,
        stats_sum_squares=payload.stats_sum_squares,
        stats_samples=list(payload.stats_samples),
    )


def _from_pb_conv2d_result_payload(payload: runtime_pb2.Conv2dResultPayload):
    """Use this during protobuf parsing to convert a generated conv2d result payload.

    Args: payload generated protobuf Conv2dResultPayload message.
    Returns: The internal Conv2dResultPayload object.
    """
    runtime = _runtime()
    return runtime.Conv2dResultPayload(
        start_oc=payload.start_oc,
        end_oc=payload.end_oc,
        output_h=payload.output_h,
        output_w=payload.output_w,
        output_length=payload.output_length,
        output_vector=payload.output_vector,
        result_artifact_id=payload.result_artifact_id,
        stats_element_count=payload.stats_element_count,
        stats_sum=payload.stats_sum,
        stats_sum_squares=payload.stats_sum_squares,
        stats_samples=tuple(payload.stats_samples),
    )


def encode_envelope(message) -> bytes:
    """Use this when one internal RuntimeEnvelope should become wire protobuf bytes.

    Args: message internal runtime envelope object to serialize.
    Returns: The serialized protobuf payload bytes for that envelope.
    """

    runtime = _runtime()
    envelope = runtime_pb2.RuntimeEnvelope(kind=int(message.kind))

    if message.kind == runtime.MessageKind.REGISTER_WORKER:
        if message.register_worker is None:
            raise ValueError("REGISTER_WORKER envelope missing payload")
        envelope.register_worker.node_name = message.register_worker.node_name
        envelope.register_worker.hardware.CopyFrom(_to_pb_hardware_profile(message.register_worker.hardware))
        envelope.register_worker.performance.CopyFrom(_to_pb_compute_performance_summary(message.register_worker.performance))
    elif message.kind == runtime.MessageKind.REGISTER_OK:
        if message.register_ok is None:
            raise ValueError("REGISTER_OK envelope missing payload")
        envelope.register_ok.main_node_name = message.register_ok.main_node_name
        envelope.register_ok.main_node_ip = message.register_ok.main_node_ip
        envelope.register_ok.main_node_port = message.register_ok.main_node_port
        envelope.register_ok.node_id = message.register_ok.node_id
    elif message.kind == runtime.MessageKind.HEARTBEAT:
        if message.heartbeat is None:
            raise ValueError("HEARTBEAT envelope missing payload")
        envelope.heartbeat.main_node_name = message.heartbeat.main_node_name
        envelope.heartbeat.unix_time_ms = message.heartbeat.unix_time_ms
    elif message.kind == runtime.MessageKind.HEARTBEAT_OK:
        if message.heartbeat_ok is None:
            raise ValueError("HEARTBEAT_OK envelope missing payload")
        envelope.heartbeat_ok.node_name = message.heartbeat_ok.node_name
        envelope.heartbeat_ok.heartbeat_unix_time_ms = message.heartbeat_ok.heartbeat_unix_time_ms
        envelope.heartbeat_ok.received_unix_time_ms = message.heartbeat_ok.received_unix_time_ms
        envelope.heartbeat_ok.active_task_ids.extend(message.heartbeat_ok.active_task_ids)
        envelope.heartbeat_ok.node_status = int(message.heartbeat_ok.node_status)
        envelope.heartbeat_ok.completed_task_count = message.heartbeat_ok.completed_task_count
        envelope.heartbeat_ok.node_id = message.heartbeat_ok.node_id
    elif message.kind == runtime.MessageKind.CLIENT_JOIN:
        if message.client_join is None:
            raise ValueError("CLIENT_JOIN envelope missing payload")
        envelope.client_join.client_name = message.client_join.client_name
    elif message.kind == runtime.MessageKind.CLIENT_INFO_REQUEST:
        if message.client_info_request is None:
            raise ValueError("CLIENT_INFO_REQUEST envelope missing payload")
        envelope.client_info_request.client_id = message.client_info_request.client_id
        envelope.client_info_request.client_name = message.client_info_request.client_name
        envelope.client_info_request.timestamp_ms = message.client_info_request.timestamp_ms
    elif message.kind == runtime.MessageKind.CLIENT_INFO_REPLY:
        if message.client_info_reply is None:
            raise ValueError("CLIENT_INFO_REPLY envelope missing payload")
        envelope.client_info_reply.client_id = message.client_info_reply.client_id
        envelope.client_info_reply.request_timestamp_ms = message.client_info_reply.request_timestamp_ms
        envelope.client_info_reply.reply_timestamp_ms = message.client_info_reply.reply_timestamp_ms
        envelope.client_info_reply.timeout_ms = message.client_info_reply.timeout_ms
        envelope.client_info_reply.has_active_tasks = message.client_info_reply.has_active_tasks
        envelope.client_info_reply.active_task_ids.extend(message.client_info_reply.active_task_ids)
    elif message.kind == runtime.MessageKind.CLIENT_REQUEST_OK:
        if message.client_request_ok is None:
            raise ValueError("CLIENT_REQUEST_OK envelope missing payload")
        envelope.client_request_ok.client_id = message.client_request_ok.client_id
        envelope.client_request_ok.task_id = message.client_request_ok.task_id
        envelope.client_request_ok.method = message.client_request_ok.method
        envelope.client_request_ok.size = message.client_request_ok.size
        envelope.client_request_ok.object_id = message.client_request_ok.object_id
        envelope.client_request_ok.accepted_timestamp_ms = message.client_request_ok.accepted_timestamp_ms
        envelope.client_request_ok.upload_id = message.client_request_ok.upload_id
        envelope.client_request_ok.download_id = message.client_request_ok.download_id
        envelope.client_request_ok.data_endpoint_host = message.client_request_ok.data_endpoint_host
        envelope.client_request_ok.data_endpoint_port = message.client_request_ok.data_endpoint_port
    elif message.kind == runtime.MessageKind.CLIENT_REQUEST:
        if message.client_request is None:
            raise ValueError("CLIENT_REQUEST envelope missing payload")
        envelope.client_request.request_id = message.client_request.request_id
        envelope.client_request.client_name = message.client_request.client_name
        envelope.client_request.method = message.client_request.method
        envelope.client_request.size = message.client_request.size
        envelope.client_request.object_id = message.client_request.object_id
        envelope.client_request.stream_id = message.client_request.stream_id
        envelope.client_request.timestamp_ms = message.client_request.timestamp_ms
        envelope.client_request.iteration_count = message.client_request.iteration_count
        if message.client_request.gemv_payload is not None:
            envelope.client_request.gemv.CopyFrom(
                _to_pb_gemv_request_payload(
                    message.client_request.gemv_payload
                )
            )
        elif message.client_request.conv2d_payload is not None:
            envelope.client_request.conv2d.CopyFrom(
                _to_pb_conv2d_request_payload(message.client_request.conv2d_payload)
            )
        elif message.client_request.gemm_payload is not None:
            envelope.client_request.gemm.CopyFrom(
                _to_pb_gemm_request_payload(message.client_request.gemm_payload)
            )
    elif message.kind == runtime.MessageKind.CLIENT_RESPONSE:
        if message.client_response is None:
            raise ValueError("CLIENT_RESPONSE envelope missing payload")
        envelope.client_response.request_id = message.client_response.request_id
        envelope.client_response.method = message.client_response.method
        envelope.client_response.size = message.client_response.size
        envelope.client_response.object_id = message.client_response.object_id
        envelope.client_response.stream_id = message.client_response.stream_id
        envelope.client_response.timestamp_ms = message.client_response.timestamp_ms
        envelope.client_response.status_code = message.client_response.status_code
        envelope.client_response.error_message = message.client_response.error_message
        envelope.client_response.worker_count = message.client_response.worker_count
        envelope.client_response.client_count = message.client_response.client_count
        envelope.client_response.client_id = message.client_response.client_id
        envelope.client_response.iteration_count = message.client_response.iteration_count
        envelope.client_response.task_id = message.client_response.task_id
        envelope.client_response.elapsed_ms = message.client_response.elapsed_ms
        if message.client_response.gemv_payload is not None:
            envelope.client_response.gemv.CopyFrom(
                _to_pb_gemv_response_payload(
                    message.client_response.gemv_payload
                )
            )
        elif message.client_response.conv2d_payload is not None:
            envelope.client_response.conv2d.CopyFrom(
                _to_pb_conv2d_response_payload(message.client_response.conv2d_payload)
            )
        elif message.client_response.gemm_payload is not None:
            envelope.client_response.gemm.CopyFrom(
                _to_pb_gemm_response_payload(message.client_response.gemm_payload)
            )
        if message.client_response.result_artifact is not None:
            envelope.client_response.result_artifact.CopyFrom(
                _to_pb_artifact_descriptor(message.client_response.result_artifact)
            )
        if message.client_response.timing is not None:
            envelope.client_response.timing.CopyFrom(
                _to_pb_response_timing(message.client_response.timing)
            )
    elif message.kind == runtime.MessageKind.TASK_ASSIGN:
        if message.task_assign is None:
            raise ValueError("TASK_ASSIGN envelope missing payload")
        envelope.task_assign.request_id = message.task_assign.request_id
        envelope.task_assign.node_id = message.task_assign.node_id
        envelope.task_assign.task_id = message.task_assign.task_id
        envelope.task_assign.method = message.task_assign.method
        envelope.task_assign.size = message.task_assign.size
        envelope.task_assign.object_id = message.task_assign.object_id
        envelope.task_assign.stream_id = message.task_assign.stream_id
        envelope.task_assign.timestamp_ms = message.task_assign.timestamp_ms
        envelope.task_assign.iteration_count = message.task_assign.iteration_count
        envelope.task_assign.transfer_mode = int(message.task_assign.transfer_mode)
        envelope.task_assign.artifact_id = message.task_assign.artifact_id
        envelope.task_assign.artifact_timeout_ms = message.task_assign.artifact_timeout_ms
        if message.task_assign.gemv_payload is not None:
            envelope.task_assign.gemv.CopyFrom(
                _to_pb_gemv_task_payload(
                    message.task_assign.gemv_payload
                )
            )
        elif message.task_assign.conv2d_payload is not None:
            envelope.task_assign.conv2d.CopyFrom(
                _to_pb_conv2d_task_payload(message.task_assign.conv2d_payload)
            )
        elif message.task_assign.gemm_payload is not None:
            envelope.task_assign.gemm.CopyFrom(
                _to_pb_gemm_task_payload(message.task_assign.gemm_payload)
            )
    elif message.kind == runtime.MessageKind.TASK_ACCEPT:
        if message.task_accept is None:
            raise ValueError("TASK_ACCEPT envelope missing payload")
        envelope.task_accept.request_id = message.task_accept.request_id
        envelope.task_accept.node_id = message.task_accept.node_id
        envelope.task_accept.task_id = message.task_accept.task_id
        envelope.task_accept.timestamp_ms = message.task_accept.timestamp_ms
        envelope.task_accept.status_code = message.task_accept.status_code
    elif message.kind == runtime.MessageKind.TASK_FAIL:
        if message.task_fail is None:
            raise ValueError("TASK_FAIL envelope missing payload")
        envelope.task_fail.request_id = message.task_fail.request_id
        envelope.task_fail.node_id = message.task_fail.node_id
        envelope.task_fail.task_id = message.task_fail.task_id
        envelope.task_fail.timestamp_ms = message.task_fail.timestamp_ms
        envelope.task_fail.status_code = message.task_fail.status_code
        envelope.task_fail.error_message = message.task_fail.error_message
    elif message.kind == runtime.MessageKind.TASK_RESULT:
        if message.task_result is None:
            raise ValueError("TASK_RESULT envelope missing payload")
        envelope.task_result.request_id = message.task_result.request_id
        envelope.task_result.node_id = message.task_result.node_id
        envelope.task_result.task_id = message.task_result.task_id
        envelope.task_result.timestamp_ms = message.task_result.timestamp_ms
        envelope.task_result.status_code = message.task_result.status_code
        envelope.task_result.iteration_count = message.task_result.iteration_count
        envelope.task_result.computation_ms = message.task_result.computation_ms
        envelope.task_result.peripheral_ms = message.task_result.peripheral_ms
        if message.task_result.gemv_payload is not None:
            envelope.task_result.gemv.CopyFrom(
                _to_pb_gemv_result_payload(
                    message.task_result.gemv_payload
                )
            )
        elif message.task_result.conv2d_payload is not None:
            envelope.task_result.conv2d.CopyFrom(
                _to_pb_conv2d_result_payload(message.task_result.conv2d_payload)
            )
        elif message.task_result.gemm_payload is not None:
            envelope.task_result.gemm.CopyFrom(
                _to_pb_gemm_result_payload(message.task_result.gemm_payload)
            )
        if message.task_result.result_artifact is not None:
            envelope.task_result.result_artifact.CopyFrom(
                _to_pb_artifact_descriptor(message.task_result.result_artifact)
            )
    elif message.kind == runtime.MessageKind.ARTIFACT_RELEASE:
        if message.artifact_release is None:
            raise ValueError("ARTIFACT_RELEASE envelope missing payload")
        envelope.artifact_release.node_id = message.artifact_release.node_id
        envelope.artifact_release.task_id = message.artifact_release.task_id
        envelope.artifact_release.artifact_id = message.artifact_release.artifact_id
        envelope.artifact_release.timestamp_ms = message.artifact_release.timestamp_ms
    elif message.kind == runtime.MessageKind.WORKER_UPDATE:
        if message.worker_update is None:
            raise ValueError("WORKER_UPDATE envelope missing payload")
        envelope.worker_update.node_id = message.worker_update.node_id
        envelope.worker_update.timestamp_ms = message.worker_update.timestamp_ms
        envelope.worker_update.performance.CopyFrom(
            _to_pb_compute_performance_summary(message.worker_update.performance)
        )

    return envelope.SerializeToString()


def parse_envelope(payload: bytes):
    """Use this when incoming protobuf bytes should be converted into the internal runtime model.

    Args: payload raw protobuf bytes from the runtime TCP stream.
    Returns: The decoded internal RuntimeEnvelope object.
    """

    runtime = _runtime()
    envelope_pb = runtime_pb2.RuntimeEnvelope()
    envelope_pb.ParseFromString(payload)

    kind = runtime.MessageKind(envelope_pb.kind)
    register_worker = None
    register_ok = None
    heartbeat = None
    heartbeat_ok = None
    client_join = None
    client_info_request = None
    client_info_reply = None
    client_request_ok = None
    client_request = None
    client_response = None
    task_assign = None
    task_accept = None
    task_fail = None
    task_result = None
    artifact_release = None
    worker_update = None

    if envelope_pb.HasField("register_worker"):
        register_worker = runtime.RegisterWorker(
            node_name=envelope_pb.register_worker.node_name,
            hardware=_from_pb_hardware_profile(envelope_pb.register_worker.hardware),
            performance=_from_pb_compute_performance_summary(envelope_pb.register_worker.performance),
        )
    if envelope_pb.HasField("register_ok"):
        register_ok = runtime.RegisterOk(
            main_node_name=envelope_pb.register_ok.main_node_name,
            main_node_ip=envelope_pb.register_ok.main_node_ip,
            main_node_port=envelope_pb.register_ok.main_node_port,
            node_id=envelope_pb.register_ok.node_id,
        )
    if envelope_pb.HasField("heartbeat"):
        heartbeat = runtime.Heartbeat(
            main_node_name=envelope_pb.heartbeat.main_node_name,
            unix_time_ms=envelope_pb.heartbeat.unix_time_ms,
        )
    if envelope_pb.HasField("heartbeat_ok"):
        heartbeat_ok = runtime.HeartbeatOk(
            node_name=envelope_pb.heartbeat_ok.node_name,
            heartbeat_unix_time_ms=envelope_pb.heartbeat_ok.heartbeat_unix_time_ms,
            received_unix_time_ms=envelope_pb.heartbeat_ok.received_unix_time_ms,
            active_task_ids=tuple(envelope_pb.heartbeat_ok.active_task_ids),
            node_status=runtime.NodeStatus(envelope_pb.heartbeat_ok.node_status),
            completed_task_count=envelope_pb.heartbeat_ok.completed_task_count,
            node_id=envelope_pb.heartbeat_ok.node_id,
        )
    if envelope_pb.HasField("client_join"):
        client_join = runtime.ClientJoin(client_name=envelope_pb.client_join.client_name)
    if envelope_pb.HasField("client_info_request"):
        client_info_request = runtime.ClientInfoRequest(
            client_id=envelope_pb.client_info_request.client_id,
            client_name=envelope_pb.client_info_request.client_name,
            timestamp_ms=envelope_pb.client_info_request.timestamp_ms,
        )
    if envelope_pb.HasField("client_info_reply"):
        client_info_reply = runtime.ClientInfoReply(
            client_id=envelope_pb.client_info_reply.client_id,
            request_timestamp_ms=envelope_pb.client_info_reply.request_timestamp_ms,
            reply_timestamp_ms=envelope_pb.client_info_reply.reply_timestamp_ms,
            timeout_ms=envelope_pb.client_info_reply.timeout_ms,
            has_active_tasks=envelope_pb.client_info_reply.has_active_tasks,
            active_task_ids=tuple(envelope_pb.client_info_reply.active_task_ids),
        )
    if envelope_pb.HasField("client_request_ok"):
        client_request_ok = runtime.ClientRequestOk(
            client_id=envelope_pb.client_request_ok.client_id,
            task_id=envelope_pb.client_request_ok.task_id,
            method=envelope_pb.client_request_ok.method,
            size=envelope_pb.client_request_ok.size,
            object_id=envelope_pb.client_request_ok.object_id,
            accepted_timestamp_ms=envelope_pb.client_request_ok.accepted_timestamp_ms,
            upload_id=envelope_pb.client_request_ok.upload_id,
            download_id=envelope_pb.client_request_ok.download_id,
            data_endpoint_host=envelope_pb.client_request_ok.data_endpoint_host,
            data_endpoint_port=envelope_pb.client_request_ok.data_endpoint_port,
        )
    if envelope_pb.HasField("client_request"):
        request_payload = None
        if envelope_pb.client_request.HasField("gemv"):
            request_payload = _from_pb_gemv_request_payload(
                envelope_pb.client_request.gemv
            )
        elif envelope_pb.client_request.HasField("conv2d"):
            request_payload = _from_pb_conv2d_request_payload(
                envelope_pb.client_request.conv2d
            )
        elif envelope_pb.client_request.HasField("gemm"):
            request_payload = _from_pb_gemm_request_payload(
                envelope_pb.client_request.gemm
            )
        client_request = runtime.ClientRequest(
            request_id=envelope_pb.client_request.request_id,
            client_name=envelope_pb.client_request.client_name,
            method=envelope_pb.client_request.method,
            size=envelope_pb.client_request.size,
            object_id=envelope_pb.client_request.object_id,
            stream_id=envelope_pb.client_request.stream_id,
            timestamp_ms=envelope_pb.client_request.timestamp_ms,
            iteration_count=envelope_pb.client_request.iteration_count,
            request_payload=request_payload,
        )
    if envelope_pb.HasField("client_response"):
        response_payload = None
        if envelope_pb.client_response.HasField("gemv"):
            response_payload = _from_pb_gemv_response_payload(
                envelope_pb.client_response.gemv
            )
        elif envelope_pb.client_response.HasField("conv2d"):
            response_payload = _from_pb_conv2d_response_payload(
                envelope_pb.client_response.conv2d
            )
        elif envelope_pb.client_response.HasField("gemm"):
            response_payload = _from_pb_gemm_response_payload(
                envelope_pb.client_response.gemm
            )
        client_response = runtime.ClientResponse(
            request_id=envelope_pb.client_response.request_id,
            method=envelope_pb.client_response.method,
            size=envelope_pb.client_response.size,
            object_id=envelope_pb.client_response.object_id,
            stream_id=envelope_pb.client_response.stream_id,
            timestamp_ms=envelope_pb.client_response.timestamp_ms,
            status_code=envelope_pb.client_response.status_code,
            error_message=envelope_pb.client_response.error_message,
            worker_count=envelope_pb.client_response.worker_count,
            client_count=envelope_pb.client_response.client_count,
            client_id=envelope_pb.client_response.client_id,
            iteration_count=envelope_pb.client_response.iteration_count,
            task_id=envelope_pb.client_response.task_id,
            elapsed_ms=envelope_pb.client_response.elapsed_ms,
            response_payload=response_payload,
            result_artifact=(
                _from_pb_artifact_descriptor(envelope_pb.client_response.result_artifact)
                if envelope_pb.client_response.HasField("result_artifact")
                else None
            ),
            timing=(
                _from_pb_response_timing(envelope_pb.client_response.timing)
                if envelope_pb.client_response.HasField("timing")
                else None
            ),
        )
    if envelope_pb.HasField("task_assign"):
        task_payload = None
        if envelope_pb.task_assign.HasField("gemv"):
            task_payload = _from_pb_gemv_task_payload(
                envelope_pb.task_assign.gemv
            )
        elif envelope_pb.task_assign.HasField("conv2d"):
            task_payload = _from_pb_conv2d_task_payload(
                envelope_pb.task_assign.conv2d
            )
        elif envelope_pb.task_assign.HasField("gemm"):
            task_payload = _from_pb_gemm_task_payload(
                envelope_pb.task_assign.gemm
            )
        task_assign = runtime.TaskAssign(
            request_id=envelope_pb.task_assign.request_id,
            node_id=envelope_pb.task_assign.node_id,
            task_id=envelope_pb.task_assign.task_id,
            method=envelope_pb.task_assign.method,
            size=envelope_pb.task_assign.size,
            object_id=envelope_pb.task_assign.object_id,
            stream_id=envelope_pb.task_assign.stream_id,
            timestamp_ms=envelope_pb.task_assign.timestamp_ms,
            iteration_count=envelope_pb.task_assign.iteration_count,
            transfer_mode=runtime.TransferMode(envelope_pb.task_assign.transfer_mode),
            artifact_id=envelope_pb.task_assign.artifact_id,
            artifact_timeout_ms=envelope_pb.task_assign.artifact_timeout_ms,
            task_payload=task_payload,
        )
    if envelope_pb.HasField("task_accept"):
        task_accept = runtime.TaskAccept(
            request_id=envelope_pb.task_accept.request_id,
            node_id=envelope_pb.task_accept.node_id,
            task_id=envelope_pb.task_accept.task_id,
            timestamp_ms=envelope_pb.task_accept.timestamp_ms,
            status_code=envelope_pb.task_accept.status_code,
        )
    if envelope_pb.HasField("task_fail"):
        task_fail = runtime.TaskFail(
            request_id=envelope_pb.task_fail.request_id,
            node_id=envelope_pb.task_fail.node_id,
            task_id=envelope_pb.task_fail.task_id,
            timestamp_ms=envelope_pb.task_fail.timestamp_ms,
            status_code=envelope_pb.task_fail.status_code,
            error_message=envelope_pb.task_fail.error_message,
        )
    if envelope_pb.HasField("task_result"):
        result_payload = None
        if envelope_pb.task_result.HasField("gemv"):
            result_payload = _from_pb_gemv_result_payload(
                envelope_pb.task_result.gemv
            )
        elif envelope_pb.task_result.HasField("conv2d"):
            result_payload = _from_pb_conv2d_result_payload(
                envelope_pb.task_result.conv2d
            )
        elif envelope_pb.task_result.HasField("gemm"):
            result_payload = _from_pb_gemm_result_payload(
                envelope_pb.task_result.gemm
            )
        task_result = runtime.TaskResult(
            request_id=envelope_pb.task_result.request_id,
            node_id=envelope_pb.task_result.node_id,
            task_id=envelope_pb.task_result.task_id,
            timestamp_ms=envelope_pb.task_result.timestamp_ms,
            status_code=envelope_pb.task_result.status_code,
            iteration_count=envelope_pb.task_result.iteration_count,
            result_payload=result_payload,
            result_artifact=(
                _from_pb_artifact_descriptor(envelope_pb.task_result.result_artifact)
                if envelope_pb.task_result.HasField("result_artifact")
                else None
            ),
            computation_ms=envelope_pb.task_result.computation_ms,
            peripheral_ms=envelope_pb.task_result.peripheral_ms,
        )
    if envelope_pb.HasField("artifact_release"):
        artifact_release = runtime.ArtifactRelease(
            node_id=envelope_pb.artifact_release.node_id,
            task_id=envelope_pb.artifact_release.task_id,
            artifact_id=envelope_pb.artifact_release.artifact_id,
            timestamp_ms=envelope_pb.artifact_release.timestamp_ms,
        )
    if envelope_pb.HasField("worker_update"):
        worker_update = runtime.WorkerUpdate(
            node_id=envelope_pb.worker_update.node_id,
            timestamp_ms=envelope_pb.worker_update.timestamp_ms,
            performance=_from_pb_compute_performance_summary(envelope_pb.worker_update.performance),
        )

    return runtime.RuntimeEnvelope(
        kind=kind,
        register_worker=register_worker,
        register_ok=register_ok,
        heartbeat=heartbeat,
        heartbeat_ok=heartbeat_ok,
        client_join=client_join,
        client_info_request=client_info_request,
        client_info_reply=client_info_reply,
        client_request_ok=client_request_ok,
        client_request=client_request,
        client_response=client_response,
        task_assign=task_assign,
        task_accept=task_accept,
        task_fail=task_fail,
        task_result=task_result,
        artifact_release=artifact_release,
        worker_update=worker_update,
    )
