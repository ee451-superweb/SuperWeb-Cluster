"""Internal control-plane models exchanged among main and compute nodes.

Use this module when runtime code needs typed payloads for worker registration,
heartbeats, task dispatch, task results, artifact release, or worker updates.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass

from app.constants import METHOD_FIXED_MATRIX_VECTOR_MULTIPLICATION, METHOD_SPATIAL_CONVOLUTION
from common.types import (
    ComputePerformanceSummary,
    HardwareProfile,
)
from wire.external_protocol.data_plane import ArtifactDescriptor
from wire.internal_protocol._model_utils import initvar_or_default
from wire.internal_protocol.common import NodeStatus, TransferMode


@dataclass(slots=True)
class RegisterWorker:
    """Compute-node registration payload."""

    node_name: str
    hardware: HardwareProfile
    performance: ComputePerformanceSummary


@dataclass(slots=True)
class RegisterOk:
    """Main-node acceptance payload."""

    main_node_name: str
    main_node_ip: str
    main_node_port: int
    node_id: str


@dataclass(slots=True)
class Heartbeat:
    """One-way main-node heartbeat payload."""

    main_node_name: str
    unix_time_ms: int


@dataclass(slots=True)
class HeartbeatOk:
    """Worker acknowledgement for one heartbeat payload."""

    node_name: str
    heartbeat_unix_time_ms: int
    received_unix_time_ms: int
    active_task_ids: tuple[str, ...] = ()
    node_status: NodeStatus = NodeStatus.UNKNOWN
    completed_task_count: int = 0
    node_id: str = ""


@dataclass(slots=True)
class FixedMatrixVectorTaskPayload:
    """Method-specific task payload for fixed-matrix vector multiplication."""

    row_start: int = 0
    row_end: int = 0
    vector_length: int = 0
    vector_data: bytes = b""


@dataclass(slots=True)
class SpatialConvolutionTaskPayload:
    """Method-specific task payload for spatial convolution."""

    start_oc: int = 0
    end_oc: int = 0
    tensor_h: int = 0
    tensor_w: int = 0
    channels_in: int = 0
    channels_out: int = 0
    kernel_size: int = 0
    padding: int = 0
    stride: int = 1
    weight_data: bytes = b""


@dataclass(slots=True)
class FixedMatrixVectorResultPayload:
    """Method-specific task result payload for fixed-matrix vector multiplication."""

    row_start: int = 0
    row_end: int = 0
    output_length: int = 0
    output_vector: bytes = b""


@dataclass(slots=True)
class SpatialConvolutionResultPayload:
    """Method-specific task result payload for spatial convolution."""

    start_oc: int = 0
    end_oc: int = 0
    output_h: int = 0
    output_w: int = 0
    output_length: int = 0
    output_vector: bytes = b""
    result_artifact_id: str = ""


@dataclass(slots=True)
class ArtifactRelease:
    """Main-node acknowledgement that one worker artifact can be deleted."""

    node_id: str
    task_id: str
    artifact_id: str
    timestamp_ms: int


@dataclass(slots=True)
class WorkerUpdate:
    """Idle-period worker performance refresh sent from compute node to main node."""

    node_id: str
    timestamp_ms: int
    performance: ComputePerformanceSummary


@dataclass(slots=True)
class TaskAssign:
    """Main-node instruction telling one worker to compute one method-specific slice."""

    request_id: str
    node_id: str
    task_id: str
    method: str
    object_id: str
    stream_id: str
    timestamp_ms: int
    iteration_count: int
    transfer_mode: TransferMode = TransferMode.UNSPECIFIED
    artifact_id: str = ""
    artifact_timeout_ms: int = 0
    task_payload: FixedMatrixVectorTaskPayload | SpatialConvolutionTaskPayload | None = None
    row_start: InitVar[int] = 0
    row_end: InitVar[int] = 0
    vector_length: InitVar[int] = 0
    vector_data: InitVar[bytes] = b""
    start_oc: InitVar[int] = 0
    end_oc: InitVar[int] = 0
    tensor_h: InitVar[int] = 0
    tensor_w: InitVar[int] = 0
    channels_in: InitVar[int] = 0
    channels_out: InitVar[int] = 0
    kernel_size: InitVar[int] = 0
    padding: InitVar[int] = 0
    stride: InitVar[int] = 1
    weight_data: InitVar[bytes] = b""

    def __post_init__(
        self,
        row_start: int,
        row_end: int,
        vector_length: int,
        vector_data: bytes,
        start_oc: int,
        end_oc: int,
        tensor_h: int,
        tensor_w: int,
        channels_in: int,
        channels_out: int,
        kernel_size: int,
        padding: int,
        stride: int,
        weight_data: bytes,
    ) -> None:
        """Normalize legacy initvars into the typed task payload."""
        self.transfer_mode = TransferMode(self.transfer_mode)
        row_start = initvar_or_default(row_start, 0)
        row_end = initvar_or_default(row_end, 0)
        vector_length = initvar_or_default(vector_length, 0)
        vector_data = initvar_or_default(vector_data, b"")
        start_oc = initvar_or_default(start_oc, 0)
        end_oc = initvar_or_default(end_oc, 0)
        tensor_h = initvar_or_default(tensor_h, 0)
        tensor_w = initvar_or_default(tensor_w, 0)
        channels_in = initvar_or_default(channels_in, 0)
        channels_out = initvar_or_default(channels_out, 0)
        kernel_size = initvar_or_default(kernel_size, 0)
        padding = initvar_or_default(padding, 0)
        stride = initvar_or_default(stride, 1)
        weight_data = initvar_or_default(weight_data, b"")
        if self.task_payload is None:
            if self.method == METHOD_SPATIAL_CONVOLUTION or any(
                value
                for value in (
                    start_oc,
                    end_oc,
                    tensor_h,
                    tensor_w,
                    channels_in,
                    channels_out,
                    kernel_size,
                    padding,
                    len(weight_data),
                )
            ):
                self.task_payload = SpatialConvolutionTaskPayload(
                    start_oc=start_oc,
                    end_oc=end_oc,
                    tensor_h=tensor_h,
                    tensor_w=tensor_w,
                    channels_in=channels_in,
                    channels_out=channels_out,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    weight_data=weight_data,
                )
            else:
                self.task_payload = FixedMatrixVectorTaskPayload(
                    row_start=row_start,
                    row_end=row_end,
                    vector_length=vector_length,
                    vector_data=vector_data,
                )
        elif self.method == METHOD_FIXED_MATRIX_VECTOR_MULTIPLICATION and not isinstance(
            self.task_payload,
            FixedMatrixVectorTaskPayload,
        ):
            raise ValueError("fixed_matrix_vector_multiplication tasks require a matching payload")
        elif self.method == METHOD_SPATIAL_CONVOLUTION and not isinstance(
            self.task_payload,
            SpatialConvolutionTaskPayload,
        ):
            raise ValueError("spatial_convolution tasks require a matching payload")

    @property
    def fixed_matrix_vector_multiplication_payload(self) -> FixedMatrixVectorTaskPayload | None:
        """Return the FMVM task payload when this assignment targets FMVM."""
        if isinstance(self.task_payload, FixedMatrixVectorTaskPayload):
            return self.task_payload
        return None

    @property
    def spatial_convolution_payload(self) -> SpatialConvolutionTaskPayload | None:
        """Return the spatial task payload when this assignment targets Conv2D."""
        if isinstance(self.task_payload, SpatialConvolutionTaskPayload):
            return self.task_payload
        return None

    @property
    def row_start(self) -> int:
        """Return the FMVM starting row encoded in the task payload."""
        payload = self.fixed_matrix_vector_multiplication_payload
        return payload.row_start if payload is not None else 0

    @property
    def row_end(self) -> int:
        """Return the FMVM ending row encoded in the task payload."""
        payload = self.fixed_matrix_vector_multiplication_payload
        return payload.row_end if payload is not None else 0

    @property
    def vector_length(self) -> int:
        """Return the FMVM vector length encoded in the task payload."""
        payload = self.fixed_matrix_vector_multiplication_payload
        return payload.vector_length if payload is not None else 0

    @property
    def vector_data(self) -> bytes:
        """Return the FMVM vector bytes encoded in the task payload."""
        payload = self.fixed_matrix_vector_multiplication_payload
        return payload.vector_data if payload is not None else b""

    @property
    def start_oc(self) -> int:
        """Return the spatial starting output-channel index in the task payload."""
        payload = self.spatial_convolution_payload
        return payload.start_oc if payload is not None else 0

    @property
    def end_oc(self) -> int:
        """Return the spatial ending output-channel index in the task payload."""
        payload = self.spatial_convolution_payload
        return payload.end_oc if payload is not None else 0

    @property
    def tensor_h(self) -> int:
        """Return the spatial input height encoded in the task payload."""
        payload = self.spatial_convolution_payload
        return payload.tensor_h if payload is not None else 0

    @property
    def tensor_w(self) -> int:
        """Return the spatial input width encoded in the task payload."""
        payload = self.spatial_convolution_payload
        return payload.tensor_w if payload is not None else 0

    @property
    def channels_in(self) -> int:
        """Return the spatial input-channel count encoded in the task payload."""
        payload = self.spatial_convolution_payload
        return payload.channels_in if payload is not None else 0

    @property
    def channels_out(self) -> int:
        """Return the spatial total output-channel count encoded in the task payload."""
        payload = self.spatial_convolution_payload
        return payload.channels_out if payload is not None else 0

    @property
    def kernel_size(self) -> int:
        """Return the spatial kernel size encoded in the task payload."""
        payload = self.spatial_convolution_payload
        return payload.kernel_size if payload is not None else 0

    @property
    def padding(self) -> int:
        """Return the spatial padding encoded in the task payload."""
        payload = self.spatial_convolution_payload
        return payload.padding if payload is not None else 0

    @property
    def stride(self) -> int:
        """Return the spatial stride encoded in the task payload."""
        payload = self.spatial_convolution_payload
        return payload.stride if payload is not None else 1

    @property
    def weight_data(self) -> bytes:
        """Return the spatial weight bytes encoded in the task payload."""
        payload = self.spatial_convolution_payload
        return payload.weight_data if payload is not None else b""


@dataclass(slots=True)
class TaskAccept:
    """Worker acknowledgement that one assigned task was accepted."""

    request_id: str
    node_id: str
    task_id: str
    timestamp_ms: int
    status_code: int


@dataclass(slots=True)
class TaskFail:
    """Worker error result for one assigned task."""

    request_id: str
    node_id: str
    task_id: str
    timestamp_ms: int
    status_code: int
    error_message: str


@dataclass(slots=True)
class TaskResult:
    """Worker-computed method-specific task result."""

    request_id: str
    node_id: str
    task_id: str
    timestamp_ms: int
    status_code: int
    iteration_count: int
    result_payload: FixedMatrixVectorResultPayload | SpatialConvolutionResultPayload | None = None
    result_artifact: ArtifactDescriptor | None = None
    local_result_path: str = ""
    row_start: InitVar[int] = 0
    row_end: InitVar[int] = 0
    output_length: InitVar[int] = 0
    output_vector: InitVar[bytes] = b""
    start_oc: InitVar[int] = 0
    end_oc: InitVar[int] = 0
    output_h: InitVar[int] = 0
    output_w: InitVar[int] = 0
    result_artifact_id: InitVar[str] = ""

    def __post_init__(
        self,
        row_start: int,
        row_end: int,
        output_length: int,
        output_vector: bytes,
        start_oc: int,
        end_oc: int,
        output_h: int,
        output_w: int,
        result_artifact_id: str,
    ) -> None:
        """Normalize legacy initvars into the typed task-result payload."""
        row_start = initvar_or_default(row_start, 0)
        row_end = initvar_or_default(row_end, 0)
        output_length = initvar_or_default(output_length, 0)
        output_vector = initvar_or_default(output_vector, b"")
        start_oc = initvar_or_default(start_oc, 0)
        end_oc = initvar_or_default(end_oc, 0)
        output_h = initvar_or_default(output_h, 0)
        output_w = initvar_or_default(output_w, 0)
        result_artifact_id = initvar_or_default(result_artifact_id, "")
        if self.result_payload is None:
            if any(value for value in (start_oc, end_oc, output_h, output_w)) or result_artifact_id:
                self.result_payload = SpatialConvolutionResultPayload(
                    start_oc=start_oc,
                    end_oc=end_oc,
                    output_h=output_h,
                    output_w=output_w,
                    output_length=output_length,
                    output_vector=output_vector,
                    result_artifact_id=result_artifact_id,
                )
            else:
                self.result_payload = FixedMatrixVectorResultPayload(
                    row_start=row_start,
                    row_end=row_end,
                    output_length=output_length,
                    output_vector=output_vector,
                )

    @property
    def fixed_matrix_vector_multiplication_payload(self) -> FixedMatrixVectorResultPayload | None:
        """Return the FMVM result payload when this task result targets FMVM."""
        if isinstance(self.result_payload, FixedMatrixVectorResultPayload):
            return self.result_payload
        return None

    @property
    def spatial_convolution_payload(self) -> SpatialConvolutionResultPayload | None:
        """Return the spatial result payload when this task result targets Conv2D."""
        if isinstance(self.result_payload, SpatialConvolutionResultPayload):
            return self.result_payload
        return None

    @property
    def row_start(self) -> int:
        """Return the FMVM starting row encoded in the result payload."""
        payload = self.fixed_matrix_vector_multiplication_payload
        return payload.row_start if payload is not None else 0

    @property
    def row_end(self) -> int:
        """Return the FMVM ending row encoded in the result payload."""
        payload = self.fixed_matrix_vector_multiplication_payload
        return payload.row_end if payload is not None else 0

    @property
    def output_length(self) -> int:
        """Return the output element count stored in the result payload."""
        if self.fixed_matrix_vector_multiplication_payload is not None:
            return self.fixed_matrix_vector_multiplication_payload.output_length
        if self.spatial_convolution_payload is not None:
            return self.spatial_convolution_payload.output_length
        return 0

    @property
    def output_vector(self) -> bytes:
        """Return the inline output bytes stored in the result payload."""
        if self.fixed_matrix_vector_multiplication_payload is not None:
            return self.fixed_matrix_vector_multiplication_payload.output_vector
        if self.spatial_convolution_payload is not None:
            return self.spatial_convolution_payload.output_vector
        return b""

    @property
    def start_oc(self) -> int:
        """Return the spatial starting output-channel index in the result payload."""
        payload = self.spatial_convolution_payload
        return payload.start_oc if payload is not None else 0

    @property
    def end_oc(self) -> int:
        """Return the spatial ending output-channel index in the result payload."""
        payload = self.spatial_convolution_payload
        return payload.end_oc if payload is not None else 0

    @property
    def output_h(self) -> int:
        """Return the spatial output height encoded in the result payload."""
        payload = self.spatial_convolution_payload
        return payload.output_h if payload is not None else 0

    @property
    def output_w(self) -> int:
        """Return the spatial output width encoded in the result payload."""
        payload = self.spatial_convolution_payload
        return payload.output_w if payload is not None else 0

    @property
    def result_artifact_id(self) -> str:
        """Return the artifact id associated with this task result, if any."""
        if self.result_artifact is not None:
            return self.result_artifact.artifact_id
        payload = self.spatial_convolution_payload
        return payload.result_artifact_id if payload is not None else ""
