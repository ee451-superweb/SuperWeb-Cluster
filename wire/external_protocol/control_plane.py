"""External control-plane models exchanged between clients and the main node.

Use this module when runtime code needs typed payloads for client join, info
polls, requests, and responses on the public-facing control plane.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass

from core.constants import METHOD_GEMM, METHOD_GEMV, METHOD_CONV2D
from wire.external_protocol.data_plane import ArtifactDescriptor
from wire.internal_protocol.model_utils import initvar_or_default


@dataclass(slots=True)
class ClientJoin:
    """Client registration payload."""

    client_name: str


@dataclass(slots=True)
class ClientInfoRequest:
    """Periodic liveness/info poll sent from one client to the main node."""

    client_id: str
    client_name: str
    timestamp_ms: int


@dataclass(slots=True)
class ClientInfoReply:
    """Main-node response describing whether one client still owns active work."""

    client_id: str
    request_timestamp_ms: int
    reply_timestamp_ms: int
    timeout_ms: int
    has_active_tasks: bool
    active_task_ids: tuple[str, ...] = ()


@dataclass(slots=True)
class GemvRequestPayload:
    """Method-specific request payload for fixed-matrix vector multiplication."""

    vector_length: int = 0
    vector_data: bytes = b""


@dataclass(slots=True)
class Conv2dRequestPayload:
    """Method-specific request payload for conv2d."""

    tensor_h: int = 0
    tensor_w: int = 0
    channels_in: int = 0
    channels_out: int = 0
    kernel_size: int = 0
    padding: int = 0
    stride: int = 1
    # Conv2dClientResponseMode: 0 = full output artifact, 1 = stats-only (see proto).
    client_response_mode: int = 0
    # Cap on stats_samples returned when client_response_mode == STATS_ONLY. 0 means cluster default.
    stats_max_samples: int = 0
    # Byte count of the weight payload that the client will push over the data plane.
    # Zero means the request carries no client-supplied weights and main should use
    # its own weight dataset.
    upload_size_bytes: int = 0
    # SHA-256 of the declared upload, validated when the bytes arrive on the DELIVER
    # channel. Empty means no pre-validation.
    upload_checksum: str = ""


@dataclass(slots=True)
class GemmRequestPayload:
    """Method-specific request payload for cuBLAS GEMM.

    The client supplies no matrix bytes; both A and B are pre-generated on
    each compute node from fixed seeds keyed by request size. Shape fields
    on this payload are intentionally empty — shape is derived from the
    request ``size`` by main and worker alike.
    """


@dataclass(slots=True)
class GemmResponsePayload:
    """Method-specific client response payload for cuBLAS GEMM."""

    output_length: int = 0
    output_vector: bytes = b""


@dataclass(slots=True)
class GemvResponsePayload:
    """Method-specific client response payload for fixed-matrix vector multiplication."""

    output_length: int = 0
    output_vector: bytes = b""


@dataclass(slots=True)
class Conv2dResponsePayload:
    """Method-specific client response payload for conv2d."""

    output_length: int = 0
    output_vector: bytes = b""
    result_artifact_id: str = ""
    stats_element_count: int = 0
    stats_sum: float = 0.0
    stats_sum_squares: float = 0.0
    stats_samples: tuple[float, ...] = ()


@dataclass(slots=True)
class ClientRequest:
    """Client request sent to the main node with a typed method payload."""

    request_id: str
    client_name: str
    method: str
    size: str
    object_id: str
    stream_id: str
    timestamp_ms: int
    iteration_count: int
    request_payload: GemvRequestPayload | Conv2dRequestPayload | GemmRequestPayload | None = None
    vector_length: InitVar[int] = 0
    vector_data: InitVar[bytes] = b""
    tensor_h: InitVar[int] = 0
    tensor_w: InitVar[int] = 0
    channels_in: InitVar[int] = 0
    channels_out: InitVar[int] = 0
    kernel_size: InitVar[int] = 0
    padding: InitVar[int] = 0
    stride: InitVar[int] = 1
    conv2d_client_response_mode: InitVar[int] = 0
    conv2d_stats_max_samples: InitVar[int] = 0

    def __post_init__(
        self,
        vector_length: int,
        vector_data: bytes,
        tensor_h: int,
        tensor_w: int,
        channels_in: int,
        channels_out: int,
        kernel_size: int,
        padding: int,
        stride: int,
        conv2d_client_response_mode: int,
        conv2d_stats_max_samples: int,
    ) -> None:
        """Normalize legacy initvars into the typed client request payload."""
        vector_length = initvar_or_default(vector_length, 0)
        vector_data = initvar_or_default(vector_data, b"")
        tensor_h = initvar_or_default(tensor_h, 0)
        tensor_w = initvar_or_default(tensor_w, 0)
        channels_in = initvar_or_default(channels_in, 0)
        channels_out = initvar_or_default(channels_out, 0)
        kernel_size = initvar_or_default(kernel_size, 0)
        padding = initvar_or_default(padding, 0)
        stride = initvar_or_default(stride, 1)
        conv2d_client_response_mode = initvar_or_default(conv2d_client_response_mode, 0)
        conv2d_stats_max_samples = initvar_or_default(conv2d_stats_max_samples, 0)
        if self.request_payload is None:
            if self.method == METHOD_GEMM:
                self.request_payload = GemmRequestPayload()
            elif self.method == METHOD_CONV2D or any(
                value for value in (tensor_h, tensor_w, channels_in, channels_out, kernel_size, padding)
            ):
                self.request_payload = Conv2dRequestPayload(
                    tensor_h=tensor_h,
                    tensor_w=tensor_w,
                    channels_in=channels_in,
                    channels_out=channels_out,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    client_response_mode=conv2d_client_response_mode,
                    stats_max_samples=conv2d_stats_max_samples,
                )
            else:
                self.request_payload = GemvRequestPayload(
                    vector_length=vector_length,
                    vector_data=vector_data,
                )
        elif self.method == METHOD_GEMV and not isinstance(
            self.request_payload,
            GemvRequestPayload,
        ):
            raise ValueError("gemv requests require a matching payload")
        elif self.method == METHOD_CONV2D and not isinstance(
            self.request_payload,
            Conv2dRequestPayload,
        ):
            raise ValueError("conv2d requests require a matching payload")
        elif self.method == METHOD_GEMM and not isinstance(
            self.request_payload,
            GemmRequestPayload,
        ):
            raise ValueError("gemm requests require a matching payload")

    @property
    def gemv_payload(self) -> GemvRequestPayload | None:
        """Return the GEMV request payload when this request targets GEMV."""
        if isinstance(self.request_payload, GemvRequestPayload):
            return self.request_payload
        return None

    @property
    def conv2d_payload(self) -> Conv2dRequestPayload | None:
        """Return the conv2d request payload when this request targets Conv2D."""
        if isinstance(self.request_payload, Conv2dRequestPayload):
            return self.request_payload
        return None

    @property
    def gemm_payload(self) -> GemmRequestPayload | None:
        """Return the GEMM request payload when this request targets GEMM."""
        if isinstance(self.request_payload, GemmRequestPayload):
            return self.request_payload
        return None

    @property
    def vector_length(self) -> int:
        """Return the GEMV vector length encoded in the request payload."""
        payload = self.gemv_payload
        return payload.vector_length if payload is not None else 0

    @property
    def vector_data(self) -> bytes:
        """Return the GEMV vector bytes encoded in the request payload."""
        payload = self.gemv_payload
        return payload.vector_data if payload is not None else b""

    @property
    def tensor_h(self) -> int:
        """Return the spatial input height encoded in the request payload."""
        payload = self.conv2d_payload
        return payload.tensor_h if payload is not None else 0

    @property
    def tensor_w(self) -> int:
        """Return the spatial input width encoded in the request payload."""
        payload = self.conv2d_payload
        return payload.tensor_w if payload is not None else 0

    @property
    def channels_in(self) -> int:
        """Return the spatial input-channel count encoded in the request payload."""
        payload = self.conv2d_payload
        return payload.channels_in if payload is not None else 0

    @property
    def channels_out(self) -> int:
        """Return the spatial output-channel count encoded in the request payload."""
        payload = self.conv2d_payload
        return payload.channels_out if payload is not None else 0

    @property
    def kernel_size(self) -> int:
        """Return the spatial kernel size encoded in the request payload."""
        payload = self.conv2d_payload
        return payload.kernel_size if payload is not None else 0

    @property
    def padding(self) -> int:
        """Return the spatial padding encoded in the request payload."""
        payload = self.conv2d_payload
        return payload.padding if payload is not None else 0

    @property
    def stride(self) -> int:
        """Return the spatial stride encoded in the request payload."""
        payload = self.conv2d_payload
        return payload.stride if payload is not None else 1


@dataclass(slots=True)
class ClientRequestOk:
    """Early main-node acknowledgement carrying the server-assigned task id."""

    client_id: str
    task_id: str
    method: str
    size: str
    object_id: str
    accepted_timestamp_ms: int
    # Data-plane handshake fields. Non-empty upload_id means the client must open a
    # data-plane socket and push its bytes via DELIVER before main can start compute.
    # Non-empty download_id means the client should issue REQUEST(download_id) on the
    # same socket (or a fresh one) to pull the result artifact.
    upload_id: str = ""
    download_id: str = ""
    data_endpoint_host: str = ""
    data_endpoint_port: int = 0


@dataclass(slots=True)
class WorkerTiming:
    """Per-worker timing breakdown observed by the main node."""

    node_id: str
    task_id: str
    slice: str
    wall_ms: int
    artifact_fetch_ms: int = 0
    computation_ms: int = 0
    peripheral_ms: int = 0


@dataclass(slots=True)
class ResponseTiming:
    """Stage-by-stage timing breakdown for one client response."""

    dispatch_ms: int = 0
    task_window_ms: int = 0
    aggregate_ms: int = 0
    workers: tuple[WorkerTiming, ...] = ()


@dataclass(slots=True)
class ClientResponse:
    """Main-node response sent back to a client with a typed method payload."""

    request_id: str
    method: str
    size: str
    object_id: str
    stream_id: str
    timestamp_ms: int
    status_code: int
    error_message: str
    worker_count: int
    client_count: int
    client_id: str
    iteration_count: int
    task_id: str = ""
    elapsed_ms: int = 0
    response_payload: GemvResponsePayload | Conv2dResponsePayload | GemmResponsePayload | None = None
    result_artifact: ArtifactDescriptor | None = None
    timing: ResponseTiming | None = None
    output_length: InitVar[int] = 0
    output_vector: InitVar[bytes] = b""
    result_artifact_id: InitVar[str] = ""

    def __post_init__(self, output_length: int, output_vector: bytes, result_artifact_id: str) -> None:
        """Normalize legacy initvars into the typed client response payload."""
        output_length = initvar_or_default(output_length, 0)
        output_vector = initvar_or_default(output_vector, b"")
        result_artifact_id = initvar_or_default(result_artifact_id, "")
        if self.response_payload is None:
            if self.method == METHOD_GEMM:
                self.response_payload = GemmResponsePayload(
                    output_length=output_length,
                    output_vector=output_vector,
                )
            elif self.method == METHOD_CONV2D or result_artifact_id:
                self.response_payload = Conv2dResponsePayload(
                    output_length=output_length,
                    output_vector=output_vector,
                    result_artifact_id=result_artifact_id,
                )
            else:
                self.response_payload = GemvResponsePayload(
                    output_length=output_length,
                    output_vector=output_vector,
                )
        elif self.method == METHOD_GEMV and not isinstance(
            self.response_payload,
            GemvResponsePayload,
        ):
            raise ValueError("gemv responses require a matching payload")
        elif self.method == METHOD_CONV2D and not isinstance(
            self.response_payload,
            Conv2dResponsePayload,
        ):
            raise ValueError("conv2d responses require a matching payload")
        elif self.method == METHOD_GEMM and not isinstance(
            self.response_payload,
            GemmResponsePayload,
        ):
            raise ValueError("gemm responses require a matching payload")

    @property
    def gemv_payload(self) -> GemvResponsePayload | None:
        """Return the GEMV response payload when this response targets GEMV."""
        if isinstance(self.response_payload, GemvResponsePayload):
            return self.response_payload
        return None

    @property
    def conv2d_payload(self) -> Conv2dResponsePayload | None:
        """Return the conv2d response payload when this response targets Conv2D."""
        if isinstance(self.response_payload, Conv2dResponsePayload):
            return self.response_payload
        return None

    @property
    def gemm_payload(self) -> GemmResponsePayload | None:
        """Return the GEMM response payload when this response targets GEMM."""
        if isinstance(self.response_payload, GemmResponsePayload):
            return self.response_payload
        return None

    @property
    def output_length(self) -> int:
        """Return the output element count stored in the response payload."""
        if self.gemv_payload is not None:
            return self.gemv_payload.output_length
        if self.conv2d_payload is not None:
            return self.conv2d_payload.output_length
        if self.gemm_payload is not None:
            return self.gemm_payload.output_length
        return 0

    @property
    def output_vector(self) -> bytes:
        """Return the inline output bytes stored in the response payload."""
        if self.gemv_payload is not None:
            return self.gemv_payload.output_vector
        if self.conv2d_payload is not None:
            return self.conv2d_payload.output_vector
        if self.gemm_payload is not None:
            return self.gemm_payload.output_vector
        return b""

    @property
    def result_artifact_id(self) -> str:
        """Return the artifact id associated with this client response, if any."""
        if self.result_artifact is not None:
            return self.result_artifact.artifact_id
        payload = self.conv2d_payload
        return payload.result_artifact_id if payload is not None else ""
