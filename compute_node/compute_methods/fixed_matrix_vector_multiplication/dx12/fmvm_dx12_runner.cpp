#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <d3d12.h>
#include <d3dcompiler.h>
#include <dxgi1_6.h>
#include <windows.h>
#include <wrl/client.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace {

constexpr const char* kShaderSource = R"(
cbuffer Params : register(b0)
{
    uint cols;
    uint matrix_row_offset;
    uint output_base_row;
    uint output_rows;
    uint rows_per_thread;
};

StructuredBuffer<float> Matrix : register(t0);
StructuredBuffer<float> Vector : register(t1);
RWStructuredBuffer<float> Output : register(u0);
groupshared float PartialSums[THREAD_GROUP_SIZE];

[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID)
{
    const uint lane = group_thread_id.x;
    const uint base_output_row = group_id.x * rows_per_thread;

    [loop]
    for (uint local_row = 0; local_row < rows_per_thread; ++local_row)
    {
        const uint output_row = base_output_row + local_row;
        if (output_row >= output_rows)
        {
            break;
        }

        const uint matrix_row = matrix_row_offset + output_row;
        const uint row_base = matrix_row * cols;

        float sum = 0.0f;
        [loop]
        for (uint col = lane; col < cols; col += THREAD_GROUP_SIZE)
        {
            sum += Matrix[row_base + col] * Vector[col];
        }

        PartialSums[lane] = sum;
        GroupMemoryBarrierWithGroupSync();

        for (uint stride = THREAD_GROUP_SIZE / 2; stride > 0; stride >>= 1)
        {
            if (lane < stride)
            {
                PartialSums[lane] += PartialSums[lane + stride];
            }
            GroupMemoryBarrierWithGroupSync();
        }

        if (lane == 0)
        {
            Output[output_base_row + output_row] = PartialSums[0];
        }
        GroupMemoryBarrierWithGroupSync();
    }
}
)";

struct Options {
    std::string matrix_path;
    std::string vector_path;
    std::string output_path;
    int rows = 0;
    int cols = 0;
    int row_start = 0;
    int row_end = 0;
    std::vector<int> thread_group_sizes;
    std::vector<int> rows_per_thread_values;
    int fixed_thread_group_size = 0;
    int fixed_rows_per_thread = 0;
    int autotune_repeats = 1;
    int measurement_repeats = 1;
    std::string accumulation_precision = "fp32";
    bool task_mode = false;
    bool server_mode = false;
};

struct PhaseMetrics {
    int repeats = 0;
    double wall_clock_latency_seconds = std::numeric_limits<double>::infinity();
    double effective_gflops = 0.0;
    std::string checksum;
    int dispatches_per_repeat = 0;
};

struct TrialMetrics {
    int thread_group_size = 0;
    int rows_per_thread = 0;
    PhaseMetrics autotune;
    PhaseMetrics measurement;
};

struct alignas(256) ShaderParams {
    std::uint32_t cols = 0;
    std::uint32_t matrix_row_offset = 0;
    std::uint32_t output_base_row = 0;
    std::uint32_t output_rows = 0;
    std::uint32_t rows_per_thread = 0;
    std::uint32_t padding[59] = {};
};

bool is_supported_accumulation_precision(const std::string& value);
void throw_if_failed(HRESULT result, const char* message);
std::vector<int> parse_int_list(const std::string& text);
std::vector<std::string> split_tab_fields(const std::string& line);
Options parse_args(int argc, char** argv);
std::string wide_to_utf8(const wchar_t* text);
void read_binary_file_into_pointer(const std::string& path, void* destination, std::size_t expected_bytes);
void write_float32_file(const std::string& path, const std::vector<float>& values);
std::string fnv1a64_checksum(const std::vector<float>& values);
bool contains_nvidia_vendor_name(const std::string& text);
UINT64 align_to_256(UINT64 value);
D3D12_RESOURCE_DESC buffer_desc(UINT64 size, D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE);
D3D12_HEAP_PROPERTIES heap_properties(D3D12_HEAP_TYPE type);
D3D12_RESOURCE_BARRIER transition_barrier(
    ID3D12Resource* resource,
    D3D12_RESOURCE_STATES before_state,
    D3D12_RESOURCE_STATES after_state
);
ComPtr<IDXGIAdapter1> choose_adapter(
    ComPtr<IDXGIFactory6>& factory,
    std::string& device_name,
    std::string& adapter_kind
);

class Dx12Runner {
public:
    Dx12Runner(const Options& options, const std::vector<int>& thread_group_sizes);
    ~Dx12Runner();

    PhaseMetrics run_configuration(int thread_group_size, int rows_per_thread, int repeats);
    std::vector<float> run_task_output(int thread_group_size, int rows_per_thread, int repeats, PhaseMetrics& metrics);
    std::vector<float> run_task_output_from_vector(
        const std::string& vector_path,
        int row_start,
        int row_end,
        int thread_group_size,
        int rows_per_thread,
        int repeats,
        PhaseMetrics& metrics
    );

    const std::string& device_name() const { return device_name_; }
    const std::string& adapter_kind() const { return adapter_kind_; }
    double static_upload_wall_clock_latency_seconds() const { return static_upload_wall_clock_latency_seconds_; }
    double last_vector_upload_wall_clock_latency_seconds() const { return last_vector_upload_wall_clock_latency_seconds_; }

private:
    void create_root_signature();
    void create_pipelines(const std::vector<int>& thread_group_sizes);
    void create_resources();
    void upload_static_matrix();
    void upload_vector_from_file(const std::string& vector_path);
    std::vector<float> read_output_values(UINT row_count, UINT64 output_bytes);
    void wait_for_gpu();
    ComPtr<ID3D12PipelineState> pipeline_for_thread_group_size(int thread_group_size) const;

    Options options_;
    UINT total_rows_ = 0;
    UINT64 output_capacity_bytes_ = 0;
    ComPtr<IDXGIFactory6> factory_;
    ComPtr<IDXGIAdapter1> adapter_;
    ComPtr<ID3D12Device> device_;
    ComPtr<ID3D12CommandQueue> queue_;
    ComPtr<ID3D12CommandAllocator> allocator_;
    ComPtr<ID3D12GraphicsCommandList> command_list_;
    ComPtr<ID3D12Fence> fence_;
    HANDLE fence_event_ = nullptr;
    UINT64 next_fence_value_ = 1;
    ComPtr<ID3D12RootSignature> root_signature_;
    std::vector<std::pair<int, ComPtr<ID3D12PipelineState>>> pipelines_;
    ComPtr<ID3D12Resource> matrix_buffer_;
    ComPtr<ID3D12Resource> vector_buffer_;
    ComPtr<ID3D12Resource> vector_upload_buffer_;
    ComPtr<ID3D12Resource> output_buffer_;
    ComPtr<ID3D12Resource> readback_buffer_;
    std::string device_name_;
    std::string adapter_kind_;
    double static_upload_wall_clock_latency_seconds_ = 0.0;
    double last_vector_upload_wall_clock_latency_seconds_ = 0.0;
    bool vector_uploaded_ = false;
    void* vector_upload_mapped_ = nullptr;
};

bool is_supported_accumulation_precision(const std::string& value) {
    return value == "fp32";
}

void throw_if_failed(HRESULT result, const char* message) {
    if (FAILED(result)) {
        std::ostringstream builder;
        builder << message << " (HRESULT=0x" << std::hex << static_cast<unsigned long>(result) << ")";
        throw std::runtime_error(builder.str());
    }
}

std::vector<int> parse_int_list(const std::string& text) {
    std::vector<int> values;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (item.empty()) {
            continue;
        }
        values.push_back(std::stoi(item));
    }
    if (values.empty()) {
        throw std::runtime_error("expected a non-empty integer list");
    }
    return values;
}

std::vector<std::string> split_tab_fields(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream stream(line);
    std::string item;
    while (std::getline(stream, item, '\t')) {
        fields.push_back(item);
    }
    return fields;
}

Options parse_args(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc) {
            throw std::runtime_error("missing value for command line flag");
        }

        const std::string key = argv[index];
        const std::string value = argv[index + 1];
        if (key == "--matrix") {
            options.matrix_path = value;
        } else if (key == "--vector") {
            options.vector_path = value;
        } else if (key == "--output") {
            options.output_path = value;
        } else if (key == "--rows") {
            options.rows = std::stoi(value);
        } else if (key == "--cols") {
            options.cols = std::stoi(value);
        } else if (key == "--row-start") {
            options.row_start = std::stoi(value);
        } else if (key == "--row-end") {
            options.row_end = std::stoi(value);
        } else if (key == "--thread-group-sizes") {
            options.thread_group_sizes = parse_int_list(value);
        } else if (key == "--rows-per-thread") {
            options.rows_per_thread_values = parse_int_list(value);
        } else if (key == "--fixed-thread-group-size") {
            options.fixed_thread_group_size = std::stoi(value);
        } else if (key == "--fixed-rows-per-thread") {
            options.fixed_rows_per_thread = std::stoi(value);
        } else if (key == "--autotune-repeats") {
            options.autotune_repeats = std::stoi(value);
        } else if (key == "--measurement-repeats" || key == "--iteration-count") {
            // Task execution reuses the measurement loop but exposes the more
            // domain-specific name iteration-count to the runtime layer.
            options.measurement_repeats = std::stoi(value);
        } else if (key == "--accumulation-precision") {
            options.accumulation_precision = value;
        } else if (key == "--server") {
            options.server_mode = (value == "1" || value == "true" || value == "TRUE");
        } else {
            throw std::runtime_error("unknown flag: " + key);
        }
    }

    if (options.matrix_path.empty()) {
        throw std::runtime_error("matrix path is required");
    }
    if (options.rows <= 0 || options.cols <= 0) {
        throw std::runtime_error("rows and cols must be positive");
    }
    if (options.row_end == 0) {
        options.row_end = options.rows;
    }
    if (options.row_start < 0 || options.row_end <= options.row_start || options.row_end > options.rows) {
        throw std::runtime_error("row range is invalid");
    }
    if (options.measurement_repeats <= 0) {
        throw std::runtime_error("measurement repeats must be positive");
    }
    if (!is_supported_accumulation_precision(options.accumulation_precision)) {
        throw std::runtime_error("unsupported accumulation precision: " + options.accumulation_precision);
    }

    const bool has_fixed_thread_group_size = options.fixed_thread_group_size > 0;
    const bool has_fixed_rows_per_thread = options.fixed_rows_per_thread > 0;
    if (has_fixed_thread_group_size != has_fixed_rows_per_thread) {
        throw std::runtime_error("task mode requires fixed-thread-group-size and fixed-rows-per-thread");
    }
    options.task_mode = has_fixed_thread_group_size && has_fixed_rows_per_thread;
    if (options.server_mode && options.task_mode) {
        throw std::runtime_error("server mode cannot be combined with task mode");
    }

    if (options.server_mode) {
        if (options.thread_group_sizes.empty()) {
            options.thread_group_sizes = {256, 512};
        }
    } else if (!options.task_mode) {
        if (options.vector_path.empty()) {
            throw std::runtime_error("vector path is required");
        }
        if (options.thread_group_sizes.empty() || options.rows_per_thread_values.empty()) {
            throw std::runtime_error("thread-group-size and rows-per-thread candidate lists are required");
        }
        if (options.autotune_repeats <= 0) {
            throw std::runtime_error("autotune repeats must be positive");
        }
    } else if (options.output_path.empty() || options.vector_path.empty()) {
        throw std::runtime_error("task mode requires an output path");
    }

    return options;
}

std::string wide_to_utf8(const wchar_t* text) {
    if (text == nullptr || *text == L'\0') {
        return {};
    }

    const int size = WideCharToMultiByte(CP_UTF8, 0, text, -1, nullptr, 0, nullptr, nullptr);
    if (size <= 1) {
        return {};
    }

    std::string converted(static_cast<size_t>(size), '\0');
    WideCharToMultiByte(CP_UTF8, 0, text, -1, converted.data(), size, nullptr, nullptr);
    converted.resize(static_cast<size_t>(size - 1));
    return converted;
}

void read_binary_file_into_pointer(const std::string& path, void* destination, std::size_t expected_bytes) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("unable to open file: " + path);
    }

    stream.seekg(0, std::ios::end);
    const auto bytes = static_cast<std::size_t>(stream.tellg());
    stream.seekg(0, std::ios::beg);
    if (bytes != expected_bytes) {
        std::ostringstream builder;
        builder << "unexpected byte size for " << path << ": expected " << expected_bytes << ", got " << bytes;
        throw std::runtime_error(builder.str());
    }

    constexpr std::size_t kChunkBytes = 8 * 1024 * 1024;
    auto* destination_bytes = static_cast<char*>(destination);
    std::size_t copied = 0;
    while (copied < expected_bytes) {
        const std::size_t current_chunk = (std::min)(kChunkBytes, expected_bytes - copied);
        if (!stream.read(destination_bytes + copied, static_cast<std::streamsize>(current_chunk))) {
            throw std::runtime_error("failed to read file: " + path);
        }
        copied += current_chunk;
    }
}

void write_float32_file(const std::string& path, const std::vector<float>& values) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) {
        throw std::runtime_error("unable to open output file: " + path);
    }
    stream.write(
        reinterpret_cast<const char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(float))
    );
    if (!stream) {
        throw std::runtime_error("failed to write output file: " + path);
    }
}

std::string fnv1a64_checksum(const std::vector<float>& values) {
    constexpr std::uint64_t kOffsetBasis = 14695981039346656037ull;
    constexpr std::uint64_t kPrime = 1099511628211ull;

    std::uint64_t hash = kOffsetBasis;
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(values.data());
    const std::size_t byte_count = values.size() * sizeof(float);
    for (std::size_t index = 0; index < byte_count; ++index) {
        hash ^= static_cast<std::uint64_t>(bytes[index]);
        hash *= kPrime;
    }

    std::ostringstream stream;
    stream << "fnv1a64:" << std::hex << std::setw(16) << std::setfill('0') << hash;
    return stream.str();
}

bool contains_nvidia_vendor_name(const std::string& text) {
    std::string lowered = text;
    std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char value) {
        return static_cast<char>(std::tolower(value));
    });
    return lowered.find("nvidia") != std::string::npos;
}

UINT64 align_to_256(UINT64 value) {
    return (value + 255ull) & ~255ull;
}

D3D12_RESOURCE_DESC buffer_desc(UINT64 size, D3D12_RESOURCE_FLAGS flags) {
    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Alignment = 0;
    desc.Width = size;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_UNKNOWN;
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = flags;
    return desc;
}

D3D12_HEAP_PROPERTIES heap_properties(D3D12_HEAP_TYPE type) {
    D3D12_HEAP_PROPERTIES properties = {};
    properties.Type = type;
    properties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    properties.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    properties.CreationNodeMask = 1;
    properties.VisibleNodeMask = 1;
    return properties;
}

D3D12_RESOURCE_BARRIER transition_barrier(
    ID3D12Resource* resource,
    D3D12_RESOURCE_STATES before_state,
    D3D12_RESOURCE_STATES after_state
) {
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.Transition.pResource = resource;
    barrier.Transition.StateBefore = before_state;
    barrier.Transition.StateAfter = after_state;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    return barrier;
}

ComPtr<IDXGIAdapter1> choose_adapter(
    ComPtr<IDXGIFactory6>& factory,
    std::string& device_name,
    std::string& adapter_kind
) {
    ComPtr<IDXGIFactory6> factory6;
    throw_if_failed(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "failed to create DXGI factory");

    if (SUCCEEDED(factory.As(&factory6))) {
        for (UINT index = 0;; ++index) {
            ComPtr<IDXGIAdapter1> adapter;
            if (factory6->EnumAdapterByGpuPreference(
                    index,
                    DXGI_GPU_PREFERENCE_MINIMUM_POWER,
                    IID_PPV_ARGS(&adapter)
                ) == DXGI_ERROR_NOT_FOUND) {
                break;
            }

            DXGI_ADAPTER_DESC1 desc = {};
            adapter->GetDesc1(&desc);
            const std::string description = wide_to_utf8(desc.Description);
            if ((desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0) {
                continue;
            }
            if (contains_nvidia_vendor_name(description)) {
                continue;
            }
            if (FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, _uuidof(ID3D12Device), nullptr))) {
                continue;
            }

            device_name = description;
            adapter_kind = "minimum_power";
            return adapter;
        }
    }

    for (UINT index = 0;; ++index) {
        ComPtr<IDXGIAdapter1> adapter;
        if (factory->EnumAdapters1(index, &adapter) == DXGI_ERROR_NOT_FOUND) {
            break;
        }

        DXGI_ADAPTER_DESC1 desc = {};
        adapter->GetDesc1(&desc);
        const std::string description = wide_to_utf8(desc.Description);
        if ((desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0) {
            continue;
        }
        if (contains_nvidia_vendor_name(description)) {
            continue;
        }
        if (FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, _uuidof(ID3D12Device), nullptr))) {
            continue;
        }

        device_name = description;
        adapter_kind = "hardware";
        return adapter;
    }

    throw std::runtime_error("no non-NVIDIA D3D12 hardware adapter was found");
}

Dx12Runner::Dx12Runner(const Options& options, const std::vector<int>& thread_group_sizes)
    : options_(options),
      total_rows_(static_cast<UINT>(options.rows)),
      output_capacity_bytes_(static_cast<UINT64>(options.rows) * sizeof(float)) {

    adapter_ = choose_adapter(factory_, device_name_, adapter_kind_);
    throw_if_failed(
        D3D12CreateDevice(adapter_.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device_)),
        "failed to create D3D12 device"
    );

    D3D12_COMMAND_QUEUE_DESC queue_desc = {};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    queue_desc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
    queue_desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    throw_if_failed(device_->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue_)), "failed to create command queue");
    throw_if_failed(
        device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator_)),
        "failed to create command allocator"
    );
    throw_if_failed(
        device_->CreateCommandList(
            0,
            D3D12_COMMAND_LIST_TYPE_DIRECT,
            allocator_.Get(),
            nullptr,
            IID_PPV_ARGS(&command_list_)
        ),
        "failed to create command list"
    );
    throw_if_failed(command_list_->Close(), "failed to close initial command list");
    throw_if_failed(device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_)), "failed to create fence");
    fence_event_ = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    if (fence_event_ == nullptr) {
        throw std::runtime_error("failed to create fence event");
    }

    create_root_signature();
    create_pipelines(thread_group_sizes);
    create_resources();
    upload_static_matrix();
    if (!options_.server_mode) {
        upload_vector_from_file(options_.vector_path);
    }
}

Dx12Runner::~Dx12Runner() {
    if (vector_upload_buffer_ != nullptr && vector_upload_mapped_ != nullptr) {
        vector_upload_buffer_->Unmap(0, nullptr);
    }
    if (fence_event_ != nullptr) {
        CloseHandle(fence_event_);
    }
}

void Dx12Runner::create_root_signature() {
    D3D12_ROOT_PARAMETER parameters[4] = {};

    parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    parameters[0].Constants.ShaderRegister = 0;
    parameters[0].Constants.RegisterSpace = 0;
    parameters[0].Constants.Num32BitValues = 5;
    parameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    parameters[1].Descriptor.ShaderRegister = 0;
    parameters[1].Descriptor.RegisterSpace = 0;
    parameters[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    parameters[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    parameters[2].Descriptor.ShaderRegister = 1;
    parameters[2].Descriptor.RegisterSpace = 0;
    parameters[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    parameters[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    parameters[3].Descriptor.ShaderRegister = 0;
    parameters[3].Descriptor.RegisterSpace = 0;
    parameters[3].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_ROOT_SIGNATURE_DESC root_signature_desc = {};
    root_signature_desc.NumParameters = static_cast<UINT>(sizeof(parameters) / sizeof(parameters[0]));
    root_signature_desc.pParameters = parameters;
    root_signature_desc.NumStaticSamplers = 0;
    root_signature_desc.pStaticSamplers = nullptr;
    root_signature_desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    ComPtr<ID3DBlob> signature_blob;
    ComPtr<ID3DBlob> error_blob;
    const HRESULT serialize_status = D3D12SerializeRootSignature(
        &root_signature_desc,
        D3D_ROOT_SIGNATURE_VERSION_1,
        &signature_blob,
        &error_blob
    );
    if (FAILED(serialize_status)) {
        const std::string error_text = error_blob != nullptr
            ? std::string(static_cast<const char*>(error_blob->GetBufferPointer()), error_blob->GetBufferSize())
            : "unknown root-signature serialization error";
        throw std::runtime_error("failed to serialize root signature: " + error_text);
    }

    throw_if_failed(
        device_->CreateRootSignature(
            0,
            signature_blob->GetBufferPointer(),
            signature_blob->GetBufferSize(),
            IID_PPV_ARGS(&root_signature_)
        ),
        "failed to create root signature"
    );
}

void Dx12Runner::create_pipelines(const std::vector<int>& thread_group_sizes) {
    for (const int thread_group_size : thread_group_sizes) {
        const std::string thread_group_text = std::to_string(thread_group_size);
        const D3D_SHADER_MACRO macros[] = {
            {"THREAD_GROUP_SIZE", thread_group_text.c_str()},
            {nullptr, nullptr},
        };

        ComPtr<ID3DBlob> shader_blob;
        ComPtr<ID3DBlob> error_blob;
        const HRESULT compile_status = D3DCompile(
            kShaderSource,
            std::strlen(kShaderSource),
            "fmvm_dx12_runner.hlsl",
            macros,
            nullptr,
            "main",
            "cs_5_1",
            D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3,
            0,
            &shader_blob,
            &error_blob
        );
        if (FAILED(compile_status)) {
            const std::string error_text = error_blob != nullptr
                ? std::string(static_cast<const char*>(error_blob->GetBufferPointer()), error_blob->GetBufferSize())
                : "unknown shader compile error";
            throw std::runtime_error(
                "failed to compile DX12 compute shader for thread_group_size=" + thread_group_text + ": " + error_text
            );
        }

        D3D12_COMPUTE_PIPELINE_STATE_DESC pipeline_desc = {};
        pipeline_desc.pRootSignature = root_signature_.Get();
        pipeline_desc.CS.pShaderBytecode = shader_blob->GetBufferPointer();
        pipeline_desc.CS.BytecodeLength = shader_blob->GetBufferSize();

        ComPtr<ID3D12PipelineState> pipeline_state;
        throw_if_failed(
            device_->CreateComputePipelineState(&pipeline_desc, IID_PPV_ARGS(&pipeline_state)),
            "failed to create compute pipeline state"
        );
        pipelines_.push_back({thread_group_size, pipeline_state});
    }
}

void Dx12Runner::create_resources() {
    const UINT64 matrix_bytes = static_cast<UINT64>(options_.rows) * static_cast<UINT64>(options_.cols) * sizeof(float);
    const UINT64 vector_bytes = static_cast<UINT64>(options_.cols) * sizeof(float);

    auto default_heap = heap_properties(D3D12_HEAP_TYPE_DEFAULT);
    auto readback_heap = heap_properties(D3D12_HEAP_TYPE_READBACK);

    auto matrix_desc = buffer_desc(matrix_bytes);
    auto vector_desc = buffer_desc(vector_bytes);
    auto output_desc = buffer_desc(output_capacity_bytes_, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    auto readback_desc = buffer_desc(output_capacity_bytes_);
    auto vector_upload_desc = buffer_desc(vector_bytes);

    throw_if_failed(
        device_->CreateCommittedResource(
            &default_heap,
            D3D12_HEAP_FLAG_NONE,
            &matrix_desc,
            D3D12_RESOURCE_STATE_COPY_DEST,
            nullptr,
            IID_PPV_ARGS(&matrix_buffer_)
        ),
        "failed to create matrix buffer"
    );
    throw_if_failed(
        device_->CreateCommittedResource(
            &default_heap,
            D3D12_HEAP_FLAG_NONE,
            &vector_desc,
            D3D12_RESOURCE_STATE_COPY_DEST,
            nullptr,
            IID_PPV_ARGS(&vector_buffer_)
        ),
        "failed to create vector buffer"
    );
    throw_if_failed(
        device_->CreateCommittedResource(
            &default_heap,
            D3D12_HEAP_FLAG_NONE,
            &output_desc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            nullptr,
            IID_PPV_ARGS(&output_buffer_)
        ),
        "failed to create output buffer"
    );
    throw_if_failed(
        device_->CreateCommittedResource(
            &readback_heap,
            D3D12_HEAP_FLAG_NONE,
            &readback_desc,
            D3D12_RESOURCE_STATE_COPY_DEST,
            nullptr,
            IID_PPV_ARGS(&readback_buffer_)
        ),
        "failed to create readback buffer"
    );
    auto upload_heap = heap_properties(D3D12_HEAP_TYPE_UPLOAD);
    throw_if_failed(
        device_->CreateCommittedResource(
            &upload_heap,
            D3D12_HEAP_FLAG_NONE,
            &vector_upload_desc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS(&vector_upload_buffer_)
        ),
        "failed to create vector upload buffer"
    );
    throw_if_failed(vector_upload_buffer_->Map(0, nullptr, &vector_upload_mapped_), "failed to persistently map vector upload buffer");
}

void Dx12Runner::upload_static_matrix() {
    const auto started = std::chrono::steady_clock::now();
    const UINT64 matrix_bytes = static_cast<UINT64>(options_.rows) * static_cast<UINT64>(options_.cols) * sizeof(float);

    auto upload_heap = heap_properties(D3D12_HEAP_TYPE_UPLOAD);
    auto matrix_upload_desc = buffer_desc(matrix_bytes);

    ComPtr<ID3D12Resource> matrix_upload;
    throw_if_failed(
        device_->CreateCommittedResource(
            &upload_heap,
            D3D12_HEAP_FLAG_NONE,
            &matrix_upload_desc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS(&matrix_upload)
        ),
        "failed to create matrix upload buffer"
    );

    void* matrix_mapped = nullptr;
    throw_if_failed(matrix_upload->Map(0, nullptr, &matrix_mapped), "failed to map matrix upload buffer");
    read_binary_file_into_pointer(options_.matrix_path, matrix_mapped, static_cast<std::size_t>(matrix_bytes));
    matrix_upload->Unmap(0, nullptr);

    throw_if_failed(allocator_->Reset(), "failed to reset command allocator for upload");
    throw_if_failed(command_list_->Reset(allocator_.Get(), nullptr), "failed to reset command list for upload");
    command_list_->CopyBufferRegion(matrix_buffer_.Get(), 0, matrix_upload.Get(), 0, matrix_bytes);
    auto matrix_ready = transition_barrier(
        matrix_buffer_.Get(),
        D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
    );
    command_list_->ResourceBarrier(1, &matrix_ready);
    throw_if_failed(command_list_->Close(), "failed to close upload command list");

    ID3D12CommandList* command_lists[] = {command_list_.Get()};
    queue_->ExecuteCommandLists(1, command_lists);
    wait_for_gpu();
    static_upload_wall_clock_latency_seconds_ =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
}

void Dx12Runner::upload_vector_from_file(const std::string& vector_path) {
    const auto started = std::chrono::steady_clock::now();
    const UINT64 vector_bytes = static_cast<UINT64>(options_.cols) * sizeof(float);
    if (vector_upload_mapped_ == nullptr) {
        throw std::runtime_error("vector upload buffer is not mapped");
    }
    read_binary_file_into_pointer(vector_path, vector_upload_mapped_, static_cast<std::size_t>(vector_bytes));

    throw_if_failed(allocator_->Reset(), "failed to reset command allocator for vector upload");
    throw_if_failed(command_list_->Reset(allocator_.Get(), nullptr), "failed to reset command list for vector upload");

    if (vector_uploaded_) {
        auto to_copy_dest = transition_barrier(
            vector_buffer_.Get(),
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_COPY_DEST
        );
        command_list_->ResourceBarrier(1, &to_copy_dest);
    }
    command_list_->CopyBufferRegion(vector_buffer_.Get(), 0, vector_upload_buffer_.Get(), 0, vector_bytes);
    auto to_srv = transition_barrier(
        vector_buffer_.Get(),
        D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
    );
    command_list_->ResourceBarrier(1, &to_srv);

    throw_if_failed(command_list_->Close(), "failed to close vector upload command list");
    ID3D12CommandList* command_lists[] = {command_list_.Get()};
    queue_->ExecuteCommandLists(1, command_lists);
    wait_for_gpu();

    last_vector_upload_wall_clock_latency_seconds_ =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    vector_uploaded_ = true;
}

void Dx12Runner::wait_for_gpu() {
    const UINT64 fence_value = next_fence_value_++;
    throw_if_failed(queue_->Signal(fence_.Get(), fence_value), "failed to signal fence");
    if (fence_->GetCompletedValue() < fence_value) {
        throw_if_failed(
            fence_->SetEventOnCompletion(fence_value, fence_event_),
            "failed to register fence completion event"
        );
        WaitForSingleObject(fence_event_, INFINITE);
    }
}

ComPtr<ID3D12PipelineState> Dx12Runner::pipeline_for_thread_group_size(int thread_group_size) const {
    for (const auto& [candidate_size, pipeline_state] : pipelines_) {
        if (candidate_size == thread_group_size) {
            return pipeline_state;
        }
    }
    throw std::runtime_error("unsupported DX12 thread_group_size");
}

PhaseMetrics Dx12Runner::run_configuration(int thread_group_size, int rows_per_thread, int repeats) {
    if (repeats <= 0) {
        throw std::runtime_error("repeats must be positive");
    }
    const UINT row_count = static_cast<UINT>(options_.row_end - options_.row_start);
    const UINT64 output_bytes = static_cast<UINT64>(row_count) * sizeof(float);
    if (row_count == 0) {
        throw std::runtime_error("row range resolved to zero output rows");
    }

    auto pipeline_state = pipeline_for_thread_group_size(thread_group_size);

    throw_if_failed(allocator_->Reset(), "failed to reset command allocator");
    throw_if_failed(command_list_->Reset(allocator_.Get(), pipeline_state.Get()), "failed to reset command list");

    command_list_->SetComputeRootSignature(root_signature_.Get());
    command_list_->SetComputeRootShaderResourceView(1, matrix_buffer_->GetGPUVirtualAddress());
    command_list_->SetComputeRootShaderResourceView(2, vector_buffer_->GetGPUVirtualAddress());
    command_list_->SetComputeRootUnorderedAccessView(3, output_buffer_->GetGPUVirtualAddress());

    constexpr UINT kMaxRowsPerDispatch = 512;
    int dispatches_per_repeat = 0;

    const auto started = std::chrono::steady_clock::now();
    for (int repeat = 0; repeat < repeats; ++repeat) {
        for (UINT output_base_row = 0; output_base_row < row_count; output_base_row += kMaxRowsPerDispatch) {
            const UINT chunk_rows = (std::min)(kMaxRowsPerDispatch, row_count - output_base_row);
            const UINT dispatch_x = (chunk_rows + static_cast<UINT>(rows_per_thread) - 1u) /
                static_cast<UINT>(rows_per_thread);
            const std::uint32_t root_constants[5] = {
                static_cast<std::uint32_t>(options_.cols),
                static_cast<std::uint32_t>(options_.row_start + static_cast<int>(output_base_row)),
                output_base_row,
                chunk_rows,
                static_cast<std::uint32_t>(rows_per_thread),
            };
            command_list_->SetComputeRoot32BitConstants(0, 5, root_constants, 0);
            command_list_->Dispatch(dispatch_x, 1, 1);
            dispatches_per_repeat += 1;
        }
    }

    auto to_copy = transition_barrier(
        output_buffer_.Get(),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_COPY_SOURCE
    );
    command_list_->ResourceBarrier(1, &to_copy);
    command_list_->CopyBufferRegion(readback_buffer_.Get(), 0, output_buffer_.Get(), 0, output_bytes);
    auto back_to_uav = transition_barrier(
        output_buffer_.Get(),
        D3D12_RESOURCE_STATE_COPY_SOURCE,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS
    );
    command_list_->ResourceBarrier(1, &back_to_uav);
    throw_if_failed(command_list_->Close(), "failed to close command list");

    ID3D12CommandList* command_lists[] = {command_list_.Get()};
    queue_->ExecuteCommandLists(1, command_lists);
    wait_for_gpu();
    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();

    std::vector<float> output_values = read_output_values(row_count, output_bytes);

    PhaseMetrics metrics;
    metrics.repeats = repeats;
    metrics.wall_clock_latency_seconds = elapsed;
    metrics.effective_gflops =
        (2.0 * static_cast<double>(row_count) * static_cast<double>(options_.cols) * static_cast<double>(repeats)) /
        (elapsed * 1'000'000'000.0);
    metrics.checksum = fnv1a64_checksum(output_values);
    metrics.dispatches_per_repeat = dispatches_per_repeat / repeats;
    return metrics;
}

std::vector<float> Dx12Runner::read_output_values(UINT row_count, UINT64 output_bytes) {
    std::vector<float> output_values(static_cast<size_t>(row_count));
    void* mapped_readback = nullptr;
    D3D12_RANGE range = {0, static_cast<SIZE_T>(output_bytes)};
    throw_if_failed(readback_buffer_->Map(0, &range, &mapped_readback), "failed to map readback buffer");
    std::memcpy(output_values.data(), mapped_readback, static_cast<size_t>(output_bytes));
    D3D12_RANGE empty_range = {0, 0};
    readback_buffer_->Unmap(0, &empty_range);
    return output_values;
}

std::vector<float> Dx12Runner::run_task_output(
    int thread_group_size,
    int rows_per_thread,
    int repeats,
    PhaseMetrics& metrics
) {
    metrics = run_configuration(thread_group_size, rows_per_thread, repeats);
    const UINT row_count = static_cast<UINT>(options_.row_end - options_.row_start);
    const UINT64 output_bytes = static_cast<UINT64>(row_count) * sizeof(float);
    return read_output_values(row_count, output_bytes);
}

std::vector<float> Dx12Runner::run_task_output_from_vector(
    const std::string& vector_path,
    int row_start,
    int row_end,
    int thread_group_size,
    int rows_per_thread,
    int repeats,
    PhaseMetrics& metrics
) {
    options_.row_start = row_start;
    options_.row_end = row_end;
    upload_vector_from_file(vector_path);
    return run_task_output(thread_group_size, rows_per_thread, repeats, metrics);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);
        const std::vector<int> thread_group_sizes = options.task_mode
            ? std::vector<int>{options.fixed_thread_group_size}
            : options.thread_group_sizes;
        const auto setup_started = std::chrono::steady_clock::now();
        Dx12Runner runner(options, thread_group_sizes);
        const double setup_wall_clock_latency_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - setup_started).count();

        if (options.server_mode) {
            std::cout << "READY\t"
                      << "{\"backend\":\"dx12\",\"device_name\":\"" << runner.device_name()
                      << "\",\"adapter_kind\":\"" << runner.adapter_kind()
                      << "\",\"setup_wall_clock_latency_seconds\":" << std::fixed << std::setprecision(9)
                      << setup_wall_clock_latency_seconds
                      << ",\"static_upload_wall_clock_latency_seconds\":" << std::fixed << std::setprecision(9)
                      << runner.static_upload_wall_clock_latency_seconds()
                      << ",\"last_vector_upload_wall_clock_latency_seconds\":" << std::fixed << std::setprecision(9)
                      << runner.last_vector_upload_wall_clock_latency_seconds()
                      << "}" << std::endl;

            std::string line;
            while (std::getline(std::cin, line)) {
                if (line == "QUIT") {
                    break;
                }
                if (line.empty()) {
                    continue;
                }

                try {
                    const auto fields = split_tab_fields(line);
                    if (fields.size() != 8 || fields[0] != "RUN") {
                        throw std::runtime_error("server command must be RUN<TAB>vector<TAB>output<TAB>row_start<TAB>row_end<TAB>thread_group_size<TAB>rows_per_thread<TAB>iteration_count");
                    }

                    const std::string& vector_path = fields[1];
                    const std::string& output_path = fields[2];
                    const int row_start = std::stoi(fields[3]);
                    const int row_end = std::stoi(fields[4]);
                    const int thread_group_size = std::stoi(fields[5]);
                    const int rows_per_thread = std::stoi(fields[6]);
                    const int iteration_count = std::stoi(fields[7]);

                    PhaseMetrics metrics;
                    const auto output_values = runner.run_task_output_from_vector(
                        vector_path,
                        row_start,
                        row_end,
                        thread_group_size,
                        rows_per_thread,
                        iteration_count,
                        metrics
                    );
                    write_float32_file(output_path, output_values);

                    std::cout << "OK\t"
                              << "{"
                              << "\"backend\":\"dx12\","
                              << "\"device_name\":\"" << runner.device_name() << "\","
                              << "\"adapter_kind\":\"" << runner.adapter_kind() << "\","
                              << "\"row_start\":" << row_start << ","
                              << "\"row_end\":" << row_end << ","
                              << "\"thread_group_size\":" << thread_group_size << ","
                              << "\"rows_per_thread\":" << rows_per_thread << ","
                              << "\"iteration_count\":" << iteration_count << ","
                              << "\"kernel_layout\":\"thread_group_reduction\","
                              << "\"dispatches_per_repeat\":" << metrics.dispatches_per_repeat << ","
                              << "\"vector_upload_wall_clock_latency_seconds\":" << std::fixed << std::setprecision(9)
                              << runner.last_vector_upload_wall_clock_latency_seconds() << ","
                              << "\"wall_clock_latency_seconds\":" << std::fixed << std::setprecision(9)
                              << metrics.wall_clock_latency_seconds << ","
                              << "\"effective_gflops\":" << std::fixed << std::setprecision(6)
                              << metrics.effective_gflops << ","
                              << "\"checksum\":\"" << metrics.checksum << "\""
                              << "}" << std::endl;
                } catch (const std::exception& exc) {
                    std::cout << "ERR\t" << exc.what() << std::endl;
                }
            }
            return 0;
        }

        if (options.task_mode) {
            PhaseMetrics metrics;
            const auto output_values = runner.run_task_output(
                options.fixed_thread_group_size,
                options.fixed_rows_per_thread,
                options.measurement_repeats,
                metrics
            );
            write_float32_file(options.output_path, output_values);

            std::cout << "{"
                      << "\"mode\":\"task\","
                      << "\"backend\":\"dx12\","
                      << "\"device_name\":\"" << runner.device_name() << "\","
                      << "\"adapter_kind\":\"" << runner.adapter_kind() << "\","
                      << "\"row_start\":" << options.row_start << ","
                      << "\"row_end\":" << options.row_end << ","
                      << "\"thread_group_size\":" << options.fixed_thread_group_size << ","
                      << "\"rows_per_thread\":" << options.fixed_rows_per_thread << ","
                      << "\"iteration_count\":" << options.measurement_repeats << ","
                      << "\"accumulation_precision\":\"fp32\","
                      << "\"kernel_layout\":\"thread_group_reduction\","
                      << "\"setup_wall_clock_latency_seconds\":" << std::fixed << std::setprecision(9)
                      << setup_wall_clock_latency_seconds << ","
                      << "\"static_upload_wall_clock_latency_seconds\":" << std::fixed << std::setprecision(9)
                      << runner.static_upload_wall_clock_latency_seconds() << ","
                      << "\"vector_upload_wall_clock_latency_seconds\":" << std::fixed << std::setprecision(9)
                      << runner.last_vector_upload_wall_clock_latency_seconds() << ","
                      << "\"dispatches_per_repeat\":" << metrics.dispatches_per_repeat << ","
                      << "\"wall_clock_latency_seconds\":" << std::fixed << std::setprecision(9)
                      << metrics.wall_clock_latency_seconds << ","
                      << "\"effective_gflops\":" << std::fixed << std::setprecision(6) << metrics.effective_gflops
                      << ","
                      << "\"checksum\":\"" << metrics.checksum << "\""
                      << "}\n";
            return 0;
        }

        TrialMetrics best_metrics;
        bool has_best_metrics = false;
        int trials_run = 0;
        for (const int thread_group_size : options.thread_group_sizes) {
            for (const int rows_per_thread : options.rows_per_thread_values) {
                TrialMetrics candidate;
                candidate.thread_group_size = thread_group_size;
                candidate.rows_per_thread = rows_per_thread;
                candidate.autotune = runner.run_configuration(thread_group_size, rows_per_thread, options.autotune_repeats);
                ++trials_run;
                if (!has_best_metrics ||
                    candidate.autotune.wall_clock_latency_seconds < best_metrics.autotune.wall_clock_latency_seconds) {
                    best_metrics = candidate;
                    has_best_metrics = true;
                }
            }
        }

        if (!has_best_metrics) {
            throw std::runtime_error("DX12 benchmark ran zero valid trials");
        }

        best_metrics.measurement = runner.run_configuration(
            best_metrics.thread_group_size,
            best_metrics.rows_per_thread,
            options.measurement_repeats
        );

        std::cout << "{"
                  << "\"mode\":\"benchmark\","
                  << "\"backend\":\"dx12\","
                  << "\"device_name\":\"" << runner.device_name() << "\","
                  << "\"adapter_kind\":\"" << runner.adapter_kind() << "\","
                  << "\"thread_group_size\":" << best_metrics.thread_group_size << ","
                  << "\"rows_per_thread\":" << best_metrics.rows_per_thread << ","
                  << "\"autotune_repeats\":" << options.autotune_repeats << ","
                  << "\"measurement_repeats\":" << options.measurement_repeats << ","
                  << "\"trials_run\":" << trials_run << ","
                  << "\"accumulation_precision\":\"fp32\","
                  << "\"kernel_layout\":\"thread_group_reduction\","
                  << "\"setup_wall_clock_latency_seconds\":" << std::fixed << std::setprecision(9)
                  << setup_wall_clock_latency_seconds << ","
                  << "\"static_upload_wall_clock_latency_seconds\":" << std::fixed << std::setprecision(9)
                  << runner.static_upload_wall_clock_latency_seconds() << ","
                  << "\"vector_upload_wall_clock_latency_seconds\":" << std::fixed << std::setprecision(9)
                  << runner.last_vector_upload_wall_clock_latency_seconds() << ","
                  << "\"dispatches_per_repeat\":" << best_metrics.measurement.dispatches_per_repeat << ","
                  << "\"autotune_wall_clock_latency_seconds\":" << std::fixed << std::setprecision(9)
                  << best_metrics.autotune.wall_clock_latency_seconds << ","
                  << "\"autotune_effective_gflops\":" << std::fixed << std::setprecision(6)
                  << best_metrics.autotune.effective_gflops << ","
                  << "\"autotune_checksum\":\"" << best_metrics.autotune.checksum << "\","
                  << "\"measurement_wall_clock_latency_seconds\":" << std::fixed << std::setprecision(9)
                  << best_metrics.measurement.wall_clock_latency_seconds << ","
                  << "\"measurement_effective_gflops\":" << std::fixed << std::setprecision(6)
                  << best_metrics.measurement.effective_gflops << ","
                  << "\"measurement_checksum\":\"" << best_metrics.measurement.checksum << "\""
                  << "}\n";
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << exc.what() << '\n';
        return 1;
    }
}
