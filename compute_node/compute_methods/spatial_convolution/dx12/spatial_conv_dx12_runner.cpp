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
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace {

constexpr int kMaxTileSize = 8;
constexpr UINT64 kTargetOutputBatchBytes = 128ull * 1024ull * 1024ull;

constexpr const char* kShaderSource = R"(
cbuffer Params : register(b0)
{
    uint h;
    uint w;
    uint c_in;
    uint k;
    uint pad;
    uint stride;
    uint out_h;
    uint out_w;
    uint oc_offset;
    uint batch_c_out;
    uint tile_size;
};

ByteAddressBuffer Input : register(t0);
ByteAddressBuffer Weight : register(t1);
RWStructuredBuffer<float> Output : register(u0);

static const uint MAX_TILE_SIZE = 8;

[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint3 group_thread_id : SV_GroupThreadID)
{
    const uint linear_thread = group_id.x * THREAD_GROUP_SIZE + group_thread_id.x;
    const uint total_outputs = out_h * out_w * batch_c_out;
    const uint base_linear = linear_thread * tile_size;

    [unroll]
    for (uint lane = 0; lane < MAX_TILE_SIZE; ++lane)
    {
        if (lane >= tile_size)
        {
            continue;
        }

        const uint linear_index = base_linear + lane;
        if (linear_index >= total_outputs)
        {
            continue;
        }

        const uint spatial_index = linear_index / batch_c_out;
        const uint local_oc = linear_index % batch_c_out;
        const uint oh = spatial_index / out_w;
        const uint ow = spatial_index % out_w;
        const uint oc = oc_offset + local_oc;

        float sum = 0.0f;
        [loop]
        for (uint kh = 0; kh < k; ++kh)
        {
            const int ih = int(oh) * int(stride) - int(pad) + int(kh);
            if (ih < 0 || ih >= int(h))
            {
                continue;
            }

            [loop]
            for (uint kw = 0; kw < k; ++kw)
            {
                const int iw = int(ow) * int(stride) - int(pad) + int(kw);
                if (iw < 0 || iw >= int(w))
                {
                    continue;
                }

                const uint input_base = (uint(ih) * w + uint(iw)) * c_in;
                const uint weight_base = (((oc * k) + kh) * k + kw) * c_in;
                [loop]
                for (uint ic = 0; ic < c_in; ++ic)
                {
                    const float input_value = asfloat(Input.Load((input_base + ic) * 4));
                    const float weight_value = asfloat(Weight.Load((weight_base + ic) * 4));
                    sum += input_value * weight_value;
                }
            }
        }

        Output[linear_index] = sum;
    }
}
)";

struct Options {
    std::string input_path;
    std::string weight_path;
    std::string output_path;
    int h = 0;
    int w = 0;
    int c_in = 0;
    int c_out = 0;
    int k = 0;
    int pad = 0;
    int stride = 1;
    std::vector<int> thread_group_sizes;
    std::vector<int> tile_sizes;
    int fixed_thread_group_size = 0;
    int fixed_tile_size = 0;
    int autotune_repeats = 1;
    int measurement_repeats = 1;
    bool task_mode = false;
};

struct PhaseMetrics {
    int repeats = 0;
    double wall_clock_latency_seconds = 0.0;
    double effective_gflops = 0.0;
    std::string checksum;
    int dispatches_per_repeat = 0;
};

struct TrialMetrics {
    int thread_group_size = 0;
    int tile_size = 0;
    PhaseMetrics autotune;
    PhaseMetrics measurement;
};

struct StatusTargets {
    std::string status_path;
    std::string trace_path;
    std::string run_id;

    bool enabled() const {
        return !status_path.empty() && !trace_path.empty() && !run_id.empty();
    }
};

std::string json_escape(const std::string& text) {
    std::ostringstream builder;
    for (const unsigned char value : text) {
        switch (value) {
        case '\\':
            builder << "\\\\";
            break;
        case '"':
            builder << "\\\"";
            break;
        case '\b':
            builder << "\\b";
            break;
        case '\f':
            builder << "\\f";
            break;
        case '\n':
            builder << "\\n";
            break;
        case '\r':
            builder << "\\r";
            break;
        case '\t':
            builder << "\\t";
            break;
        default:
            if (value < 0x20) {
                builder << "\\u"
                        << std::hex
                        << std::setw(4)
                        << std::setfill('0')
                        << static_cast<int>(value)
                        << std::dec
                        << std::setfill(' ');
            } else {
                builder << static_cast<char>(value);
            }
            break;
        }
    }
    return builder.str();
}

std::string json_string_field(const char* key, const std::string& value) {
    return "\"" + std::string(key) + "\":\"" + json_escape(value) + "\"";
}

std::string json_int_field(const char* key, long long value) {
    return "\"" + std::string(key) + "\":" + std::to_string(value);
}

std::string json_number_field(const char* key, double value) {
    std::ostringstream builder;
    builder << "\"" << key << "\":" << std::fixed << std::setprecision(6) << value;
    return builder.str();
}

std::string json_bool_field(const char* key, bool value) {
    return "\"" + std::string(key) + "\":" + (value ? "true" : "false");
}

StatusTargets status_targets() {
    const char* status_path = std::getenv("SUPERWEB_BENCHMARK_STATUS_PATH");
    const char* trace_path = std::getenv("SUPERWEB_BENCHMARK_TRACE_PATH");
    const char* run_id = std::getenv("SUPERWEB_BENCHMARK_RUN_ID");
    if (status_path == nullptr || trace_path == nullptr || run_id == nullptr) {
        return {};
    }
    return {status_path, trace_path, run_id};
}

std::string iso_timestamp_utc(std::chrono::system_clock::time_point now) {
    const std::time_t timestamp = std::chrono::system_clock::to_time_t(now);
    std::tm utc_time = {};
    gmtime_s(&utc_time, &timestamp);

    std::ostringstream builder;
    builder << std::put_time(&utc_time, "%Y-%m-%dT%H:%M:%S");
    const auto micros =
        std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count() % 1'000'000ll;
    builder << "." << std::setw(6) << std::setfill('0') << micros << "+00:00";
    return builder.str();
}

std::string join_json_fields(const std::vector<std::string>& fields) {
    std::ostringstream builder;
    builder << "{";
    for (size_t index = 0; index < fields.size(); ++index) {
        if (index != 0) {
            builder << ",";
        }
        builder << fields[index];
    }
    builder << "}";
    return builder.str();
}

void write_text_atomic(const std::string& path_text, const std::string& content) {
    namespace fs = std::filesystem;
    std::error_code error;
    const fs::path path(path_text);
    if (!path.has_parent_path()) {
        return;
    }
    fs::create_directories(path.parent_path(), error);

    const fs::path temp_path = path.parent_path() / (
        path.filename().string() +
        ".tmp." +
        std::to_string(static_cast<unsigned long long>(GetCurrentProcessId())) +
        "." +
        std::to_string(static_cast<unsigned long long>(GetTickCount64()))
    );

    {
        std::ofstream stream(temp_path, std::ios::binary | std::ios::trunc);
        if (!stream) {
            return;
        }
        stream.write(content.data(), static_cast<std::streamsize>(content.size()));
        stream.flush();
    }

    fs::remove(path, error);
    error.clear();
    fs::rename(temp_path, path, error);
    if (error) {
        error.clear();
        fs::copy_file(temp_path, path, fs::copy_options::overwrite_existing, error);
        fs::remove(temp_path, error);
    }
}

void append_text_line(const std::string& path_text, const std::string& line) {
    namespace fs = std::filesystem;
    std::error_code error;
    const fs::path path(path_text);
    if (path.has_parent_path()) {
        fs::create_directories(path.parent_path(), error);
    }

    std::ofstream stream(path, std::ios::binary | std::ios::app);
    if (!stream) {
        return;
    }
    stream.write(line.data(), static_cast<std::streamsize>(line.size()));
    stream.flush();
}

void emit_native_status(const std::string& event, std::vector<std::string> extra_fields) {
    const StatusTargets targets = status_targets();
    if (!targets.enabled()) {
        return;
    }

    const auto now = std::chrono::system_clock::now();
    const double unix_seconds = std::chrono::duration<double>(now.time_since_epoch()).count();
    std::vector<std::string> fields;
    fields.reserve(extra_fields.size() + 5);
    fields.push_back(json_string_field("run_id", targets.run_id));
    fields.push_back(json_string_field("event", event));
    fields.push_back(json_number_field("timestamp_unix", unix_seconds));
    fields.push_back(json_string_field("timestamp_iso", iso_timestamp_utc(now)));
    fields.push_back(json_int_field("pid", static_cast<long long>(GetCurrentProcessId())));
    for (std::string& field : extra_fields) {
        fields.push_back(std::move(field));
    }

    const std::string record = join_json_fields(fields);
    write_text_atomic(targets.status_path, record + "\n");
    append_text_line(targets.trace_path, record + "\n");
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
        if (!item.empty()) {
            values.push_back(std::stoi(item));
        }
    }
    return values;
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

bool contains_nvidia_vendor_name(const std::string& text) {
    std::string lowered = text;
    std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char value) {
        return static_cast<char>(std::tolower(value));
    });
    return lowered.find("nvidia") != std::string::npos;
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

D3D12_RESOURCE_DESC buffer_desc(UINT64 size, D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE) {
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

int choose_output_batch_channels(int c_out, UINT64 spatial_size) {
    const UINT64 bytes_per_channel = spatial_size * sizeof(float);
    if (bytes_per_channel == 0) {
        return 1;
    }

    int channels = static_cast<int>(kTargetOutputBatchBytes / bytes_per_channel);
    channels = (std::max)(1, channels);
    channels = (std::min)(channels, c_out);
    if (channels >= 16) {
        channels = (channels / 16) * 16;
    } else if (channels >= 8) {
        channels = (channels / 8) * 8;
    } else if (channels >= 4) {
        channels = (channels / 4) * 4;
    }
    return (std::max)(1, channels);
}

Options parse_args(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc) {
            throw std::runtime_error("missing value for command line flag");
        }

        const std::string key = argv[index];
        const std::string value = argv[index + 1];
        if (key == "--input") {
            options.input_path = value;
        } else if (key == "--weight") {
            options.weight_path = value;
        } else if (key == "--output") {
            options.output_path = value;
        } else if (key == "--h") {
            options.h = std::stoi(value);
        } else if (key == "--w") {
            options.w = std::stoi(value);
        } else if (key == "--cin") {
            options.c_in = std::stoi(value);
        } else if (key == "--cout") {
            options.c_out = std::stoi(value);
        } else if (key == "--k") {
            options.k = std::stoi(value);
        } else if (key == "--pad") {
            options.pad = std::stoi(value);
        } else if (key == "--stride") {
            options.stride = std::stoi(value);
        } else if (key == "--thread-group-sizes") {
            options.thread_group_sizes = parse_int_list(value);
        } else if (key == "--tile-sizes") {
            options.tile_sizes = parse_int_list(value);
        } else if (key == "--fixed-thread-group-size") {
            options.fixed_thread_group_size = std::stoi(value);
        } else if (key == "--fixed-tile-size") {
            options.fixed_tile_size = std::stoi(value);
        } else if (key == "--autotune-repeats") {
            options.autotune_repeats = std::stoi(value);
        } else if (key == "--measurement-repeats" || key == "--iteration-count") {
            options.measurement_repeats = std::stoi(value);
        } else {
            throw std::runtime_error("unknown flag: " + key);
        }
    }

    if (options.input_path.empty() || options.weight_path.empty()) {
        throw std::runtime_error("input and weight paths are required");
    }
    if (options.h <= 0 || options.w <= 0 || options.c_in <= 0 || options.c_out <= 0 || options.k <= 0 || options.stride <= 0) {
        throw std::runtime_error("invalid convolution dimensions");
    }

    const int out_h = (options.h + 2 * options.pad - options.k) / options.stride + 1;
    const int out_w = (options.w + 2 * options.pad - options.k) / options.stride + 1;
    if (out_h <= 0 || out_w <= 0) {
        throw std::runtime_error("invalid output shape after applying padding/kernel size/stride");
    }
    if (options.measurement_repeats <= 0 || options.autotune_repeats <= 0) {
        throw std::runtime_error("repeat counts must be positive");
    }

    const bool has_fixed_thread_group_size = options.fixed_thread_group_size > 0;
    const bool has_fixed_tile_size = options.fixed_tile_size > 0;
    if (has_fixed_thread_group_size != has_fixed_tile_size) {
        throw std::runtime_error("fixed execution requires fixed-thread-group-size and fixed-tile-size");
    }
    options.task_mode = has_fixed_thread_group_size && has_fixed_tile_size;

    if (options.thread_group_sizes.empty()) {
        options.thread_group_sizes = {64, 128, 256, 512};
    }
    if (options.tile_sizes.empty()) {
        options.tile_sizes = {1, 2, 4, 8};
    }
    options.tile_sizes.erase(
        std::remove_if(
            options.tile_sizes.begin(),
            options.tile_sizes.end(),
            [&](int value) { return value <= 0 || value > kMaxTileSize || value > options.c_out; }
        ),
        options.tile_sizes.end()
    );
    if (options.tile_sizes.empty()) {
        options.tile_sizes.push_back((std::min)(options.c_out, kMaxTileSize));
    }

    if (options.task_mode && options.output_path.empty()) {
        throw std::runtime_error("fixed execution requires an output path");
    }

    return options;
}

class Dx12Runner {
public:
    Dx12Runner(const Options& options, const std::vector<int>& thread_group_sizes);
    ~Dx12Runner();

    PhaseMetrics run_configuration(
        int thread_group_size,
        int tile_size,
        int repeats,
        const std::string* output_path = nullptr
    );

    const std::string& device_name() const { return device_name_; }
    const std::string& adapter_kind() const { return adapter_kind_; }
    double static_upload_wall_clock_latency_seconds() const { return static_upload_wall_clock_latency_seconds_; }

private:
    void create_root_signature();
    void create_pipelines(const std::vector<int>& thread_group_sizes);
    void create_resources();
    void upload_static_inputs();
    void wait_for_gpu();
    ComPtr<ID3D12PipelineState> pipeline_for_thread_group_size(int thread_group_size) const;
    void dispatch_capture_batch(
        int thread_group_size,
        int tile_size,
        int oc_offset,
        int batch_c_out,
        std::vector<float>* full_output,
        double& checksum
    );

    Options options_;
    UINT out_h_ = 0;
    UINT out_w_ = 0;
    UINT64 spatial_size_ = 0;
    int output_batch_channels_ = 1;
    UINT64 output_batch_capacity_bytes_ = 0;
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
    ComPtr<ID3D12Resource> input_buffer_;
    ComPtr<ID3D12Resource> weight_buffer_;
    ComPtr<ID3D12Resource> output_buffer_;
    ComPtr<ID3D12Resource> readback_buffer_;
    std::string device_name_;
    std::string adapter_kind_;
    double static_upload_wall_clock_latency_seconds_ = 0.0;
};

Dx12Runner::Dx12Runner(const Options& options, const std::vector<int>& thread_group_sizes)
    : options_(options),
      out_h_(static_cast<UINT>((options.h + 2 * options.pad - options.k) / options.stride + 1)),
      out_w_(static_cast<UINT>((options.w + 2 * options.pad - options.k) / options.stride + 1)),
      spatial_size_(static_cast<UINT64>(out_h_) * static_cast<UINT64>(out_w_)) {

    output_batch_channels_ = choose_output_batch_channels(options_.c_out, spatial_size_);
    output_batch_capacity_bytes_ = spatial_size_ * static_cast<UINT64>(output_batch_channels_) * sizeof(float);

    adapter_ = choose_adapter(factory_, device_name_, adapter_kind_);
    throw_if_failed(
        D3D12CreateDevice(adapter_.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device_)),
        "failed to create D3D12 device"
    );

    D3D12_COMMAND_QUEUE_DESC queue_desc = {};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
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
    upload_static_inputs();
}

Dx12Runner::~Dx12Runner() {
    if (fence_event_ != nullptr) {
        CloseHandle(fence_event_);
    }
}

void Dx12Runner::create_root_signature() {
    D3D12_ROOT_PARAMETER parameters[4] = {};

    parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    parameters[0].Constants.ShaderRegister = 0;
    parameters[0].Constants.RegisterSpace = 0;
    parameters[0].Constants.Num32BitValues = 11;
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
            nullptr,
            macros,
            nullptr,
            "main",
            "cs_5_0",
            D3DCOMPILE_OPTIMIZATION_LEVEL3,
            0,
            &shader_blob,
            &error_blob
        );
        if (FAILED(compile_status)) {
            const std::string error_text = error_blob != nullptr
                ? std::string(static_cast<const char*>(error_blob->GetBufferPointer()), error_blob->GetBufferSize())
                : "unknown shader compile error";
            throw std::runtime_error("failed to compile DX12 compute shader: " + error_text);
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
    const UINT64 input_bytes = static_cast<UINT64>(options_.h) * static_cast<UINT64>(options_.w) *
        static_cast<UINT64>(options_.c_in) * sizeof(float);
    const UINT64 weight_bytes = static_cast<UINT64>(options_.k) * static_cast<UINT64>(options_.k) *
        static_cast<UINT64>(options_.c_in) * static_cast<UINT64>(options_.c_out) * sizeof(float);
    auto upload_heap = heap_properties(D3D12_HEAP_TYPE_UPLOAD);
    auto default_heap = heap_properties(D3D12_HEAP_TYPE_DEFAULT);
    auto readback_heap = heap_properties(D3D12_HEAP_TYPE_READBACK);
    auto input_desc = buffer_desc(input_bytes);
    auto weight_desc = buffer_desc(weight_bytes);
    auto output_desc = buffer_desc(output_batch_capacity_bytes_, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    auto readback_desc = buffer_desc(output_batch_capacity_bytes_);

    throw_if_failed(
        device_->CreateCommittedResource(
            &upload_heap,
            D3D12_HEAP_FLAG_NONE,
            &input_desc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS(&input_buffer_)
        ),
        "failed to create input buffer"
    );
    throw_if_failed(
        device_->CreateCommittedResource(
            &upload_heap,
            D3D12_HEAP_FLAG_NONE,
            &weight_desc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS(&weight_buffer_)
        ),
        "failed to create weight buffer"
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
}

void Dx12Runner::upload_static_inputs() {
    const UINT64 input_bytes = static_cast<UINT64>(options_.h) * static_cast<UINT64>(options_.w) *
        static_cast<UINT64>(options_.c_in) * sizeof(float);
    const UINT64 weight_bytes = static_cast<UINT64>(options_.k) * static_cast<UINT64>(options_.k) *
        static_cast<UINT64>(options_.c_in) * static_cast<UINT64>(options_.c_out) * sizeof(float);
    const auto started = std::chrono::steady_clock::now();
    void* mapped_input = nullptr;
    throw_if_failed(input_buffer_->Map(0, nullptr, &mapped_input), "failed to map input buffer");
    read_binary_file_into_pointer(options_.input_path, mapped_input, static_cast<std::size_t>(input_bytes));
    input_buffer_->Unmap(0, nullptr);

    void* mapped_weight = nullptr;
    throw_if_failed(weight_buffer_->Map(0, nullptr, &mapped_weight), "failed to map weight buffer");
    read_binary_file_into_pointer(options_.weight_path, mapped_weight, static_cast<std::size_t>(weight_bytes));
    weight_buffer_->Unmap(0, nullptr);

    static_upload_wall_clock_latency_seconds_ =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
}

void Dx12Runner::wait_for_gpu() {
    const UINT64 signal_value = next_fence_value_++;
    throw_if_failed(queue_->Signal(fence_.Get(), signal_value), "failed to signal fence");
    if (fence_->GetCompletedValue() < signal_value) {
        throw_if_failed(
            fence_->SetEventOnCompletion(signal_value, fence_event_),
            "failed to wait for fence completion"
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

void Dx12Runner::dispatch_capture_batch(
    int thread_group_size,
    int tile_size,
    int oc_offset,
    int batch_c_out,
    std::vector<float>* full_output,
    double& checksum
) {
    const UINT64 total_outputs = spatial_size_ * static_cast<UINT64>(batch_c_out);
    const UINT dispatch_x = static_cast<UINT>(
        (total_outputs + static_cast<UINT64>(thread_group_size * tile_size) - 1ull) /
        static_cast<UINT64>(thread_group_size * tile_size)
    );
    const UINT64 batch_output_bytes = spatial_size_ * static_cast<UINT64>(batch_c_out) * sizeof(float);
    auto pipeline_state = pipeline_for_thread_group_size(thread_group_size);

    throw_if_failed(allocator_->Reset(), "failed to reset command allocator");
    throw_if_failed(command_list_->Reset(allocator_.Get(), pipeline_state.Get()), "failed to reset command list");

    command_list_->SetComputeRootSignature(root_signature_.Get());
    command_list_->SetComputeRootShaderResourceView(1, input_buffer_->GetGPUVirtualAddress());
    command_list_->SetComputeRootShaderResourceView(2, weight_buffer_->GetGPUVirtualAddress());
    command_list_->SetComputeRootUnorderedAccessView(3, output_buffer_->GetGPUVirtualAddress());

    const std::uint32_t root_constants[11] = {
        static_cast<std::uint32_t>(options_.h),
        static_cast<std::uint32_t>(options_.w),
        static_cast<std::uint32_t>(options_.c_in),
        static_cast<std::uint32_t>(options_.k),
        static_cast<std::uint32_t>(options_.pad),
        static_cast<std::uint32_t>(options_.stride),
        static_cast<std::uint32_t>(out_h_),
        static_cast<std::uint32_t>(out_w_),
        static_cast<std::uint32_t>(oc_offset),
        static_cast<std::uint32_t>(batch_c_out),
        static_cast<std::uint32_t>(tile_size),
    };
    command_list_->SetComputeRoot32BitConstants(0, 11, root_constants, 0);
    command_list_->Dispatch(dispatch_x, 1, 1);

    auto to_copy = transition_barrier(
        output_buffer_.Get(),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_COPY_SOURCE
    );
    command_list_->ResourceBarrier(1, &to_copy);
    command_list_->CopyBufferRegion(readback_buffer_.Get(), 0, output_buffer_.Get(), 0, batch_output_bytes);
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

    std::vector<float> batch_values(static_cast<size_t>(spatial_size_) * static_cast<size_t>(batch_c_out));
    void* mapped_readback = nullptr;
    D3D12_RANGE range = {0, static_cast<SIZE_T>(batch_output_bytes)};
    throw_if_failed(readback_buffer_->Map(0, &range, &mapped_readback), "failed to map readback buffer");
    std::memcpy(batch_values.data(), mapped_readback, static_cast<size_t>(batch_output_bytes));
    D3D12_RANGE empty_range = {0, 0};
    readback_buffer_->Unmap(0, &empty_range);

    for (UINT64 spatial_index = 0; spatial_index < spatial_size_; ++spatial_index) {
        const size_t batch_base = static_cast<size_t>(spatial_index) * static_cast<size_t>(batch_c_out);
        const size_t output_base = static_cast<size_t>(spatial_index) * static_cast<size_t>(options_.c_out) + static_cast<size_t>(oc_offset);
        for (int local_oc = 0; local_oc < batch_c_out; ++local_oc) {
            const float value = batch_values[batch_base + static_cast<size_t>(local_oc)];
            checksum += std::abs(static_cast<double>(value));
            if (full_output != nullptr) {
                (*full_output)[output_base + static_cast<size_t>(local_oc)] = value;
            }
        }
    }
}

PhaseMetrics Dx12Runner::run_configuration(
    int thread_group_size,
    int tile_size,
    int repeats,
    const std::string* output_path
) {
    if (repeats <= 0) {
        throw std::runtime_error("repeats must be positive");
    }
    if (tile_size <= 0 || tile_size > kMaxTileSize) {
        throw std::runtime_error("tile_size is out of range");
    }

    auto pipeline_state = pipeline_for_thread_group_size(thread_group_size);

    throw_if_failed(allocator_->Reset(), "failed to reset command allocator");
    throw_if_failed(command_list_->Reset(allocator_.Get(), pipeline_state.Get()), "failed to reset command list");

    command_list_->SetComputeRootSignature(root_signature_.Get());
    command_list_->SetComputeRootShaderResourceView(1, input_buffer_->GetGPUVirtualAddress());
    command_list_->SetComputeRootShaderResourceView(2, weight_buffer_->GetGPUVirtualAddress());
    command_list_->SetComputeRootUnorderedAccessView(3, output_buffer_->GetGPUVirtualAddress());

    int dispatches_per_repeat = 0;
    const auto started = std::chrono::steady_clock::now();
    for (int repeat = 0; repeat < repeats; ++repeat) {
        for (int oc_offset = 0; oc_offset < options_.c_out; oc_offset += output_batch_channels_) {
            const int batch_c_out = (std::min)(output_batch_channels_, options_.c_out - oc_offset);
            const UINT64 total_outputs = spatial_size_ * static_cast<UINT64>(batch_c_out);
            const UINT dispatch_x = static_cast<UINT>(
                (total_outputs + static_cast<UINT64>(thread_group_size * tile_size) - 1ull) /
                static_cast<UINT64>(thread_group_size * tile_size)
            );
            const std::uint32_t root_constants[11] = {
                static_cast<std::uint32_t>(options_.h),
                static_cast<std::uint32_t>(options_.w),
                static_cast<std::uint32_t>(options_.c_in),
                static_cast<std::uint32_t>(options_.k),
                static_cast<std::uint32_t>(options_.pad),
                static_cast<std::uint32_t>(options_.stride),
                static_cast<std::uint32_t>(out_h_),
                static_cast<std::uint32_t>(out_w_),
                static_cast<std::uint32_t>(oc_offset),
                static_cast<std::uint32_t>(batch_c_out),
                static_cast<std::uint32_t>(tile_size),
            };
            command_list_->SetComputeRoot32BitConstants(0, 11, root_constants, 0);
            command_list_->Dispatch(dispatch_x, 1, 1);
            dispatches_per_repeat += 1;
        }
    }
    throw_if_failed(command_list_->Close(), "failed to close command list");

    ID3D12CommandList* command_lists[] = {command_list_.Get()};
    queue_->ExecuteCommandLists(1, command_lists);
    wait_for_gpu();
    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();

    std::vector<float> full_output;
    if (output_path != nullptr && !output_path->empty()) {
        full_output.assign(static_cast<size_t>(spatial_size_) * static_cast<size_t>(options_.c_out), 0.0f);
    }
    double checksum_value = 0.0;
    for (int oc_offset = 0; oc_offset < options_.c_out; oc_offset += output_batch_channels_) {
        const int batch_c_out = (std::min)(output_batch_channels_, options_.c_out - oc_offset);
        dispatch_capture_batch(
            thread_group_size,
            tile_size,
            oc_offset,
            batch_c_out,
            full_output.empty() ? nullptr : &full_output,
            checksum_value
        );
    }

    if (output_path != nullptr && !output_path->empty()) {
        write_float32_file(*output_path, full_output);
    }

    const double flops_per_run =
        2.0 * static_cast<double>(out_h_) * static_cast<double>(out_w_) *
        static_cast<double>(options_.c_out) * static_cast<double>(options_.c_in) *
        static_cast<double>(options_.k) * static_cast<double>(options_.k);
    const long long checksum_rounded = static_cast<long long>(checksum_value);

    PhaseMetrics metrics;
    metrics.repeats = repeats;
    metrics.wall_clock_latency_seconds = elapsed / static_cast<double>(repeats);
    metrics.effective_gflops = flops_per_run / metrics.wall_clock_latency_seconds / 1'000'000'000.0;
    metrics.checksum = "chk_" + std::to_string(checksum_rounded);
    metrics.dispatches_per_repeat = dispatches_per_repeat / repeats;
    return metrics;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);
        const std::vector<int> thread_group_sizes = options.task_mode
            ? std::vector<int>{options.fixed_thread_group_size}
            : options.thread_group_sizes;
        emit_native_status(
            "method.spatial_convolution.backend.native_runner.process.start",
            {
                json_string_field("status", "running"),
                json_string_field("method", "spatial_convolution"),
                json_string_field("backend", "dx12"),
                json_string_field("mode", options.task_mode ? "task" : "benchmark"),
                json_int_field("h", options.h),
                json_int_field("w", options.w),
                json_int_field("cin", options.c_in),
                json_int_field("cout", options.c_out),
                json_int_field("k", options.k),
                json_int_field("pad", options.pad),
                json_int_field("stride", options.stride),
                json_int_field("autotune_repeats", options.autotune_repeats),
                json_int_field("measurement_repeats", options.measurement_repeats),
            }
        );
        emit_native_status(
            "method.spatial_convolution.backend.native_runner.setup.start",
            {
                json_string_field("status", "running"),
                json_string_field("method", "spatial_convolution"),
                json_string_field("backend", "dx12"),
            }
        );
        const auto setup_started = std::chrono::steady_clock::now();
        Dx12Runner runner(options, thread_group_sizes);
        const double setup_wall_clock_latency_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - setup_started).count();
        emit_native_status(
            "method.spatial_convolution.backend.native_runner.setup.complete",
            {
                json_string_field("status", "running"),
                json_string_field("method", "spatial_convolution"),
                json_string_field("backend", "dx12"),
                json_string_field("device_name", runner.device_name()),
                json_string_field("adapter_kind", runner.adapter_kind()),
                json_number_field("setup_wall_clock_latency_seconds", setup_wall_clock_latency_seconds),
            }
        );

        if (options.task_mode) {
            emit_native_status(
                "method.spatial_convolution.backend.native_runner.task.measurement.start",
                {
                    json_string_field("status", "running"),
                    json_string_field("method", "spatial_convolution"),
                    json_string_field("backend", "dx12"),
                    json_int_field("thread_group_size", options.fixed_thread_group_size),
                    json_int_field("tile_size", options.fixed_tile_size),
                    json_int_field("repeats", options.measurement_repeats),
                }
            );
            const PhaseMetrics metrics = runner.run_configuration(
                options.fixed_thread_group_size,
                options.fixed_tile_size,
                options.measurement_repeats,
                &options.output_path
            );
            emit_native_status(
                "method.spatial_convolution.backend.native_runner.task.measurement.complete",
                {
                    json_string_field("status", "running"),
                    json_string_field("method", "spatial_convolution"),
                    json_string_field("backend", "dx12"),
                    json_int_field("thread_group_size", options.fixed_thread_group_size),
                    json_int_field("tile_size", options.fixed_tile_size),
                    json_number_field("wall_clock_latency_seconds", metrics.wall_clock_latency_seconds),
                    json_number_field("effective_gflops", metrics.effective_gflops),
                    json_string_field("checksum", metrics.checksum),
                }
            );
            emit_native_status(
                "method.spatial_convolution.backend.native_runner.process.complete",
                {
                    json_string_field("status", "completed"),
                    json_string_field("method", "spatial_convolution"),
                    json_string_field("backend", "dx12"),
                    json_string_field("mode", "task"),
                }
            );

            std::cout << "{"
                      << "\"mode\":\"task\","
                      << "\"backend\":\"dx12\","
                      << "\"device_name\":\"" << runner.device_name() << "\","
                      << "\"adapter_kind\":\"" << runner.adapter_kind() << "\","
                      << "\"thread_group_size\":" << options.fixed_thread_group_size << ","
                      << "\"tile_size\":" << options.fixed_tile_size << ","
                      << "\"iteration_count\":" << options.measurement_repeats << ","
                      << "\"accumulation_precision\":\"fp32\","
                      << "\"kernel_layout\":\"spatial_major_oc_tiles\","
                      << "\"static_input_heap\":\"upload\","
                      << "\"setup_wall_clock_latency_seconds\":" << std::fixed << std::setprecision(9)
                      << setup_wall_clock_latency_seconds << ","
                      << "\"static_upload_wall_clock_latency_seconds\":" << std::fixed << std::setprecision(9)
                      << runner.static_upload_wall_clock_latency_seconds() << ","
                      << "\"dispatches_per_repeat\":" << metrics.dispatches_per_repeat << ","
                      << "\"wall_clock_latency_seconds\":" << std::fixed << std::setprecision(9)
                      << metrics.wall_clock_latency_seconds << ","
                      << "\"effective_gflops\":" << std::fixed << std::setprecision(6)
                      << metrics.effective_gflops << ","
                      << "\"checksum\":\"" << metrics.checksum << "\""
                      << "}\n";
            return 0;
        }

        TrialMetrics best_metrics;
        bool has_best_metrics = false;
        int trials_run = 0;
        const int total_trials = static_cast<int>(options.thread_group_sizes.size() * options.tile_sizes.size());
        for (const int thread_group_size : options.thread_group_sizes) {
            for (const int tile_size : options.tile_sizes) {
                const int trial_index = trials_run + 1;
                emit_native_status(
                    "method.spatial_convolution.backend.native_runner.autotune.trial.start",
                    {
                        json_string_field("status", "running"),
                        json_string_field("method", "spatial_convolution"),
                        json_string_field("backend", "dx12"),
                        json_int_field("trial_index", trial_index),
                        json_int_field("total_trials", total_trials),
                        json_int_field("thread_group_size", thread_group_size),
                        json_int_field("tile_size", tile_size),
                        json_int_field("repeats", options.autotune_repeats),
                    }
                );
                TrialMetrics candidate;
                candidate.thread_group_size = thread_group_size;
                candidate.tile_size = tile_size;
                candidate.autotune = runner.run_configuration(thread_group_size, tile_size, options.autotune_repeats);
                ++trials_run;
                const bool became_best =
                    !has_best_metrics ||
                    candidate.autotune.wall_clock_latency_seconds < best_metrics.autotune.wall_clock_latency_seconds;
                if (became_best) {
                    best_metrics = candidate;
                    has_best_metrics = true;
                }
                emit_native_status(
                    "method.spatial_convolution.backend.native_runner.autotune.trial.complete",
                    {
                        json_string_field("status", "running"),
                        json_string_field("method", "spatial_convolution"),
                        json_string_field("backend", "dx12"),
                        json_int_field("trial_index", trials_run),
                        json_int_field("total_trials", total_trials),
                        json_int_field("thread_group_size", thread_group_size),
                        json_int_field("tile_size", tile_size),
                        json_number_field("wall_clock_latency_seconds", candidate.autotune.wall_clock_latency_seconds),
                        json_number_field("effective_gflops", candidate.autotune.effective_gflops),
                        json_string_field("checksum", candidate.autotune.checksum),
                        json_int_field("dispatches_per_repeat", candidate.autotune.dispatches_per_repeat),
                        json_bool_field("best_so_far", became_best),
                    }
                );
            }
        }

        if (!has_best_metrics) {
            throw std::runtime_error("DX12 benchmark ran zero valid trials");
        }

        emit_native_status(
            "method.spatial_convolution.backend.native_runner.autotune.best_selected",
            {
                json_string_field("status", "running"),
                json_string_field("method", "spatial_convolution"),
                json_string_field("backend", "dx12"),
                json_int_field("thread_group_size", best_metrics.thread_group_size),
                json_int_field("tile_size", best_metrics.tile_size),
                json_number_field("wall_clock_latency_seconds", best_metrics.autotune.wall_clock_latency_seconds),
                json_number_field("effective_gflops", best_metrics.autotune.effective_gflops),
                json_string_field("checksum", best_metrics.autotune.checksum),
            }
        );

        const std::string* output_path = options.output_path.empty() ? nullptr : &options.output_path;
        emit_native_status(
            "method.spatial_convolution.backend.native_runner.measurement.start",
            {
                json_string_field("status", "running"),
                json_string_field("method", "spatial_convolution"),
                json_string_field("backend", "dx12"),
                json_int_field("thread_group_size", best_metrics.thread_group_size),
                json_int_field("tile_size", best_metrics.tile_size),
                json_int_field("repeats", options.measurement_repeats),
            }
        );
        best_metrics.measurement = runner.run_configuration(
            best_metrics.thread_group_size,
            best_metrics.tile_size,
            options.measurement_repeats,
            output_path
        );
        emit_native_status(
            "method.spatial_convolution.backend.native_runner.measurement.complete",
            {
                json_string_field("status", "running"),
                json_string_field("method", "spatial_convolution"),
                json_string_field("backend", "dx12"),
                json_int_field("thread_group_size", best_metrics.thread_group_size),
                json_int_field("tile_size", best_metrics.tile_size),
                json_number_field("wall_clock_latency_seconds", best_metrics.measurement.wall_clock_latency_seconds),
                json_number_field("effective_gflops", best_metrics.measurement.effective_gflops),
                json_string_field("checksum", best_metrics.measurement.checksum),
                json_int_field("dispatches_per_repeat", best_metrics.measurement.dispatches_per_repeat),
            }
        );
        emit_native_status(
            "method.spatial_convolution.backend.native_runner.process.complete",
            {
                json_string_field("status", "completed"),
                json_string_field("method", "spatial_convolution"),
                json_string_field("backend", "dx12"),
                json_string_field("mode", "benchmark"),
                json_int_field("trials_run", trials_run),
            }
        );

        std::cout << "{"
                  << "\"mode\":\"benchmark\","
                  << "\"backend\":\"dx12\","
                  << "\"device_name\":\"" << runner.device_name() << "\","
                  << "\"adapter_kind\":\"" << runner.adapter_kind() << "\","
                  << "\"thread_group_size\":" << best_metrics.thread_group_size << ","
                  << "\"tile_size\":" << best_metrics.tile_size << ","
                  << "\"autotune_repeats\":" << options.autotune_repeats << ","
                  << "\"measurement_repeats\":" << options.measurement_repeats << ","
                  << "\"trials_run\":" << trials_run << ","
                  << "\"accumulation_precision\":\"fp32\","
                  << "\"kernel_layout\":\"spatial_major_oc_tiles\","
                  << "\"static_input_heap\":\"upload\","
                  << "\"setup_wall_clock_latency_seconds\":" << std::fixed << std::setprecision(9)
                  << setup_wall_clock_latency_seconds << ","
                  << "\"static_upload_wall_clock_latency_seconds\":" << std::fixed << std::setprecision(9)
                  << runner.static_upload_wall_clock_latency_seconds() << ","
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
        emit_native_status(
            "method.spatial_convolution.backend.native_runner.process.error",
            {
                json_string_field("status", "failed"),
                json_string_field("method", "spatial_convolution"),
                json_string_field("backend", "dx12"),
                json_string_field("error", exc.what()),
            }
        );
        std::cerr << exc.what() << '\n';
        return 1;
    }
}
