/*
 * CUDA Conv2D Benchmark Runner
 *
 * The benchmark previously fell back to a purely direct kernel after an older
 * tiled path returned invalid all-zero outputs. This version keeps the safer
 * validation logic, but reintroduces spatial tiling in a narrower form: each
 * block stages an input footprint into shared memory and then reuses it across
 * neighboring output pixels and output-channel lanes.
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;

#define CUDA_CHECK(call)                                                                     \
    do {                                                                                     \
        cudaError_t err__ = (call);                                                          \
        if (err__ != cudaSuccess) {                                                          \
            cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " code=" << err__   \
                 << " \"" << cudaGetErrorString(err__) << "\"" << endl;                    \
            return EXIT_FAILURE;                                                             \
        }                                                                                    \
    } while (0)

struct TileConfig {
    int tile_w;
    int tile_h;
    int shared_input;
    float milliseconds;
};

constexpr int kOutputsPerThread = 4;

vector<int> parse_list(const string& text) {
    vector<int> values;
    stringstream stream(text);
    string item;
    while (getline(stream, item, ',')) {
        if (!item.empty()) {
            values.push_back(stoi(item));
        }
    }
    return values;
}

vector<float> load_binary(const string& path, size_t expected_values) {
    ifstream file(path, ios::binary);
    if (!file) {
        throw runtime_error("failed to open input file: " + path);
    }

    const size_t expected_bytes = expected_values * sizeof(float);
    file.seekg(0, ios::end);
    const size_t actual_bytes = static_cast<size_t>(file.tellg());
    file.seekg(0, ios::beg);
    if (actual_bytes != expected_bytes) {
        throw runtime_error(
            "binary size mismatch for " + path + ": expected " + to_string(expected_bytes) +
            " bytes, got " + to_string(actual_bytes)
        );
    }

    vector<float> data(expected_values);
    file.read(reinterpret_cast<char*>(data.data()), static_cast<streamsize>(expected_bytes));
    if (!file) {
        throw runtime_error("failed to read full binary payload from " + path);
    }
    return data;
}

vector<float> transpose_weight_for_oc_vectorization(
    const vector<float>& weight_src,
    int c_out,
    int k,
    int c_in
) {
    vector<float> weight_dst(weight_src.size());
    for (int oc = 0; oc < c_out; ++oc) {
        for (int kh = 0; kh < k; ++kh) {
            for (int kw = 0; kw < k; ++kw) {
                for (int ic = 0; ic < c_in; ++ic) {
                    const size_t src_index = static_cast<size_t>(((oc * k + kh) * k + kw) * c_in + ic);
                    const size_t dst_index = static_cast<size_t>(((kh * k + kw) * c_in + ic) * c_out + oc);
                    weight_dst[dst_index] = weight_src[src_index];
                }
            }
        }
    }
    return weight_dst;
}

vector<TileConfig> build_candidate_tiles(const vector<int>& block_sizes, const vector<int>& tile_sizes) {
    vector<TileConfig> defaults = {
        {8, 8, 0, 0.0f},
        {8, 8, 1, 0.0f},
        {16, 8, 0, 0.0f},
        {16, 8, 1, 0.0f},
        {8, 16, 0, 0.0f},
        {8, 16, 1, 0.0f},
        {16, 16, 0, 0.0f},
        {16, 16, 1, 0.0f},
        {32, 8, 0, 0.0f},
        {32, 8, 1, 0.0f},
        {32, 16, 0, 0.0f},
        {32, 16, 1, 0.0f},
        {16, 32, 0, 0.0f},
        {16, 32, 1, 0.0f},
    };

    if (block_sizes.empty()) {
        return defaults;
    }

    vector<TileConfig> filtered;
    for (const TileConfig& candidate : defaults) {
        const int threads = candidate.tile_w * candidate.tile_h;
        const bool block_matches = block_sizes.empty() ||
            find(block_sizes.begin(), block_sizes.end(), threads) != block_sizes.end();
        const bool tile_matches = tile_sizes.empty() ||
            find(tile_sizes.begin(), tile_sizes.end(), candidate.tile_w) != tile_sizes.end();
        if (block_matches && tile_matches) {
            filtered.push_back(candidate);
        }
    }

    if (filtered.empty()) {
        return defaults;
    }
    return filtered;
}

size_t input_tile_footprint_elements(const TileConfig& config, int k, int stride) {
    return static_cast<size_t>((config.tile_w - 1) * stride + k) *
           static_cast<size_t>((config.tile_h - 1) * stride + k);
}

int choose_input_channel_tile(
    const TileConfig& config,
    int k,
    int stride,
    int c_in,
    size_t shared_mem_limit_bytes
) {
    // Keep the tile within per-block shared-memory limits while still amortizing global reads.
    const size_t footprint_elements = input_tile_footprint_elements(config, k, stride);
    if (footprint_elements == 0) {
        return 0;
    }

    const size_t bytes_per_channel = footprint_elements * sizeof(float);
    if (bytes_per_channel > shared_mem_limit_bytes) {
        return 0;
    }

    int channels = static_cast<int>(shared_mem_limit_bytes / bytes_per_channel);
    channels = min(channels, c_in);
    if (channels >= 8) {
        return (channels / 8) * 8;
    }
    if (channels >= 4) {
        return (channels / 4) * 4;
    }
    return max(1, channels);
}

__device__ __forceinline__ void accumulate_output_lanes(
    float input_value,
    const float* weight_ptr,
    int valid_outputs,
    float sums[kOutputsPerThread]
) {
    if (valid_outputs == kOutputsPerThread) {
        const float4 weight_vec = *reinterpret_cast<const float4*>(weight_ptr);
        sums[0] += input_value * weight_vec.x;
        sums[1] += input_value * weight_vec.y;
        sums[2] += input_value * weight_vec.z;
        sums[3] += input_value * weight_vec.w;
        return;
    }

    #pragma unroll
    for (int lane = 0; lane < kOutputsPerThread; ++lane) {
        if (lane < valid_outputs) {
            sums[lane] += input_value * weight_ptr[lane];
        }
    }
}

__device__ __forceinline__ void store_output_lanes(
    float* output_ptr,
    int valid_outputs,
    const float sums[kOutputsPerThread]
) {
    if (valid_outputs == kOutputsPerThread) {
        *reinterpret_cast<float4*>(output_ptr) = make_float4(sums[0], sums[1], sums[2], sums[3]);
        return;
    }

    #pragma unroll
    for (int lane = 0; lane < kOutputsPerThread; ++lane) {
        if (lane < valid_outputs) {
            output_ptr[lane] = sums[lane];
        }
    }
}

__global__ void conv2d_input_tiled(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output_batch,
    int h,
    int w,
    int c_in,
    int c_out,
    int k,
    int pad,
    int stride,
    int oc_offset,
    int batch_c_out,
    int out_h,
    int out_w,
    int input_channels_per_tile
) {
    extern __shared__ float shared_input[];

    const int ow = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int oh = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    const int tile_ow_start = static_cast<int>(blockIdx.x * blockDim.x * stride);
    const int tile_oh_start = static_cast<int>(blockIdx.y * blockDim.y * stride);
    const int local_oc_base = static_cast<int>(blockIdx.z) * kOutputsPerThread;
    const int valid_outputs = min(kOutputsPerThread, batch_c_out - local_oc_base);
    const int shared_w = (static_cast<int>(blockDim.x) - 1) * stride + k;
    const int shared_h = (static_cast<int>(blockDim.y) - 1) * stride + k;
    const size_t shared_spatial_size = static_cast<size_t>(shared_w) * static_cast<size_t>(shared_h);
    const int thread_linear = static_cast<int>(threadIdx.y * blockDim.x + threadIdx.x);
    const int thread_count = static_cast<int>(blockDim.x * blockDim.y);

    if (local_oc_base >= batch_c_out) {
        return;
    }

    float sums[kOutputsPerThread] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int ic_base = 0; ic_base < c_in; ic_base += input_channels_per_tile) {
        const int chunk_channels = min(input_channels_per_tile, c_in - ic_base);
        const size_t chunk_elements = shared_spatial_size * static_cast<size_t>(chunk_channels);

        // Layout the shared tile as [spatial][channel] so NHWC input loads stay contiguous.
        for (size_t linear_index = static_cast<size_t>(thread_linear);
             linear_index < chunk_elements;
             linear_index += static_cast<size_t>(thread_count)) {
            const size_t spatial_index = linear_index / static_cast<size_t>(chunk_channels);
            const int local_ic = static_cast<int>(linear_index % static_cast<size_t>(chunk_channels));
            const int local_x = static_cast<int>(spatial_index % static_cast<size_t>(shared_w));
            const int local_y = static_cast<int>(spatial_index / static_cast<size_t>(shared_w));
            const int ih = tile_oh_start - pad + local_y;
            const int iw = tile_ow_start - pad + local_x;

            float input_value = 0.0f;
            if (ih >= 0 && ih < h && iw >= 0 && iw < w) {
                const size_t input_index =
                    (static_cast<size_t>(ih) * static_cast<size_t>(w) + static_cast<size_t>(iw)) *
                    static_cast<size_t>(c_in) +
                    static_cast<size_t>(ic_base + local_ic);
                input_value = input[input_index];
            }
            shared_input[linear_index] = input_value;
        }
        __syncthreads();

        if (ow < out_w && oh < out_h) {
            for (int kh = 0; kh < k; ++kh) {
                const int shared_y = static_cast<int>(threadIdx.y) * stride + kh;
                for (int kw = 0; kw < k; ++kw) {
                    const int shared_x = static_cast<int>(threadIdx.x) * stride + kw;
                    const size_t shared_base =
                        (static_cast<size_t>(shared_y) * static_cast<size_t>(shared_w) + static_cast<size_t>(shared_x)) *
                        static_cast<size_t>(chunk_channels);

                    for (int local_ic = 0; local_ic < chunk_channels; ++local_ic) {
                        const float input_value = shared_input[shared_base + static_cast<size_t>(local_ic)];
                        const size_t weight_base =
                            static_cast<size_t>(((kh * k + kw) * c_in + ic_base + local_ic) * c_out + oc_offset + local_oc_base);
                        accumulate_output_lanes(input_value, weight + weight_base, valid_outputs, sums);
                    }
                }
            }
        }
        __syncthreads();
    }

    if (ow < out_w && oh < out_h) {
        const size_t output_base =
            static_cast<size_t>(oh * out_w + ow) * static_cast<size_t>(batch_c_out) + static_cast<size_t>(local_oc_base);
        store_output_lanes(output_batch + output_base, valid_outputs, sums);
    }
}

__global__ void conv2d_direct(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output_batch,
    int h,
    int w,
    int c_in,
    int c_out,
    int k,
    int pad,
    int stride,
    int oc_offset,
    int batch_c_out,
    int out_h,
    int out_w
) {
    const int ow = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int oh = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    const int local_oc_base = static_cast<int>(blockIdx.z) * kOutputsPerThread;
    const int valid_outputs = min(kOutputsPerThread, batch_c_out - local_oc_base);

    if (ow >= out_w || oh >= out_h || local_oc_base >= batch_c_out) {
        return;
    }

    float sums[kOutputsPerThread] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int kh = 0; kh < k; ++kh) {
        const int ih = oh * stride - pad + kh;
        if (ih < 0 || ih >= h) {
            continue;
        }

        for (int kw = 0; kw < k; ++kw) {
            const int iw = ow * stride - pad + kw;
            if (iw < 0 || iw >= w) {
                continue;
            }

            const int input_base = (ih * w + iw) * c_in;
            for (int ic = 0; ic < c_in; ++ic) {
                const float input_value = input[input_base + ic];
                const size_t weight_base = static_cast<size_t>(((kh * k + kw) * c_in + ic) * c_out + oc_offset + local_oc_base);
                accumulate_output_lanes(input_value, weight + weight_base, valid_outputs, sums);
            }
        }
    }

    const size_t output_base = static_cast<size_t>(oh * out_w + ow) * static_cast<size_t>(batch_c_out) + static_cast<size_t>(local_oc_base);
    store_output_lanes(output_batch + output_base, valid_outputs, sums);
}

void launch_kernel(
    const TileConfig& config,
    const float* d_input,
    const float* d_weight,
    float* d_output_batch,
    int h,
    int w,
    int c_in,
    int c_out,
    int k,
    int pad,
    int stride,
    int oc_offset,
    int batch_c_out,
    int out_h,
    int out_w,
    size_t shared_mem_limit_bytes
) {
    const dim3 block(config.tile_w, config.tile_h);
    const dim3 grid(
        static_cast<unsigned int>((out_w + config.tile_w - 1) / config.tile_w),
        static_cast<unsigned int>((out_h + config.tile_h - 1) / config.tile_h),
        static_cast<unsigned int>((batch_c_out + kOutputsPerThread - 1) / kOutputsPerThread)
    );
    if (config.shared_input == 0) {
        conv2d_direct<<<grid, block>>>(d_input, d_weight, d_output_batch, h, w, c_in, c_out, k, pad, stride, oc_offset, batch_c_out, out_h, out_w);
        return;
    }

    const int input_channels_per_tile = choose_input_channel_tile(config, k, stride, c_in, shared_mem_limit_bytes);
    if (input_channels_per_tile <= 0) {
        conv2d_direct<<<grid, block>>>(d_input, d_weight, d_output_batch, h, w, c_in, c_out, k, pad, stride, oc_offset, batch_c_out, out_h, out_w);
        return;
    }

    const size_t shared_mem_bytes =
        input_tile_footprint_elements(config, k, stride) * static_cast<size_t>(input_channels_per_tile) * sizeof(float);
    conv2d_input_tiled<<<grid, block, shared_mem_bytes>>>(
        d_input,
        d_weight,
        d_output_batch,
        h,
        w,
        c_in,
        c_out,
        k,
        pad,
        stride,
        oc_offset,
        batch_c_out,
        out_h,
        out_w,
        input_channels_per_tile
    );
}

double checksum_abs_sum(const vector<float>& values) {
    double sum = 0.0;
    for (float value : values) {
        sum += abs(static_cast<double>(value));
    }
    return sum;
}

int choose_output_channel_batch(int c_out) {
    return min(c_out, 32);
}

int main(int argc, char** argv) {
    try {
        string input_path;
        string weight_path;
        string output_path;
        int h = 0;
        int w = 0;
        int c_in = 0;
        int c_out = 0;
        int k = 0;
        int pad = 0;
        int stride = 1;
        int autotune_repeats = 1;
        int measurement_repeats = 1;
        vector<int> block_sizes;
        vector<int> tile_sizes;

        for (int i = 1; i < argc; ++i) {
            const string arg = argv[i];
            if (arg == "--input" && i + 1 < argc) input_path = argv[++i];
            else if (arg == "--weight" && i + 1 < argc) weight_path = argv[++i];
            else if (arg == "--output" && i + 1 < argc) output_path = argv[++i];
            else if (arg == "--h" && i + 1 < argc) h = stoi(argv[++i]);
            else if (arg == "--w" && i + 1 < argc) w = stoi(argv[++i]);
            else if (arg == "--cin" && i + 1 < argc) c_in = stoi(argv[++i]);
            else if (arg == "--cout" && i + 1 < argc) c_out = stoi(argv[++i]);
            else if (arg == "--k" && i + 1 < argc) k = stoi(argv[++i]);
            else if (arg == "--pad" && i + 1 < argc) pad = stoi(argv[++i]);
            else if (arg == "--stride" && i + 1 < argc) stride = stoi(argv[++i]);
            else if (arg == "--block-sizes" && i + 1 < argc) block_sizes = parse_list(argv[++i]);
            else if (arg == "--tile-sizes" && i + 1 < argc) tile_sizes = parse_list(argv[++i]);
            else if (arg == "--transpose-modes" && i + 1 < argc) ++i;
            else if (arg == "--autotune-repeats" && i + 1 < argc) autotune_repeats = stoi(argv[++i]);
            else if (arg == "--measurement-repeats" && i + 1 < argc) measurement_repeats = stoi(argv[++i]);
        }

        if (input_path.empty() || weight_path.empty()) {
            throw runtime_error("both --input and --weight are required");
        }
        if (h <= 0 || w <= 0 || c_in <= 0 || c_out <= 0 || k <= 0 || stride <= 0) {
            throw runtime_error("invalid convolution dimensions");
        }
        if (autotune_repeats <= 0 || measurement_repeats <= 0) {
            throw runtime_error("repeat counts must be positive");
        }

        const int out_h = (h + 2 * pad - k) / stride + 1;
        const int out_w = (w + 2 * pad - k) / stride + 1;
        if (out_h <= 0 || out_w <= 0) {
            throw runtime_error("invalid output shape after applying padding/kernel size");
        }

        int device_index = 0;
        CUDA_CHECK(cudaGetDevice(&device_index));
        cudaDeviceProp device_properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&device_properties, device_index));
        const size_t shared_mem_limit_bytes = static_cast<size_t>(device_properties.sharedMemPerBlock);

        const size_t input_size = static_cast<size_t>(h) * static_cast<size_t>(w) * static_cast<size_t>(c_in);
        const size_t weight_size = static_cast<size_t>(k) * static_cast<size_t>(k) * static_cast<size_t>(c_in) * static_cast<size_t>(c_out);
        const size_t spatial_size = static_cast<size_t>(out_h) * static_cast<size_t>(out_w);
        const size_t output_size = spatial_size * static_cast<size_t>(c_out);
        const int channel_batch = choose_output_channel_batch(c_out);
        const size_t batch_output_capacity = spatial_size * static_cast<size_t>(channel_batch);

        const vector<float> h_input = load_binary(input_path, input_size);
        const vector<float> h_weight_raw = load_binary(weight_path, weight_size);
        const vector<float> h_weight = transpose_weight_for_oc_vectorization(h_weight_raw, c_out, k, c_in);
        vector<float> h_output;
        if (!output_path.empty()) {
            h_output.assign(output_size, 0.0f);
        }
        vector<float> h_output_batch(batch_output_capacity, 0.0f);

        float* d_input = nullptr;
        float* d_weight = nullptr;
        float* d_output_batch = nullptr;
        CUDA_CHECK(cudaMalloc(&d_input, input_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_weight, weight_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output_batch, batch_output_capacity * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), input_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_weight, h_weight.data(), weight_size * sizeof(float), cudaMemcpyHostToDevice));

        cudaEvent_t ev_start = nullptr;
        cudaEvent_t ev_stop = nullptr;
        CUDA_CHECK(cudaEventCreate(&ev_start));
        CUDA_CHECK(cudaEventCreate(&ev_stop));

        vector<TileConfig> candidates = build_candidate_tiles(block_sizes, tile_sizes);
        TileConfig best_config = candidates.front();
        float best_autotune_ms = FLT_MAX;

        for (TileConfig& candidate : candidates) {
            for (int oc_offset = 0; oc_offset < c_out; oc_offset += channel_batch) {
                const int launch_c_out = min(channel_batch, c_out - oc_offset);
                launch_kernel(candidate, d_input, d_weight, d_output_batch, h, w, c_in, c_out, k, pad, stride, oc_offset, launch_c_out, out_h, out_w, shared_mem_limit_bytes);
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaGetLastError());

            CUDA_CHECK(cudaEventRecord(ev_start));
            for (int repeat = 0; repeat < autotune_repeats; ++repeat) {
                for (int oc_offset = 0; oc_offset < c_out; oc_offset += channel_batch) {
                    const int launch_c_out = min(channel_batch, c_out - oc_offset);
                    launch_kernel(candidate, d_input, d_weight, d_output_batch, h, w, c_in, c_out, k, pad, stride, oc_offset, launch_c_out, out_h, out_w, shared_mem_limit_bytes);
                }
            }
            CUDA_CHECK(cudaEventRecord(ev_stop));
            CUDA_CHECK(cudaEventSynchronize(ev_stop));
            CUDA_CHECK(cudaGetLastError());

            float elapsed_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop));
            candidate.milliseconds = elapsed_ms / static_cast<float>(autotune_repeats);
            if (candidate.milliseconds < best_autotune_ms) {
                best_autotune_ms = candidate.milliseconds;
                best_config = candidate;
            }
        }

        CUDA_CHECK(cudaEventRecord(ev_start));
        for (int repeat = 0; repeat < measurement_repeats; ++repeat) {
            for (int oc_offset = 0; oc_offset < c_out; oc_offset += channel_batch) {
                const int launch_c_out = min(channel_batch, c_out - oc_offset);
                launch_kernel(best_config, d_input, d_weight, d_output_batch, h, w, c_in, c_out, k, pad, stride, oc_offset, launch_c_out, out_h, out_w, shared_mem_limit_bytes);
            }
        }
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        CUDA_CHECK(cudaGetLastError());

        float measurement_ms_total = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&measurement_ms_total, ev_start, ev_stop));
        const double measurement_seconds = (measurement_ms_total / 1000.0) / static_cast<double>(measurement_repeats);
        const double autotune_seconds = static_cast<double>(best_autotune_ms) / 1000.0;

        double checksum_value = 0.0;
        for (int oc_offset = 0; oc_offset < c_out; oc_offset += channel_batch) {
            const int launch_c_out = min(channel_batch, c_out - oc_offset);
            const size_t batch_output_size = spatial_size * static_cast<size_t>(launch_c_out);

            launch_kernel(best_config, d_input, d_weight, d_output_batch, h, w, c_in, c_out, k, pad, stride, oc_offset, launch_c_out, out_h, out_w, shared_mem_limit_bytes);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(
                h_output_batch.data(),
                d_output_batch,
                batch_output_size * sizeof(float),
                cudaMemcpyDeviceToHost
            ));

            for (size_t spatial_index = 0; spatial_index < spatial_size; ++spatial_index) {
                const size_t batch_base = spatial_index * static_cast<size_t>(launch_c_out);
                const size_t output_base = spatial_index * static_cast<size_t>(c_out) + static_cast<size_t>(oc_offset);
                for (int local_oc = 0; local_oc < launch_c_out; ++local_oc) {
                    const float value = h_output_batch[batch_base + static_cast<size_t>(local_oc)];
                    checksum_value += abs(static_cast<double>(value));
                    if (!output_path.empty()) {
                        h_output[output_base + static_cast<size_t>(local_oc)] = value;
                    }
                }
            }
        }

        if (!output_path.empty()) {
            ofstream output_stream(output_path, ios::binary);
            if (!output_stream) {
                throw runtime_error("failed to open output file: " + output_path);
            }
            output_stream.write(reinterpret_cast<const char*>(h_output.data()), static_cast<streamsize>(h_output.size() * sizeof(float)));
            if (!output_stream) {
                throw runtime_error("failed to write output file: " + output_path);
            }
        }

        const double flops_per_run = 2.0 * static_cast<double>(out_h) * static_cast<double>(out_w) *
                                     static_cast<double>(c_out) * static_cast<double>(c_in) *
                                     static_cast<double>(k) * static_cast<double>(k);
        const long long checksum_rounded = static_cast<long long>(checksum_value);

        cout << "{\n"
             << "  \"device_name\": \"" << device_properties.name << "\",\n"
             << "  \"transpose\": 0,\n"
             << "  \"shared_input\": " << best_config.shared_input << ",\n"
             << "  \"block_size\": " << (best_config.tile_w * best_config.tile_h) << ",\n"
             << "  \"tile_size\": " << best_config.tile_w << ",\n"
             << "  \"autotune_repeats\": " << autotune_repeats << ",\n"
             << "  \"measurement_repeats\": " << measurement_repeats << ",\n"
             << "  \"trials_run\": " << candidates.size() << ",\n"
             << "  \"autotune_wall_clock_latency_seconds\": " << fixed << setprecision(9) << autotune_seconds << ",\n"
             << "  \"autotune_effective_gflops\": " << (flops_per_run / autotune_seconds / 1e9) << ",\n"
             << "  \"autotune_checksum\": \"chk_" << checksum_rounded << "\",\n"
             << "  \"measurement_wall_clock_latency_seconds\": " << fixed << setprecision(9) << measurement_seconds << ",\n"
             << "  \"measurement_effective_gflops\": " << (flops_per_run / measurement_seconds / 1e9) << ",\n"
             << "  \"measurement_checksum\": \"chk_" << checksum_rounded << "\"\n"
             << "}\n";

        CUDA_CHECK(cudaEventDestroy(ev_start));
        CUDA_CHECK(cudaEventDestroy(ev_stop));
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_weight));
        CUDA_CHECK(cudaFree(d_output_batch));
        return EXIT_SUCCESS;
    } catch (const exception& exc) {
        cerr << exc.what() << endl;
        return EXIT_FAILURE;
    }
}
