#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Options {
    std::string matrix_path;
    std::string vector_path;
    std::string output_path;
    int rows = 0;
    int cols = 0;
    int row_start = 0;
    int row_end = 0;
    std::vector<int> transpose_modes;
    std::vector<int> block_sizes;
    std::vector<int> tile_sizes;
    int fixed_transpose = -1;
    int fixed_block_size = 0;
    int fixed_tile_size = 0;
    int autotune_repeats = 1;
    int measurement_repeats = 1;
    std::string accumulation_precision = "fp32";
    bool task_mode = false;
    bool verbose = false;
};

struct PhaseMetrics {
    int repeats = 0;
    double wall_clock_latency_seconds = std::numeric_limits<double>::infinity();
    double effective_gflops = 0.0;
    double device_to_host_seconds = 0.0;  // single post-kernel D2H memcpy
    std::string checksum;
};

// One per-trial record for the raw_report.trials array. Schema mirrors the
// conv2d CPU/GPU layout so downstream aggregators share a common shape.
struct TrialRecord {
    std::string phase;                  // "autotune" or "measurement"
    int candidate_index = 0;
    int candidate_total = 0;
    int trial_index_within_candidate = 0;
    int repeats_for_candidate = 0;
    int transpose = 0;
    int block_size = 0;
    int tile_size = 0;
    double host_prep_seconds = 0.0;       // 0 per-trial (H2D is upfront)
    double host_compute_seconds = 0.0;    // averaged kernel time (cudaEvent)
    double device_to_host_seconds = 0.0;  // single D2H memcpy time (chrono)
    double host_postproc_seconds = 0.0;   // 0 (checksum not timed)
    double total_wall_seconds = 0.0;      // host_compute + device_to_host
    std::string checksum;
};

struct TrialMetrics {
    int transpose = 0;
    int block_size = 0;
    int tile_size = 0;
    PhaseMetrics autotune;
    PhaseMetrics measurement;
};

bool is_supported_accumulation_precision(const std::string& value) {
    return value == "fp32" || value == "fp64_accumulate";
}

inline void cuda_check(cudaError_t status, const char* message) {
    if (status != cudaSuccess) {
        std::ostringstream builder;
        builder << message << ": " << cudaGetErrorString(status);
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

Options parse_args(int argc, char** argv) {
    Options options;
    int index = 1;
    while (index < argc) {
        const std::string key = argv[index];
        if (key == "--verbose") {
            options.verbose = true;
            index += 1;
            continue;
        }

        if (index + 1 >= argc) {
            throw std::runtime_error("missing value for command line flag");
        }
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
        } else if (key == "--transpose-modes") {
            options.transpose_modes = parse_int_list(value);
        } else if (key == "--block-sizes") {
            options.block_sizes = parse_int_list(value);
        } else if (key == "--tile-sizes") {
            options.tile_sizes = parse_int_list(value);
        } else if (key == "--fixed-transpose") {
            options.fixed_transpose = std::stoi(value);
        } else if (key == "--fixed-block-size") {
            options.fixed_block_size = std::stoi(value);
        } else if (key == "--fixed-tile-size") {
            options.fixed_tile_size = std::stoi(value);
        } else if (key == "--autotune-repeats") {
            options.autotune_repeats = std::stoi(value);
        } else if (key == "--measurement-repeats" || key == "--iteration-count") {
            // Task execution reuses the measurement loop but exposes the more
            // domain-specific name iteration-count to the runtime layer.
            options.measurement_repeats = std::stoi(value);
        } else if (key == "--accumulation-precision") {
            options.accumulation_precision = value;
        } else {
            throw std::runtime_error("unknown flag: " + key);
        }
        index += 2;
    }

    if (options.matrix_path.empty() || options.vector_path.empty()) {
        throw std::runtime_error("matrix and vector paths are required");
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

    const bool has_fixed_transpose = options.fixed_transpose >= 0;
    const bool has_fixed_block_size = options.fixed_block_size > 0;
    const bool has_fixed_tile_size = options.fixed_tile_size > 0;
    if ((has_fixed_transpose || has_fixed_block_size || has_fixed_tile_size) &&
        !(has_fixed_transpose && has_fixed_block_size && has_fixed_tile_size)) {
        throw std::runtime_error(
            "task mode requires fixed-transpose, fixed-block-size, and fixed-tile-size together"
        );
    }
    options.task_mode = has_fixed_transpose && has_fixed_block_size && has_fixed_tile_size;

    if (!options.task_mode) {
        if (options.transpose_modes.empty() || options.block_sizes.empty() || options.tile_sizes.empty()) {
            throw std::runtime_error("transpose/block/tile candidate lists are required");
        }
        if (options.autotune_repeats <= 0) {
            throw std::runtime_error("autotune repeats must be positive");
        }
    }
    return options;
}

std::vector<float> read_float32_file(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("unable to open file: " + path);
    }
    stream.seekg(0, std::ios::end);
    const std::streamsize bytes = stream.tellg();
    stream.seekg(0, std::ios::beg);
    if (bytes < 0 || bytes % static_cast<std::streamsize>(sizeof(float)) != 0) {
        throw std::runtime_error("file size is not a multiple of float32: " + path);
    }
    std::vector<float> values(static_cast<size_t>(bytes) / sizeof(float));
    if (!stream.read(reinterpret_cast<char*>(values.data()), bytes)) {
        throw std::runtime_error("failed to read file: " + path);
    }
    return values;
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

template <typename Accumulator>
__device__ Accumulator* shared_partial_buffer() {
    extern __shared__ __align__(sizeof(double)) unsigned char partial_raw[];
    return reinterpret_cast<Accumulator*>(partial_raw);
}

template <int TileSize, typename Accumulator>
__global__ void gemv_row_major_kernel(
    const float* matrix,
    const float* vector,
    float* output,
    int rows,
    int cols,
    int row_start,
    int row_count
) {
    const int output_row = blockIdx.x;
    const int thread_id = threadIdx.x;
    if (output_row >= row_count) {
        return;
    }

    const int row = row_start + output_row;
    Accumulator* partial = shared_partial_buffer<Accumulator>();
    Accumulator local_sum = static_cast<Accumulator>(0.0);
    const int row_base = row * cols;
    const int stride = blockDim.x * TileSize;

    for (int column_base = thread_id; column_base < cols; column_base += stride) {
        #pragma unroll
        for (int tile = 0; tile < TileSize; ++tile) {
            const int column = column_base + tile * blockDim.x;
            if (column < cols) {
                local_sum +=
                    static_cast<Accumulator>(matrix[row_base + column]) *
                    static_cast<Accumulator>(vector[column]);
            }
        }
    }

    partial[thread_id] = local_sum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (thread_id < offset) {
            partial[thread_id] += partial[thread_id + offset];
        }
        __syncthreads();
    }

    if (thread_id == 0) {
        output[output_row] = static_cast<float>(partial[0]);
    }
}

template <int TileSize, typename Accumulator>
__global__ void gemv_transposed_kernel(
    const float* matrix_t,
    const float* vector,
    float* output,
    int rows,
    int cols,
    int row_start,
    int row_count
) {
    const int output_row = blockIdx.x;
    const int thread_id = threadIdx.x;
    if (output_row >= row_count) {
        return;
    }

    const int row = row_start + output_row;
    Accumulator* partial = shared_partial_buffer<Accumulator>();
    Accumulator local_sum = static_cast<Accumulator>(0.0);
    const int stride = blockDim.x * TileSize;

    for (int column_base = thread_id; column_base < cols; column_base += stride) {
        #pragma unroll
        for (int tile = 0; tile < TileSize; ++tile) {
            const int column = column_base + tile * blockDim.x;
            if (column < cols) {
                local_sum +=
                    static_cast<Accumulator>(matrix_t[column * rows + row]) *
                    static_cast<Accumulator>(vector[column]);
            }
        }
    }

    partial[thread_id] = local_sum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (thread_id < offset) {
            partial[thread_id] += partial[thread_id + offset];
        }
        __syncthreads();
    }

    if (thread_id == 0) {
        output[output_row] = static_cast<float>(partial[0]);
    }
}

__global__ void transpose_matrix_kernel(
    const float* input,
    float* output,
    int rows,
    int cols
) {
    __shared__ float tile[32][33];

    const int x = blockIdx.x * 32 + threadIdx.x;
    const int y = blockIdx.y * 32 + threadIdx.y;

    if (x < cols && y < rows) {
        tile[threadIdx.y][threadIdx.x] = input[y * cols + x];
    }
    __syncthreads();

    const int transposed_x = blockIdx.y * 32 + threadIdx.x;
    const int transposed_y = blockIdx.x * 32 + threadIdx.y;
    if (transposed_x < rows && transposed_y < cols) {
        output[transposed_y * rows + transposed_x] = tile[threadIdx.x][threadIdx.y];
    }
}

void launch_row_major_kernel(
    int tile_size,
    bool use_fp64_accumulate,
    dim3 grid,
    dim3 block,
    size_t shared_bytes,
    const float* matrix,
    const float* vector,
    float* output,
    int rows,
    int cols,
    int row_start,
    int row_count
) {
    switch (tile_size) {
        case 1:
            if (use_fp64_accumulate) {
                gemv_row_major_kernel<1, double><<<grid, block, shared_bytes>>>(matrix, vector, output, rows, cols, row_start, row_count);
            } else {
                gemv_row_major_kernel<1, float><<<grid, block, shared_bytes>>>(matrix, vector, output, rows, cols, row_start, row_count);
            }
            break;
        case 2:
            if (use_fp64_accumulate) {
                gemv_row_major_kernel<2, double><<<grid, block, shared_bytes>>>(matrix, vector, output, rows, cols, row_start, row_count);
            } else {
                gemv_row_major_kernel<2, float><<<grid, block, shared_bytes>>>(matrix, vector, output, rows, cols, row_start, row_count);
            }
            break;
        case 4:
            if (use_fp64_accumulate) {
                gemv_row_major_kernel<4, double><<<grid, block, shared_bytes>>>(matrix, vector, output, rows, cols, row_start, row_count);
            } else {
                gemv_row_major_kernel<4, float><<<grid, block, shared_bytes>>>(matrix, vector, output, rows, cols, row_start, row_count);
            }
            break;
        case 8:
            if (use_fp64_accumulate) {
                gemv_row_major_kernel<8, double><<<grid, block, shared_bytes>>>(matrix, vector, output, rows, cols, row_start, row_count);
            } else {
                gemv_row_major_kernel<8, float><<<grid, block, shared_bytes>>>(matrix, vector, output, rows, cols, row_start, row_count);
            }
            break;
        default:
            throw std::runtime_error("unsupported tile size");
    }
}

void launch_transposed_kernel(
    int tile_size,
    bool use_fp64_accumulate,
    dim3 grid,
    dim3 block,
    size_t shared_bytes,
    const float* matrix,
    const float* vector,
    float* output,
    int rows,
    int cols,
    int row_start,
    int row_count
) {
    switch (tile_size) {
        case 1:
            if (use_fp64_accumulate) {
                gemv_transposed_kernel<1, double><<<grid, block, shared_bytes>>>(matrix, vector, output, rows, cols, row_start, row_count);
            } else {
                gemv_transposed_kernel<1, float><<<grid, block, shared_bytes>>>(matrix, vector, output, rows, cols, row_start, row_count);
            }
            break;
        case 2:
            if (use_fp64_accumulate) {
                gemv_transposed_kernel<2, double><<<grid, block, shared_bytes>>>(matrix, vector, output, rows, cols, row_start, row_count);
            } else {
                gemv_transposed_kernel<2, float><<<grid, block, shared_bytes>>>(matrix, vector, output, rows, cols, row_start, row_count);
            }
            break;
        case 4:
            if (use_fp64_accumulate) {
                gemv_transposed_kernel<4, double><<<grid, block, shared_bytes>>>(matrix, vector, output, rows, cols, row_start, row_count);
            } else {
                gemv_transposed_kernel<4, float><<<grid, block, shared_bytes>>>(matrix, vector, output, rows, cols, row_start, row_count);
            }
            break;
        case 8:
            if (use_fp64_accumulate) {
                gemv_transposed_kernel<8, double><<<grid, block, shared_bytes>>>(matrix, vector, output, rows, cols, row_start, row_count);
            } else {
                gemv_transposed_kernel<8, float><<<grid, block, shared_bytes>>>(matrix, vector, output, rows, cols, row_start, row_count);
            }
            break;
        default:
            throw std::runtime_error("unsupported tile size");
    }
}

std::string fnv1a64_checksum(const std::vector<float>& values) {
    uint64_t hash = 14695981039346656037ull;
    const unsigned char* bytes = reinterpret_cast<const unsigned char*>(values.data());
    const size_t byte_count = values.size() * sizeof(float);
    for (size_t index = 0; index < byte_count; ++index) {
        hash ^= static_cast<uint64_t>(bytes[index]);
        hash *= 1099511628211ull;
    }

    std::ostringstream stream;
    stream << "fnv1a64:" << std::hex << std::setw(16) << std::setfill('0') << hash;
    return stream.str();
}

std::string escape_json(const std::string& value) {
    std::ostringstream stream;
    for (const char ch : value) {
        if (ch == '\\' || ch == '"') {
            stream << '\\' << ch;
        } else {
            stream << ch;
        }
    }
    return stream.str();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);
        const std::vector<float> matrix_values = read_float32_file(options.matrix_path);
        const std::vector<float> vector_values = read_float32_file(options.vector_path);

        const size_t expected_matrix_size = static_cast<size_t>(options.rows) * static_cast<size_t>(options.cols);
        if (matrix_values.size() != expected_matrix_size) {
            throw std::runtime_error("matrix size does not match rows*cols");
        }
        if (vector_values.size() != static_cast<size_t>(options.cols)) {
            throw std::runtime_error("vector size does not match cols");
        }

        const int row_count = options.row_end - options.row_start;
        int device_index = 0;
        cudaDeviceProp device_props{};
        cuda_check(cudaGetDevice(&device_index), "cudaGetDevice");
        cuda_check(cudaGetDeviceProperties(&device_props, device_index), "cudaGetDeviceProperties");

        float* device_row_major = nullptr;
        float* device_transposed = nullptr;
        float* device_vector = nullptr;
        float* device_output = nullptr;
        cuda_check(cudaMalloc(&device_row_major, matrix_values.size() * sizeof(float)), "cudaMalloc row-major");
        cuda_check(cudaMalloc(&device_vector, vector_values.size() * sizeof(float)), "cudaMalloc vector");
        cuda_check(cudaMalloc(&device_output, static_cast<size_t>(row_count) * sizeof(float)), "cudaMalloc output");

        // Time the one-time upfront H2D uploads so the raw_report can surface
        // this fixed cost. Per-trial timing keeps H2D at zero for CUDA because
        // the matrix and vector are only uploaded once for the entire run.
        const auto h2d_started = std::chrono::steady_clock::now();
        cuda_check(
            cudaMemcpy(device_row_major, matrix_values.data(), matrix_values.size() * sizeof(float), cudaMemcpyHostToDevice),
            "cudaMemcpy row-major"
        );
        cuda_check(
            cudaMemcpy(device_vector, vector_values.data(), vector_values.size() * sizeof(float), cudaMemcpyHostToDevice),
            "cudaMemcpy vector"
        );
        const auto h2d_finished = std::chrono::steady_clock::now();
        const double one_time_h2d_seconds =
            std::chrono::duration<double>(h2d_finished - h2d_started).count();

        auto ensure_transposed_matrix = [&]() {
            if (device_transposed != nullptr) {
                return;
            }
            cuda_check(cudaMalloc(&device_transposed, matrix_values.size() * sizeof(float)), "cudaMalloc transposed");
            const dim3 block(32, 32);
            const dim3 grid(
                static_cast<unsigned int>((options.cols + block.x - 1) / block.x),
                static_cast<unsigned int>((options.rows + block.y - 1) / block.y)
            );
            transpose_matrix_kernel<<<grid, block>>>(device_row_major, device_transposed, options.rows, options.cols);
            cuda_check(cudaGetLastError(), "transpose_matrix_kernel");
            cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize transpose");
        };

        std::vector<float> host_output(static_cast<size_t>(row_count), 0.0f);
        std::vector<float> best_output_values(static_cast<size_t>(row_count), 0.0f);

        auto measure_config = [&](
            int transpose_mode,
            int block_size,
            int tile_size,
            const float* matrix_pointer,
            int repeats
        ) -> PhaseMetrics {
            const dim3 grid(static_cast<unsigned int>(row_count));
            const dim3 block(static_cast<unsigned int>(block_size));
            const bool use_fp64_accumulate = options.accumulation_precision == "fp64_accumulate";
            const size_t shared_bytes = static_cast<size_t>(block_size) * (use_fp64_accumulate ? sizeof(double) : sizeof(float));

            if (transpose_mode != 0) {
                launch_transposed_kernel(
                    tile_size,
                    use_fp64_accumulate,
                    grid,
                    block,
                    shared_bytes,
                    matrix_pointer,
                    device_vector,
                    device_output,
                    options.rows,
                    options.cols,
                    options.row_start,
                    row_count
                );
            } else {
                launch_row_major_kernel(
                    tile_size,
                    use_fp64_accumulate,
                    grid,
                    block,
                    shared_bytes,
                    matrix_pointer,
                    device_vector,
                    device_output,
                    options.rows,
                    options.cols,
                    options.row_start,
                    row_count
                );
            }
            cuda_check(cudaGetLastError(), "warmup kernel launch");
            cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize warmup");

            cudaEvent_t start_event{};
            cudaEvent_t stop_event{};
            cuda_check(cudaEventCreate(&start_event), "cudaEventCreate start");
            cuda_check(cudaEventCreate(&stop_event), "cudaEventCreate stop");
            cuda_check(cudaEventRecord(start_event), "cudaEventRecord start");

            for (int repeat = 0; repeat < repeats; ++repeat) {
                if (transpose_mode != 0) {
                    launch_transposed_kernel(
                        tile_size,
                        use_fp64_accumulate,
                        grid,
                        block,
                        shared_bytes,
                        matrix_pointer,
                        device_vector,
                        device_output,
                        options.rows,
                        options.cols,
                        options.row_start,
                        row_count
                    );
                } else {
                    launch_row_major_kernel(
                        tile_size,
                        use_fp64_accumulate,
                        grid,
                        block,
                        shared_bytes,
                        matrix_pointer,
                        device_vector,
                        device_output,
                        options.rows,
                        options.cols,
                        options.row_start,
                        row_count
                    );
                }
            }

            cuda_check(cudaGetLastError(), "kernel launch");
            cuda_check(cudaEventRecord(stop_event), "cudaEventRecord stop");
            cuda_check(cudaEventSynchronize(stop_event), "cudaEventSynchronize stop");

            float elapsed_ms = 0.0f;
            cuda_check(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event), "cudaEventElapsedTime");
            cuda_check(cudaEventDestroy(start_event), "cudaEventDestroy start");
            cuda_check(cudaEventDestroy(stop_event), "cudaEventDestroy stop");

            const double total_seconds = static_cast<double>(elapsed_ms) / 1000.0;
            const double latency_seconds = total_seconds / static_cast<double>(repeats);
            const double effective_gflops =
                (2.0 * static_cast<double>(row_count) * static_cast<double>(options.cols))
                / std::max(latency_seconds, 1e-12) / 1.0e9;

            // Time the single post-kernel D2H memcpy separately so the raw_report
            // can distinguish compute from the transfer back to host memory.
            const auto d2h_started = std::chrono::steady_clock::now();
            cuda_check(
                cudaMemcpy(
                    host_output.data(),
                    device_output,
                    host_output.size() * sizeof(float),
                    cudaMemcpyDeviceToHost
                ),
                "cudaMemcpy output"
            );
            const auto d2h_finished = std::chrono::steady_clock::now();
            const double d2h_seconds =
                std::chrono::duration<double>(d2h_finished - d2h_started).count();

            PhaseMetrics metrics;
            metrics.repeats = repeats;
            metrics.wall_clock_latency_seconds = latency_seconds;
            metrics.effective_gflops = effective_gflops;
            metrics.device_to_host_seconds = d2h_seconds;
            metrics.checksum = fnv1a64_checksum(host_output);
            return metrics;
        };

        // Memory-traffic model (FC-layer analogy): A is "weights", x is "input
        // activation", y is output. Compulsory traffic = one-touch lower bound.
        const double flops_per_run =
            2.0 * static_cast<double>(row_count) * static_cast<double>(options.cols);
        const size_t bytes_input =
            static_cast<size_t>(options.cols) * sizeof(float);
        const size_t bytes_weight =
            static_cast<size_t>(row_count) * static_cast<size_t>(options.cols) * sizeof(float);
        const size_t bytes_output =
            static_cast<size_t>(row_count) * sizeof(float);
        const size_t bytes_kernel_compulsory =
            bytes_input + bytes_weight + bytes_output;

        std::vector<TrialRecord> trials;

        auto emit_verbose_trial = [&](const TrialRecord& tr, int global_index, int global_total) {
            if (!options.verbose) return;
            const double compute_gflops = tr.host_compute_seconds > 0.0
                ? (flops_per_run / tr.host_compute_seconds / 1e9) : 0.0;
            const double effective_gflops = tr.total_wall_seconds > 0.0
                ? (flops_per_run / tr.total_wall_seconds / 1e9) : 0.0;
            fprintf(stderr,
                    "[gemv cuda %s %d/%d] candidate=%d/%d (trial %d/%d) "
                    "transpose=%d block_size=%d tile_size=%d "
                    "compute=%.6fs d2h=%.6fs total=%.6fs "
                    "compute_gflops=%.3f effective_gflops=%.3f\n",
                    tr.phase.c_str(), global_index, global_total,
                    tr.candidate_index + 1, tr.candidate_total,
                    tr.trial_index_within_candidate + 1, tr.repeats_for_candidate,
                    tr.transpose, tr.block_size, tr.tile_size,
                    tr.host_compute_seconds, tr.device_to_host_seconds, tr.total_wall_seconds,
                    compute_gflops, effective_gflops);
            fflush(stderr);
        };

        if (options.task_mode) {
            if (options.verbose) {
                fprintf(stderr,
                        "[gemv cuda plan] phase=task_execution "
                        "iteration_count=%d fixed_transpose=%d fixed_block_size=%d "
                        "fixed_tile_size=%d accumulation_precision=%s row_start=%d row_end=%d "
                        "device=\"%s\" compute_capability=%d.%d one_time_h2d=%.6fs\n",
                        options.measurement_repeats,
                        options.fixed_transpose,
                        options.fixed_block_size,
                        options.fixed_tile_size,
                        options.accumulation_precision.c_str(),
                        options.row_start,
                        options.row_end,
                        device_props.name,
                        device_props.major,
                        device_props.minor,
                        one_time_h2d_seconds);
                fflush(stderr);
            }

            const float* matrix_pointer = device_row_major;
            if (options.fixed_transpose != 0) {
                ensure_transposed_matrix();
                matrix_pointer = device_transposed;
            }

            const PhaseMetrics metrics = measure_config(
                options.fixed_transpose,
                options.fixed_block_size,
                options.fixed_tile_size,
                matrix_pointer,
                options.measurement_repeats
            );
            best_output_values = host_output;

            TrialRecord tr;
            tr.phase = "measurement";
            tr.candidate_index = 0;
            tr.candidate_total = 1;
            tr.trial_index_within_candidate = 0;
            tr.repeats_for_candidate = metrics.repeats;
            tr.transpose = options.fixed_transpose;
            tr.block_size = options.fixed_block_size;
            tr.tile_size = options.fixed_tile_size;
            tr.host_compute_seconds = metrics.wall_clock_latency_seconds;
            tr.device_to_host_seconds = metrics.device_to_host_seconds;
            tr.total_wall_seconds =
                metrics.wall_clock_latency_seconds + metrics.device_to_host_seconds;
            tr.checksum = metrics.checksum;
            emit_verbose_trial(tr, 1, 1);

            if (!options.output_path.empty()) {
                write_float32_file(options.output_path, best_output_values);
            }

            cudaFree(device_row_major);
            if (device_transposed != nullptr) {
                cudaFree(device_transposed);
            }
            cudaFree(device_vector);
            cudaFree(device_output);

            const double compute_event_ms =
                metrics.wall_clock_latency_seconds * static_cast<double>(metrics.repeats) * 1000.0;

            std::cout << "{"
                      << "\"backend\":\"cuda\","
                      << "\"mode\":\"task\","
                      << "\"device_name\":\"" << escape_json(device_props.name) << "\","
                      << "\"compute_capability\":\"" << device_props.major << device_props.minor << "\","
                      << "\"transpose\":" << options.fixed_transpose << ","
                      << "\"block_size\":" << options.fixed_block_size << ","
                      << "\"tile_size\":" << options.fixed_tile_size << ","
                      << "\"accumulation_precision\":\"" << options.accumulation_precision << "\","
                      << "\"row_start\":" << options.row_start << ","
                      << "\"row_end\":" << options.row_end << ","
                      << "\"iteration_count\":" << metrics.repeats << ","
                      << "\"wall_clock_latency_seconds\":" << std::fixed << std::setprecision(9)
                      << metrics.wall_clock_latency_seconds << ","
                      << "\"compute_event_ms\":" << std::fixed << std::setprecision(6)
                      << compute_event_ms << ","
                      << "\"effective_gflops\":" << std::fixed << std::setprecision(9)
                      << metrics.effective_gflops << ","
                      << "\"checksum\":\"" << metrics.checksum << "\""
                      << "}" << std::endl;
            return 0;
        }

        const dim3 benchmark_grid(static_cast<unsigned int>(options.rows));
        TrialMetrics best_metrics;
        bool have_best_trial = false;
        int trials_run = 0;

        const int autotune_total =
            static_cast<int>(options.transpose_modes.size()
                * options.block_sizes.size()
                * options.tile_sizes.size());
        int emitted_autotune = 0;
        int candidate_index = 0;
        if (options.verbose) {
            fprintf(stderr,
                    "[gemv cuda plan] phase=autotune transpose_candidates=%zu "
                    "block_size_candidates=%zu tile_size_candidates=%zu "
                    "autotune_repeats=%d total_candidates=%d "
                    "accumulation_precision=%s row_start=%d row_end=%d "
                    "device=\"%s\" compute_capability=%d.%d one_time_h2d=%.6fs\n",
                    options.transpose_modes.size(),
                    options.block_sizes.size(),
                    options.tile_sizes.size(),
                    options.autotune_repeats,
                    autotune_total,
                    options.accumulation_precision.c_str(),
                    options.row_start,
                    options.row_end,
                    device_props.name,
                    device_props.major,
                    device_props.minor,
                    one_time_h2d_seconds);
            fflush(stderr);
        }

        for (const int transpose_mode : options.transpose_modes) {
            const float* matrix_pointer = device_row_major;
            if (transpose_mode != 0) {
                ensure_transposed_matrix();
                matrix_pointer = device_transposed;
            }

            for (const int block_size : options.block_sizes) {
                if (block_size <= 0 || block_size > 1024) {
                    // Skipped candidates still consume an index so downstream
                    // consumers can align index numbers across runs.
                    ++candidate_index;
                    continue;
                }

                for (const int tile_size : options.tile_sizes) {
                    ++trials_run;

                    const PhaseMetrics autotune_metrics = measure_config(
                        transpose_mode,
                        block_size,
                        tile_size,
                        matrix_pointer,
                        options.autotune_repeats
                    );

                    TrialRecord tr;
                    tr.phase = "autotune";
                    tr.candidate_index = candidate_index;
                    tr.candidate_total = autotune_total;
                    tr.trial_index_within_candidate = 0;
                    tr.repeats_for_candidate = autotune_metrics.repeats;
                    tr.transpose = transpose_mode;
                    tr.block_size = block_size;
                    tr.tile_size = tile_size;
                    tr.host_compute_seconds = autotune_metrics.wall_clock_latency_seconds;
                    tr.device_to_host_seconds = autotune_metrics.device_to_host_seconds;
                    tr.total_wall_seconds =
                        autotune_metrics.wall_clock_latency_seconds
                        + autotune_metrics.device_to_host_seconds;
                    tr.checksum = autotune_metrics.checksum;
                    ++emitted_autotune;
                    emit_verbose_trial(tr, emitted_autotune, autotune_total);
                    trials.push_back(std::move(tr));

                    if (
                        !have_best_trial ||
                        autotune_metrics.wall_clock_latency_seconds < best_metrics.autotune.wall_clock_latency_seconds
                    ) {
                        have_best_trial = true;
                        best_metrics.transpose = transpose_mode;
                        best_metrics.block_size = block_size;
                        best_metrics.tile_size = tile_size;
                        best_metrics.autotune = autotune_metrics;
                    }
                    ++candidate_index;
                }
            }
        }

        if (!have_best_trial) {
            throw std::runtime_error("CUDA benchmark ran zero valid trials");
        }

        if (options.verbose) {
            fprintf(stderr,
                    "[gemv cuda plan] phase=measurement "
                    "selected_transpose=%d selected_block_size=%d selected_tile_size=%d "
                    "measurement_repeats=%d\n",
                    best_metrics.transpose,
                    best_metrics.block_size,
                    best_metrics.tile_size,
                    options.measurement_repeats);
            fflush(stderr);
        }

        const float* best_matrix_pointer = device_row_major;
        if (best_metrics.transpose != 0) {
            ensure_transposed_matrix();
            best_matrix_pointer = device_transposed;
        }
        best_metrics.measurement = measure_config(
            best_metrics.transpose,
            best_metrics.block_size,
            best_metrics.tile_size,
            best_matrix_pointer,
            options.measurement_repeats
        );
        best_output_values = host_output;

        {
            TrialRecord tr;
            tr.phase = "measurement";
            tr.candidate_index = 0;
            tr.candidate_total = 1;
            tr.trial_index_within_candidate = 0;
            tr.repeats_for_candidate = best_metrics.measurement.repeats;
            tr.transpose = best_metrics.transpose;
            tr.block_size = best_metrics.block_size;
            tr.tile_size = best_metrics.tile_size;
            tr.host_compute_seconds = best_metrics.measurement.wall_clock_latency_seconds;
            tr.device_to_host_seconds = best_metrics.measurement.device_to_host_seconds;
            tr.total_wall_seconds =
                best_metrics.measurement.wall_clock_latency_seconds
                + best_metrics.measurement.device_to_host_seconds;
            tr.checksum = best_metrics.measurement.checksum;
            emit_verbose_trial(tr, 1, 1);
            trials.push_back(std::move(tr));
        }

        if (!options.output_path.empty()) {
            write_float32_file(options.output_path, best_output_values);
        }

        cudaFree(device_row_major);
        if (device_transposed != nullptr) {
            cudaFree(device_transposed);
        }
        cudaFree(device_vector);
        cudaFree(device_output);

        std::cout << "{"
                  << "\"backend\":\"cuda\","
                  << "\"device_name\":\"" << escape_json(device_props.name) << "\","
                  << "\"compute_capability\":\"" << device_props.major << device_props.minor << "\","
                  << "\"transpose\":" << best_metrics.transpose << ","
                  << "\"block_size\":" << best_metrics.block_size << ","
                  << "\"tile_size\":" << best_metrics.tile_size << ","
                  << "\"accumulation_precision\":\"" << options.accumulation_precision << "\","
                  << "\"autotune_repeats\":" << best_metrics.autotune.repeats << ","
                  << "\"measurement_repeats\":" << best_metrics.measurement.repeats << ","
                  << "\"trials_run\":" << trials_run << ","
                  << "\"autotune_wall_clock_latency_seconds\":" << std::fixed << std::setprecision(9)
                  << best_metrics.autotune.wall_clock_latency_seconds << ","
                  << "\"autotune_effective_gflops\":" << std::fixed << std::setprecision(9)
                  << best_metrics.autotune.effective_gflops << ","
                  << "\"autotune_checksum\":\"" << best_metrics.autotune.checksum << "\","
                  << "\"measurement_wall_clock_latency_seconds\":" << std::fixed << std::setprecision(9)
                  << best_metrics.measurement.wall_clock_latency_seconds << ","
                  << "\"measurement_effective_gflops\":" << std::fixed << std::setprecision(9)
                  << best_metrics.measurement.effective_gflops << ","
                  << "\"measurement_checksum\":\"" << best_metrics.measurement.checksum << "\","
                  << "\"flops_per_run\":" << std::fixed << std::setprecision(1) << flops_per_run << ","
                  << "\"bytes_input\":" << bytes_input << ","
                  << "\"bytes_weight\":" << bytes_weight << ","
                  << "\"bytes_output\":" << bytes_output << ","
                  << "\"bytes_kernel_compulsory_memory_traffic\":" << bytes_kernel_compulsory << ","
                  << "\"one_time_host_to_device_seconds\":" << std::fixed << std::setprecision(9) << one_time_h2d_seconds << ","
                  << "\"notes_schema\":\"gemv CUDA backend: host_prep_seconds is zero per-trial because matrix A and vector x are uploaded exactly once upfront (see one_time_host_to_device_seconds); host_compute_seconds equals the per-repeat averaged cudaEventElapsedTime of the kernel launch; device_to_host_seconds is a single chrono-timed D2H memcpy performed after the repeat loop; host_postproc_seconds excludes checksum cost; memory_bandwidth model treats matrix A as weights, vector x as input activation, vector y as output, and uses compulsory one-touch DRAM traffic as a lower bound (real traffic >= compulsory).\","
                  << "\"trials\":[";
        for (size_t i = 0; i < trials.size(); ++i) {
            const TrialRecord& tr = trials[i];
            const double compute_gflops = tr.host_compute_seconds > 0.0
                ? (flops_per_run / tr.host_compute_seconds / 1e9) : 0.0;
            const double effective_gflops = tr.total_wall_seconds > 0.0
                ? (flops_per_run / tr.total_wall_seconds / 1e9) : 0.0;
            const double kernel_bandwidth_gibps = tr.host_compute_seconds > 0.0
                ? (static_cast<double>(bytes_kernel_compulsory) / tr.host_compute_seconds
                   / (1024.0 * 1024.0 * 1024.0))
                : 0.0;
            const double pcie_d2h_bandwidth_gibps = tr.device_to_host_seconds > 0.0
                ? (static_cast<double>(bytes_output) / tr.device_to_host_seconds
                   / (1024.0 * 1024.0 * 1024.0))
                : 0.0;
            std::cout << "{"
                      << "\"phase\":\"" << tr.phase << "\","
                      << "\"candidate_index\":" << tr.candidate_index << ","
                      << "\"candidate_total\":" << tr.candidate_total << ","
                      << "\"trial_index_within_candidate\":" << tr.trial_index_within_candidate << ","
                      << "\"repeats_for_candidate\":" << tr.repeats_for_candidate << ","
                      << "\"transpose\":" << tr.transpose << ","
                      << "\"block_size\":" << tr.block_size << ","
                      << "\"tile_size\":" << tr.tile_size << ","
                      << std::fixed << std::setprecision(9)
                      << "\"host_prep_seconds\":" << tr.host_prep_seconds << ","
                      << "\"host_compute_seconds\":" << tr.host_compute_seconds << ","
                      << "\"device_to_host_seconds\":" << tr.device_to_host_seconds << ","
                      << "\"host_postproc_seconds\":" << tr.host_postproc_seconds << ","
                      << "\"total_wall_seconds\":" << tr.total_wall_seconds << ","
                      << std::setprecision(6)
                      << "\"compute_gflops\":" << compute_gflops << ","
                      << "\"effective_gflops\":" << effective_gflops << ","
                      << "\"pcie_h2d_bandwidth_gibps\":0.0,"
                      << "\"pcie_d2h_bandwidth_gibps\":" << pcie_d2h_bandwidth_gibps << ","
                      << "\"kernel_memory_bandwidth_gibps_compulsory_lower_bound_model\":" << kernel_bandwidth_gibps << ","
                      << "\"checksum\":\"" << tr.checksum << "\""
                      << "}";
            if (i + 1 < trials.size()) std::cout << ",";
        }
        std::cout << "]"
                  << "}" << std::endl;
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << exc.what() << std::endl;
        return 1;
    }
}
