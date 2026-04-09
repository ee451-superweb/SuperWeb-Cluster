#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string matrix_path;
    std::string vector_path;
    std::string output_path;
    int rows = 0;
    int cols = 0;
    std::vector<int> transpose_modes;
    std::vector<int> block_sizes;
    std::vector<int> tile_sizes;
    int repeats = 2;
};

struct TrialMetrics {
    int transpose = 0;
    int block_size = 0;
    int tile_size = 0;
    int repeats = 0;
    double wall_clock_latency_seconds = std::numeric_limits<double>::infinity();
    double effective_gflops = 0.0;
    std::string checksum;
};

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
        } else if (key == "--transpose-modes") {
            options.transpose_modes = parse_int_list(value);
        } else if (key == "--block-sizes") {
            options.block_sizes = parse_int_list(value);
        } else if (key == "--tile-sizes") {
            options.tile_sizes = parse_int_list(value);
        } else if (key == "--repeats") {
            options.repeats = std::stoi(value);
        } else {
            throw std::runtime_error("unknown flag: " + key);
        }
    }

    if (options.matrix_path.empty() || options.vector_path.empty()) {
        throw std::runtime_error("matrix and vector paths are required");
    }
    if (options.rows <= 0 || options.cols <= 0) {
        throw std::runtime_error("rows and cols must be positive");
    }
    if (options.transpose_modes.empty() || options.block_sizes.empty() || options.tile_sizes.empty()) {
        throw std::runtime_error("transpose/block/tile candidate lists are required");
    }
    if (options.repeats <= 0) {
        throw std::runtime_error("repeats must be positive");
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

template <int TileSize>
__global__ void fmvm_row_major_kernel(
    const float* matrix,
    const float* vector,
    float* output,
    int rows,
    int cols
) {
    const int row = blockIdx.x;
    const int thread_id = threadIdx.x;
    if (row >= rows) {
        return;
    }

    // Keep the CUDA path in FP32 to match the benchmark's intended arithmetic
    // model and to avoid paying the throughput cost of FP64 on consumer GPUs.
    extern __shared__ float partial[];
    float local_sum = 0.0f;
    const int row_base = row * cols;
    const int stride = blockDim.x * TileSize;

    for (int column_base = thread_id; column_base < cols; column_base += stride) {
        #pragma unroll
        for (int tile = 0; tile < TileSize; ++tile) {
            const int column = column_base + tile * blockDim.x;
            if (column < cols) {
                local_sum += matrix[row_base + column] * vector[column];
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
        output[row] = partial[0];
    }
}

template <int TileSize>
__global__ void fmvm_transposed_kernel(
    const float* matrix_t,
    const float* vector,
    float* output,
    int rows,
    int cols
) {
    const int row = blockIdx.x;
    const int thread_id = threadIdx.x;
    if (row >= rows) {
        return;
    }

    // Keep the transposed path on the same FP32 arithmetic scheme as the
    // row-major path.
    extern __shared__ float partial[];
    float local_sum = 0.0f;
    const int stride = blockDim.x * TileSize;

    for (int column_base = thread_id; column_base < cols; column_base += stride) {
        #pragma unroll
        for (int tile = 0; tile < TileSize; ++tile) {
            const int column = column_base + tile * blockDim.x;
            if (column < cols) {
                local_sum += matrix_t[column * rows + row] * vector[column];
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
        output[row] = partial[0];
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
    dim3 grid,
    dim3 block,
    size_t shared_bytes,
    const float* matrix,
    const float* vector,
    float* output,
    int rows,
    int cols
) {
    switch (tile_size) {
        case 1:
            fmvm_row_major_kernel<1><<<grid, block, shared_bytes>>>(matrix, vector, output, rows, cols);
            break;
        case 2:
            fmvm_row_major_kernel<2><<<grid, block, shared_bytes>>>(matrix, vector, output, rows, cols);
            break;
        case 4:
            fmvm_row_major_kernel<4><<<grid, block, shared_bytes>>>(matrix, vector, output, rows, cols);
            break;
        case 8:
            fmvm_row_major_kernel<8><<<grid, block, shared_bytes>>>(matrix, vector, output, rows, cols);
            break;
        default:
            throw std::runtime_error("unsupported tile size");
    }
}

void launch_transposed_kernel(
    int tile_size,
    dim3 grid,
    dim3 block,
    size_t shared_bytes,
    const float* matrix,
    const float* vector,
    float* output,
    int rows,
    int cols
) {
    switch (tile_size) {
        case 1:
            fmvm_transposed_kernel<1><<<grid, block, shared_bytes>>>(matrix, vector, output, rows, cols);
            break;
        case 2:
            fmvm_transposed_kernel<2><<<grid, block, shared_bytes>>>(matrix, vector, output, rows, cols);
            break;
        case 4:
            fmvm_transposed_kernel<4><<<grid, block, shared_bytes>>>(matrix, vector, output, rows, cols);
            break;
        case 8:
            fmvm_transposed_kernel<8><<<grid, block, shared_bytes>>>(matrix, vector, output, rows, cols);
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
        cuda_check(cudaMalloc(&device_output, static_cast<size_t>(options.rows) * sizeof(float)), "cudaMalloc output");

        cuda_check(
            cudaMemcpy(device_row_major, matrix_values.data(), matrix_values.size() * sizeof(float), cudaMemcpyHostToDevice),
            "cudaMemcpy row-major");
        cuda_check(
            cudaMemcpy(device_vector, vector_values.data(), vector_values.size() * sizeof(float), cudaMemcpyHostToDevice),
            "cudaMemcpy vector");

        auto ensure_transposed_matrix = [&]() {
            if (device_transposed != nullptr) {
                return;
            }
            cuda_check(cudaMalloc(&device_transposed, matrix_values.size() * sizeof(float)), "cudaMalloc transposed");
            const dim3 block(32, 32);
            const dim3 grid(
                static_cast<unsigned int>((options.cols + block.x - 1) / block.x),
                static_cast<unsigned int>((options.rows + block.y - 1) / block.y));
            transpose_matrix_kernel<<<grid, block>>>(device_row_major, device_transposed, options.rows, options.cols);
            cuda_check(cudaGetLastError(), "transpose_matrix_kernel");
            cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize transpose");
        };

        const dim3 grid(options.rows);
        TrialMetrics best_metrics;
        std::vector<float> host_output(static_cast<size_t>(options.rows), 0.0f);
        std::vector<float> best_output_values(static_cast<size_t>(options.rows), 0.0f);
        int trials_run = 0;

        for (const int transpose_mode : options.transpose_modes) {
            const float* matrix_pointer = device_row_major;
            if (transpose_mode != 0) {
                ensure_transposed_matrix();
                matrix_pointer = device_transposed;
            }

            for (const int block_size : options.block_sizes) {
                if (block_size <= 0 || block_size > 1024) {
                    continue;
                }
                const dim3 block(block_size);
                const size_t shared_bytes = static_cast<size_t>(block_size) * sizeof(float);

                for (const int tile_size : options.tile_sizes) {
                    ++trials_run;

                    if (transpose_mode != 0) {
                        launch_transposed_kernel(
                            tile_size,
                            grid,
                            block,
                            shared_bytes,
                            matrix_pointer,
                            device_vector,
                            device_output,
                            options.rows,
                            options.cols);
                    } else {
                        launch_row_major_kernel(
                            tile_size,
                            grid,
                            block,
                            shared_bytes,
                            matrix_pointer,
                            device_vector,
                            device_output,
                            options.rows,
                            options.cols);
                    }
                    cuda_check(cudaGetLastError(), "warmup kernel launch");
                    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize warmup");

                    cudaEvent_t start_event{};
                    cudaEvent_t stop_event{};
                    cuda_check(cudaEventCreate(&start_event), "cudaEventCreate start");
                    cuda_check(cudaEventCreate(&stop_event), "cudaEventCreate stop");
                    cuda_check(cudaEventRecord(start_event), "cudaEventRecord start");

                    for (int repeat = 0; repeat < options.repeats; ++repeat) {
                        if (transpose_mode != 0) {
                            launch_transposed_kernel(
                                tile_size,
                                grid,
                                block,
                                shared_bytes,
                                matrix_pointer,
                                device_vector,
                                device_output,
                                options.rows,
                                options.cols);
                        } else {
                            launch_row_major_kernel(
                                tile_size,
                                grid,
                                block,
                                shared_bytes,
                                matrix_pointer,
                                device_vector,
                                device_output,
                                options.rows,
                                options.cols);
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
                    const double latency_seconds = total_seconds / static_cast<double>(options.repeats);
                    const double effective_gflops =
                        (2.0 * static_cast<double>(options.rows) * static_cast<double>(options.cols))
                        / std::max(latency_seconds, 1e-12) / 1.0e9;

                    if (latency_seconds < best_metrics.wall_clock_latency_seconds) {
                        cuda_check(
                            cudaMemcpy(
                                host_output.data(),
                                device_output,
                                host_output.size() * sizeof(float),
                                cudaMemcpyDeviceToHost),
                            "cudaMemcpy output");

                        best_metrics.transpose = transpose_mode;
                        best_metrics.block_size = block_size;
                        best_metrics.tile_size = tile_size;
                        best_metrics.repeats = options.repeats;
                        best_metrics.wall_clock_latency_seconds = latency_seconds;
                        best_metrics.effective_gflops = effective_gflops;
                        best_metrics.checksum = fnv1a64_checksum(host_output);
                        best_output_values = host_output;
                    }
                }
            }
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
                  << "\"repeats\":" << best_metrics.repeats << ","
                  << "\"trials_run\":" << trials_run << ","
                  << "\"wall_clock_latency_seconds\":" << std::fixed << std::setprecision(9)
                  << best_metrics.wall_clock_latency_seconds << ","
                  << "\"effective_gflops\":" << std::fixed << std::setprecision(9)
                  << best_metrics.effective_gflops << ","
                  << "\"checksum\":\"" << best_metrics.checksum << "\""
                  << "}" << std::endl;
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << exc.what() << std::endl;
        return 1;
    }
}
