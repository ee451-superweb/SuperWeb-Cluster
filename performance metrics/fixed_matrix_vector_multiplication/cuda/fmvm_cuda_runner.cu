#include <cuda_runtime.h>

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
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
    int transpose = 0;
    int block_size = 256;
    int tile_size = 4;
    int repeats = 8;
};

inline void cuda_check(cudaError_t status, const char* message) {
    if (status != cudaSuccess) {
        std::ostringstream builder;
        builder << message << ": " << cudaGetErrorString(status);
        throw std::runtime_error(builder.str());
    }
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
        } else if (key == "--transpose") {
            options.transpose = std::stoi(value);
        } else if (key == "--block-size") {
            options.block_size = std::stoi(value);
        } else if (key == "--tile-size") {
            options.tile_size = std::stoi(value);
        } else if (key == "--repeats") {
            options.repeats = std::stoi(value);
        } else {
            throw std::runtime_error("unknown flag: " + key);
        }
    }

    if (options.matrix_path.empty() || options.vector_path.empty() || options.output_path.empty()) {
        throw std::runtime_error("matrix/vector/output paths are required");
    }
    if (options.rows <= 0 || options.cols <= 0) {
        throw std::runtime_error("rows and cols must be positive");
    }
    if (options.block_size <= 0 || options.block_size > 1024) {
        throw std::runtime_error("block size must be between 1 and 1024");
    }
    if (options.tile_size <= 0) {
        throw std::runtime_error("tile size must be positive");
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
    stream.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
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

template <template <int> class Kernel>
void launch_tiled_kernel(
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
            Kernel<1><<<grid, block, shared_bytes>>>(matrix, vector, output, rows, cols);
            break;
        case 2:
            Kernel<2><<<grid, block, shared_bytes>>>(matrix, vector, output, rows, cols);
            break;
        case 4:
            Kernel<4><<<grid, block, shared_bytes>>>(matrix, vector, output, rows, cols);
            break;
        case 8:
            Kernel<8><<<grid, block, shared_bytes>>>(matrix, vector, output, rows, cols);
            break;
        default:
            throw std::runtime_error("unsupported tile size");
    }
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

        float* device_matrix = nullptr;
        float* device_vector = nullptr;
        float* device_output = nullptr;

        cuda_check(cudaMalloc(&device_matrix, matrix_values.size() * sizeof(float)), "cudaMalloc matrix");
        cuda_check(cudaMalloc(&device_vector, vector_values.size() * sizeof(float)), "cudaMalloc vector");
        cuda_check(cudaMalloc(&device_output, static_cast<size_t>(options.rows) * sizeof(float)), "cudaMalloc output");

        cuda_check(
            cudaMemcpy(device_matrix, matrix_values.data(), matrix_values.size() * sizeof(float), cudaMemcpyHostToDevice),
            "cudaMemcpy matrix");
        cuda_check(
            cudaMemcpy(device_vector, vector_values.data(), vector_values.size() * sizeof(float), cudaMemcpyHostToDevice),
            "cudaMemcpy vector");

        int device_index = 0;
        cudaDeviceProp device_props{};
        cuda_check(cudaGetDevice(&device_index), "cudaGetDevice");
        cuda_check(cudaGetDeviceProperties(&device_props, device_index), "cudaGetDeviceProperties");

        const dim3 grid(options.rows);
        const dim3 block(options.block_size);
        const size_t shared_bytes = static_cast<size_t>(options.block_size) * sizeof(float);

        // Warm-up launch.
        if (options.transpose) {
            launch_tiled_kernel<fmvm_transposed_kernel>(
                options.tile_size,
                grid,
                block,
                shared_bytes,
                device_matrix,
                device_vector,
                device_output,
                options.rows,
                options.cols);
        } else {
            launch_tiled_kernel<fmvm_row_major_kernel>(
                options.tile_size,
                grid,
                block,
                shared_bytes,
                device_matrix,
                device_vector,
                device_output,
                options.rows,
                options.cols);
        }
        cuda_check(cudaGetLastError(), "kernel launch");
        cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

        cudaEvent_t start_event{};
        cudaEvent_t stop_event{};
        cuda_check(cudaEventCreate(&start_event), "cudaEventCreate start");
        cuda_check(cudaEventCreate(&stop_event), "cudaEventCreate stop");
        cuda_check(cudaEventRecord(start_event), "cudaEventRecord start");

        for (int repeat = 0; repeat < options.repeats; ++repeat) {
            if (options.transpose) {
                launch_tiled_kernel<fmvm_transposed_kernel>(
                    options.tile_size,
                    grid,
                    block,
                    shared_bytes,
                    device_matrix,
                    device_vector,
                    device_output,
                    options.rows,
                    options.cols);
            } else {
                launch_tiled_kernel<fmvm_row_major_kernel>(
                    options.tile_size,
                    grid,
                    block,
                    shared_bytes,
                    device_matrix,
                    device_vector,
                    device_output,
                    options.rows,
                    options.cols);
            }
        }

        cuda_check(cudaGetLastError(), "kernel launch");
        cuda_check(cudaEventRecord(stop_event), "cudaEventRecord stop");
        cuda_check(cudaEventSynchronize(stop_event), "cudaEventSynchronize");

        float elapsed_ms = 0.0f;
        cuda_check(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event), "cudaEventElapsedTime");

        std::vector<float> output_values(static_cast<size_t>(options.rows));
        cuda_check(
            cudaMemcpy(output_values.data(), device_output, output_values.size() * sizeof(float), cudaMemcpyDeviceToHost),
            "cudaMemcpy output");
        write_float32_file(options.output_path, output_values);

        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
        cudaFree(device_matrix);
        cudaFree(device_vector);
        cudaFree(device_output);

        const double total_seconds = static_cast<double>(elapsed_ms) / 1000.0;
        const double per_run_seconds = total_seconds / static_cast<double>(options.repeats);

        std::cout << "{"
                  << "\"backend\":\"cuda\","
                  << "\"device_name\":\"" << escape_json(device_props.name) << "\","
                  << "\"total_seconds\":" << std::fixed << std::setprecision(9) << total_seconds << ","
                  << "\"per_run_seconds\":" << std::fixed << std::setprecision(9) << per_run_seconds << ","
                  << "\"repeats\":" << options.repeats << ","
                  << "\"block_size\":" << options.block_size << ","
                  << "\"tile_size\":" << options.tile_size << ","
                  << "\"transpose\":" << options.transpose
                  << "}" << std::endl;
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << exc.what() << std::endl;
        return 1;
    }
}
