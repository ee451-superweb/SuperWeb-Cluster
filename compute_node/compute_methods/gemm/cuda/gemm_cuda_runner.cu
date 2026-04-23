// cuBLAS-backed GEMM runner.
//
// Design note: unlike the GEMV/conv2d runners, GEMM does NOT sweep kernel
// candidates. cuBLAS already picks its own internal kernel / TensorCore path
// per (M, N, K, device). We only distinguish:
//   --mode benchmark  — 1 warmup call + iteration_count measured calls,
//                       cudaEvent-bracketed, report total kernel time in
//                       compute_event_ms (scales linearly with iteration_count,
//                       matching the GEMV/conv2d contract).
//   --mode dispatch   — 1 measured call, cudaEvent-bracketed, single-pass
//                       compute_event_ms.
//
// Partition: the worker is assigned rows [m_start, m_end) of the output C.
// A[m_start:m_end, :] @ B = C[m_start:m_end, :]. Row-major on host, cuBLAS is
// column-major so we swap operand order (see sgemm call).

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
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
    std::string input_a_path;
    std::string input_b_path;
    std::string output_path;
    int m = 0;
    int n = 0;
    int k = 0;
    int m_start = 0;
    int m_end = 0;
    int iteration_count = 1;
    bool benchmark_mode = false;
    bool verbose = false;
};

Options parse_args(int argc, char** argv) {
    Options options;
    int i = 1;
    while (i < argc) {
        const std::string key = argv[i];
        if (key == "--verbose") {
            options.verbose = true;
            ++i;
            continue;
        }
        if (i + 1 >= argc) {
            throw std::runtime_error("missing value for command line flag: " + key);
        }
        const std::string value = argv[i + 1];
        i += 2;
        if (key == "--input-a") {
            options.input_a_path = value;
        } else if (key == "--input-b") {
            options.input_b_path = value;
        } else if (key == "--output") {
            options.output_path = value;
        } else if (key == "--m") {
            options.m = std::stoi(value);
        } else if (key == "--n") {
            options.n = std::stoi(value);
        } else if (key == "--k") {
            options.k = std::stoi(value);
        } else if (key == "--m-start") {
            options.m_start = std::stoi(value);
        } else if (key == "--m-end") {
            options.m_end = std::stoi(value);
        } else if (key == "--iteration-count") {
            options.iteration_count = std::stoi(value);
        } else if (key == "--mode") {
            if (value == "dispatch") {
                options.benchmark_mode = false;
            } else if (value == "benchmark") {
                options.benchmark_mode = true;
            } else {
                throw std::runtime_error("unknown --mode value: " + value);
            }
        } else {
            throw std::runtime_error("unknown flag: " + key);
        }
    }

    if (options.input_a_path.empty() || options.input_b_path.empty()) {
        throw std::runtime_error("--input-a and --input-b are required");
    }
    if (options.m <= 0 || options.n <= 0 || options.k <= 0) {
        throw std::runtime_error("M, N, K must be positive");
    }
    if (options.m_end == 0) {
        options.m_end = options.m;
    }
    if (options.m_start < 0 || options.m_end <= options.m_start || options.m_end > options.m) {
        throw std::runtime_error("m-start/m-end slice is out of range");
    }
    if (options.iteration_count <= 0) {
        throw std::runtime_error("iteration-count must be positive");
    }
    return options;
}

void read_exact_file(const std::string& path, void* destination, size_t expected_bytes) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("cannot open file: " + path);
    }
    stream.seekg(0, std::ios::end);
    const std::streamsize bytes = stream.tellg();
    stream.seekg(0, std::ios::beg);
    if (bytes < 0 || static_cast<size_t>(bytes) != expected_bytes) {
        throw std::runtime_error("file size mismatch: " + path);
    }
    if (!stream.read(static_cast<char*>(destination), bytes)) {
        throw std::runtime_error("failed to read file: " + path);
    }
}

void write_binary_file(const std::string& path, const void* data, size_t bytes) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) {
        throw std::runtime_error("cannot open output file: " + path);
    }
    if (!stream.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes))) {
        throw std::runtime_error("failed to write output file: " + path);
    }
}

std::string fnv1a64_checksum(const void* data, size_t byte_count) {
    uint64_t hash = 14695981039346656037ull;
    const unsigned char* bytes = static_cast<const unsigned char*>(data);
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

#define CHECK_CUDA(expr)                                                           \
    do {                                                                           \
        const cudaError_t _err = (expr);                                           \
        if (_err != cudaSuccess) {                                                 \
            throw std::runtime_error(std::string("CUDA error: ") +                 \
                                     cudaGetErrorString(_err));                    \
        }                                                                          \
    } while (0)

#define CHECK_CUBLAS(expr)                                                         \
    do {                                                                           \
        const cublasStatus_t _status = (expr);                                     \
        if (_status != CUBLAS_STATUS_SUCCESS) {                                    \
            throw std::runtime_error("cuBLAS error (status=" +                     \
                                     std::to_string(static_cast<int>(_status)) +  \
                                     ")");                                         \
        }                                                                          \
    } while (0)

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);

        const int slice_m = options.m_end - options.m_start;
        const size_t a_elements = static_cast<size_t>(options.m) * options.k;
        const size_t b_elements = static_cast<size_t>(options.k) * options.n;
        const size_t c_slice_elements = static_cast<size_t>(slice_m) * options.n;
        const size_t a_slice_elements = static_cast<size_t>(slice_m) * options.k;

        // Load the full A (to slice rows [m_start, m_end)) and B.
        std::vector<float> host_a_full(a_elements);
        std::vector<float> host_b(b_elements);
        read_exact_file(options.input_a_path, host_a_full.data(), a_elements * sizeof(float));
        read_exact_file(options.input_b_path, host_b.data(), b_elements * sizeof(float));

        const float* host_a_slice = host_a_full.data() + static_cast<size_t>(options.m_start) * options.k;

        cudaDeviceProp device_props{};
        int device_id = 0;
        CHECK_CUDA(cudaGetDevice(&device_id));
        CHECK_CUDA(cudaGetDeviceProperties(&device_props, device_id));

        // Device allocations.
        float* d_a = nullptr;
        float* d_b = nullptr;
        float* d_c = nullptr;
        CHECK_CUDA(cudaMalloc(&d_a, a_slice_elements * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_b, b_elements * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_c, c_slice_elements * sizeof(float)));

        CHECK_CUDA(cudaMemcpy(d_a, host_a_slice, a_slice_elements * sizeof(float),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_b, host_b.data(), b_elements * sizeof(float),
                              cudaMemcpyHostToDevice));

        cublasHandle_t handle = nullptr;
        CHECK_CUBLAS(cublasCreate(&handle));

        // Row-major host layout:
        //   A_host is [slice_m, K]  (slice rows of full [M, K])
        //   B_host is [K, N]
        //   C_host is [slice_m, N]
        // cuBLAS is column-major. To compute C = A @ B in row-major using a
        // column-major GEMM, swap operand order:
        //   C_col[N, slice_m] = B_col[N, K] @ A_col[K, slice_m]
        // which writes the same memory layout that a row-major C[slice_m, N]
        // would occupy. Leading dimensions: ldb = N, lda = K, ldc = N.
        const float alpha = 1.0f;
        const float beta = 0.0f;

        auto run_once = [&]() {
            CHECK_CUBLAS(cublasSgemm(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                options.n,       // m in cuBLAS (rows of C_col, cols of C_host)
                slice_m,         // n in cuBLAS (cols of C_col, rows of C_host)
                options.k,       // shared inner dim
                &alpha,
                d_b, options.n,  // B stored row-major [K, N] is col-major [N, K], ld=N
                d_a, options.k,  // A stored row-major [slice_m, K] is col-major [K, slice_m], ld=K
                &beta,
                d_c, options.n   // C stored row-major [slice_m, N] is col-major [N, slice_m], ld=N
            ));
        };

        // Warmup in benchmark mode lets cuBLAS finish its internal first-call
        // kernel selection so the measured iterations reflect steady-state
        // kernel time. Dispatch mode skips this — the whole point is one call.
        if (options.benchmark_mode) {
            run_once();
            CHECK_CUDA(cudaDeviceSynchronize());
        }

        cudaEvent_t ev_start = nullptr;
        cudaEvent_t ev_stop = nullptr;
        CHECK_CUDA(cudaEventCreate(&ev_start));
        CHECK_CUDA(cudaEventCreate(&ev_stop));

        const int loop_count = options.benchmark_mode ? options.iteration_count : 1;
        CHECK_CUDA(cudaEventRecord(ev_start));
        for (int iteration = 0; iteration < loop_count; ++iteration) {
            run_once();
        }
        CHECK_CUDA(cudaEventRecord(ev_stop));
        CHECK_CUDA(cudaEventSynchronize(ev_stop));

        float elapsed_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop));

        // Pull C back once after the timing window.
        std::vector<float> host_c_slice(c_slice_elements, 0.0f);
        CHECK_CUDA(cudaMemcpy(host_c_slice.data(), d_c, c_slice_elements * sizeof(float),
                              cudaMemcpyDeviceToHost));

        cudaEventDestroy(ev_start);
        cudaEventDestroy(ev_stop);
        cublasDestroy(handle);
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);

        if (!options.output_path.empty()) {
            write_binary_file(options.output_path, host_c_slice.data(),
                              c_slice_elements * sizeof(float));
        }

        const std::string checksum = fnv1a64_checksum(host_c_slice.data(),
                                                      c_slice_elements * sizeof(float));

        // compute_event_ms contract matches GEMV/conv2d: for benchmark mode
        // it's the summed kernel time across all iterations (scales with
        // iteration_count); for dispatch mode it's the one-pass kernel time.
        const double compute_event_ms = static_cast<double>(elapsed_ms);
        const double per_iter_seconds = compute_event_ms / 1000.0 / std::max(1, loop_count);
        // FLOPs per GEMM call = 2 * slice_m * N * K.
        const double flops_per_run = 2.0 * static_cast<double>(slice_m)
                                       * static_cast<double>(options.n)
                                       * static_cast<double>(options.k);
        const double effective_gflops = per_iter_seconds > 0.0
            ? flops_per_run / per_iter_seconds / 1e9 : 0.0;

        const char* mode_str = options.benchmark_mode ? "benchmark" : "dispatch";

        if (options.verbose) {
            std::fprintf(stderr,
                         "[gemm cuda %s] device=%s iterations=%d per_iter=%.6fs "
                         "compute_event_ms=%.3f effective_gflops=%.3f\n",
                         mode_str, device_props.name, loop_count,
                         per_iter_seconds, compute_event_ms, effective_gflops);
            std::fflush(stderr);
        }

        std::cout << "{"
                  << "\"backend\":\"cuda\","
                  << "\"mode\":\"" << mode_str << "\","
                  << "\"device_name\":\"" << escape_json(device_props.name) << "\","
                  << "\"compute_capability\":\"" << device_props.major
                  << device_props.minor << "\","
                  << "\"m\":" << options.m << ","
                  << "\"n\":" << options.n << ","
                  << "\"k\":" << options.k << ","
                  << "\"m_start\":" << options.m_start << ","
                  << "\"m_end\":" << options.m_end << ","
                  << "\"iteration_count\":" << loop_count << ","
                  << "\"compute_event_ms\":" << std::fixed << std::setprecision(6)
                  << compute_event_ms << ","
                  << "\"wall_clock_latency_seconds\":" << std::fixed << std::setprecision(9)
                  << per_iter_seconds << ","
                  << "\"effective_gflops\":" << std::fixed << std::setprecision(6)
                  << effective_gflops << ","
                  << "\"flops_per_run\":" << std::fixed << std::setprecision(1)
                  << flops_per_run << ","
                  << "\"checksum\":\"" << checksum << "\""
                  << "}" << std::endl;
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << exc.what() << std::endl;
        return 1;
    }
}
