// Self-contained multithreaded SGEMM runner for the GEMM CPU backend.
//
// Mirrors the cuBLAS runner's CLI and JSON stdout contract so the Python
// executor on the main node can dispatch to either backend without caring
// which one handled the slice. Unlike the CUDA runner, there is no single
// vendor kernel to lean on, so the runner partitions the assigned M-axis
// slice across ``hardware_concurrency()`` threads and uses an i-k-j
// accumulation order inside each thread for contiguous B-row access (the
// compiler's auto-vectoriser turns the innermost j loop into SSE/AVX FMAs
// on MSVC /O2 and clang -O3).
//
//   --mode benchmark  — 1 warmup call + iteration_count measured calls,
//                       chrono-bracketed; reports total wall kernel time
//                       in compute_event_ms.
//   --mode dispatch   — 1 measured call, chrono-bracketed, single-pass
//                       compute_event_ms.
//
// Partition: the worker is assigned rows [m_start, m_end) of the output C.
// A[m_start:m_end, :] @ B = C[m_start:m_end, :] (row-major float32).

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
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
    int requested_workers = 0;  // 0 = auto (hardware_concurrency)
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
        } else if (key == "--workers") {
            // Optional knob so a benchmark or stress harness can pin a specific
            // worker count. Runtime dispatch leaves this at 0 (auto).
            options.requested_workers = std::stoi(value);
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
    if (options.requested_workers < 0) {
        throw std::runtime_error("workers must be non-negative (0 = auto)");
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

// Compute C[row_start:row_end, :] += A[row_start:row_end, :] @ B.
// Row-major. The i-k-j ordering means each A[i,k] load amortises over N
// B[k,:] reads and N C[i,:] updates, and the innermost j loop has unit
// stride on both B and C — that's the pattern the compiler's auto-vectoriser
// wants for SSE/AVX FMA.
void compute_rows(
    const float* matrix_a,
    const float* matrix_b,
    float* matrix_c_slice,  // points at C[row_start - c_base_row, 0]
    int c_base_row,         // row_start of the whole slice (for offsetting into C)
    int row_start,
    int row_end,
    int n,
    int k
) {
    for (int i = row_start; i < row_end; ++i) {
        float* c_row = matrix_c_slice + static_cast<ptrdiff_t>(i - c_base_row) * n;
        std::memset(c_row, 0, static_cast<size_t>(n) * sizeof(float));
        const float* a_row = matrix_a + static_cast<ptrdiff_t>(i) * k;
        for (int kk = 0; kk < k; ++kk) {
            const float a_ik = a_row[kk];
            const float* b_row = matrix_b + static_cast<ptrdiff_t>(kk) * n;
            for (int j = 0; j < n; ++j) {
                c_row[j] += a_ik * b_row[j];
            }
        }
    }
}

int resolve_worker_count(int requested, int slice_rows) {
    if (requested > 0) {
        return std::max(1, std::min(requested, slice_rows));
    }
    const unsigned int hw = std::thread::hardware_concurrency();
    const int auto_workers = static_cast<int>(hw == 0 ? 1 : hw);
    return std::max(1, std::min(auto_workers, slice_rows));
}

// Run one full pass of C[slice, :] = A[slice, :] @ B across ``workers`` threads.
int run_once(
    const float* matrix_a,
    const float* matrix_b,
    float* matrix_c_slice,
    int c_base_row,
    int m_start,
    int m_end,
    int n,
    int k,
    int requested_workers
) {
    const int slice_rows = m_end - m_start;
    const int workers = resolve_worker_count(requested_workers, slice_rows);
    if (workers == 1) {
        compute_rows(matrix_a, matrix_b, matrix_c_slice, c_base_row, m_start, m_end, n, k);
        return 1;
    }
    std::vector<std::thread> threads;
    threads.reserve(static_cast<size_t>(workers));
    for (int w = 0; w < workers; ++w) {
        const int partition_start = m_start + (slice_rows * w) / workers;
        const int partition_end = m_start + (slice_rows * (w + 1)) / workers;
        if (partition_end <= partition_start) {
            continue;
        }
        threads.emplace_back(
            compute_rows,
            matrix_a,
            matrix_b,
            matrix_c_slice,
            c_base_row,
            partition_start,
            partition_end,
            n,
            k
        );
    }
    for (std::thread& t : threads) {
        t.join();
    }
    return workers;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);

        const int slice_m = options.m_end - options.m_start;
        const size_t a_elements = static_cast<size_t>(options.m) * options.k;
        const size_t b_elements = static_cast<size_t>(options.k) * options.n;
        const size_t c_slice_elements = static_cast<size_t>(slice_m) * options.n;

        // Load A and B once. Keep the full A (not just the slice) because the
        // runner reads [m_start:m_end) out of it; copying the slice would
        // double peak RSS and buy us nothing.
        std::vector<float> host_a(a_elements);
        std::vector<float> host_b(b_elements);
        read_exact_file(options.input_a_path, host_a.data(), a_elements * sizeof(float));
        read_exact_file(options.input_b_path, host_b.data(), b_elements * sizeof(float));

        std::vector<float> host_c_slice(c_slice_elements, 0.0f);

        // Warmup in benchmark mode primes the caches / allocator so the
        // measured loop reflects steady-state throughput.
        int actual_workers = 0;
        if (options.benchmark_mode) {
            actual_workers = run_once(
                host_a.data(),
                host_b.data(),
                host_c_slice.data(),
                options.m_start,
                options.m_start,
                options.m_end,
                options.n,
                options.k,
                options.requested_workers
            );
        }

        const int loop_count = options.benchmark_mode ? options.iteration_count : 1;
        const auto loop_started = std::chrono::steady_clock::now();
        for (int iteration = 0; iteration < loop_count; ++iteration) {
            actual_workers = run_once(
                host_a.data(),
                host_b.data(),
                host_c_slice.data(),
                options.m_start,
                options.m_start,
                options.m_end,
                options.n,
                options.k,
                options.requested_workers
            );
        }
        const auto loop_finished = std::chrono::steady_clock::now();

        if (actual_workers == 0) {
            // Guard for the pathological case where slice_rows==0 lets no
            // thread run; run_once would return immediately without touching
            // the counter.
            actual_workers = resolve_worker_count(options.requested_workers, std::max(1, slice_m));
        }

        if (!options.output_path.empty()) {
            write_binary_file(
                options.output_path,
                host_c_slice.data(),
                c_slice_elements * sizeof(float)
            );
        }

        const std::string checksum = fnv1a64_checksum(
            host_c_slice.data(),
            c_slice_elements * sizeof(float)
        );

        // compute_event_ms contract matches the cuBLAS runner: for benchmark
        // mode it is the summed kernel-loop wall time across all iterations
        // (scales linearly with iteration_count), for dispatch mode it is the
        // single-pass wall time.
        const double loop_seconds = std::chrono::duration<double>(loop_finished - loop_started).count();
        const double compute_event_ms = loop_seconds * 1000.0;
        const double per_iter_seconds = loop_seconds / static_cast<double>(std::max(1, loop_count));
        const double flops_per_run = 2.0 * static_cast<double>(slice_m)
                                       * static_cast<double>(options.n)
                                       * static_cast<double>(options.k);
        const double effective_gflops = per_iter_seconds > 0.0
            ? flops_per_run / per_iter_seconds / 1e9 : 0.0;

        const unsigned int hardware_concurrency = std::thread::hardware_concurrency();
        const char* mode_str = options.benchmark_mode ? "benchmark" : "dispatch";

        if (options.verbose) {
            std::fprintf(stderr,
                         "[gemm cpu %s] workers=%d (auto_hw=%u) iterations=%d per_iter=%.6fs "
                         "compute_event_ms=%.3f effective_gflops=%.3f\n",
                         mode_str, actual_workers, hardware_concurrency, loop_count,
                         per_iter_seconds, compute_event_ms, effective_gflops);
            std::fflush(stderr);
        }

        std::cout << "{"
                  << "\"backend\":\"cpu\","
                  << "\"mode\":\"" << mode_str << "\","
                  << "\"hardware_concurrency\":" << hardware_concurrency << ","
                  << "\"requested_workers\":" << options.requested_workers << ","
                  << "\"actual_workers\":" << actual_workers << ","
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
