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
#include <thread>
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
    std::vector<int> worker_candidates;
    std::vector<int> tile_sizes;
    int fixed_workers = 0;
    int fixed_tile_size = 0;
    int autotune_repeats = 1;
    int measurement_repeats = 1;
    std::string accumulation_precision = "fp32";
    bool task_mode = false;
};

struct PhaseMetrics {
    int actual_workers = 0;
    int repeats = 1;
    double wall_clock_latency_seconds = std::numeric_limits<double>::infinity();
    double effective_gflops = 0.0;
    std::string checksum;
};

struct TrialMetrics {
    int requested_workers = 0;
    int actual_workers = 0;
    int tile_size = 0;
    PhaseMetrics autotune;
    PhaseMetrics measurement;
};

bool is_supported_accumulation_precision(const std::string& value) {
    return value == "fp32" || value == "fp64_accumulate";
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
        } else if (key == "--row-start") {
            options.row_start = std::stoi(value);
        } else if (key == "--row-end") {
            options.row_end = std::stoi(value);
        } else if (key == "--workers") {
            options.worker_candidates = parse_int_list(value);
        } else if (key == "--tile-sizes") {
            options.tile_sizes = parse_int_list(value);
        } else if (key == "--fixed-workers") {
            options.fixed_workers = std::stoi(value);
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

    const bool has_fixed_workers = options.fixed_workers > 0;
    const bool has_fixed_tile_size = options.fixed_tile_size > 0;
    if (has_fixed_workers != has_fixed_tile_size) {
        throw std::runtime_error("task mode requires both fixed-workers and fixed-tile-size");
    }
    options.task_mode = has_fixed_workers && has_fixed_tile_size;

    if (!options.task_mode) {
        if (options.worker_candidates.empty()) {
            throw std::runtime_error("worker candidate list is required");
        }
        if (options.tile_sizes.empty()) {
            throw std::runtime_error("tile-size candidate list is required");
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

float dot_product_tiled_fp32(const float* matrix_row, const std::vector<float>& vector_values, int cols, int tile_size) {
    float accumulator = 0.0f;
    for (int tile_start = 0; tile_start < cols; tile_start += tile_size) {
        const int tile_end = std::min(cols, tile_start + tile_size);
        for (int col = tile_start; col < tile_end; ++col) {
            accumulator += matrix_row[col] * vector_values[static_cast<size_t>(col)];
        }
    }
    return accumulator;
}

float dot_product_tiled_fp64_accumulate(
    const float* matrix_row,
    const std::vector<float>& vector_values,
    int cols,
    int tile_size
) {
    double accumulator = 0.0;
    for (int tile_start = 0; tile_start < cols; tile_start += tile_size) {
        const int tile_end = std::min(cols, tile_start + tile_size);
        for (int col = tile_start; col < tile_end; ++col) {
            accumulator += static_cast<double>(matrix_row[col]) * static_cast<double>(vector_values[static_cast<size_t>(col)]);
        }
    }
    return static_cast<float>(accumulator);
}

void compute_row_range(
    const std::vector<float>& matrix_values,
    const std::vector<float>& vector_values,
    std::vector<float>& output_values,
    int cols,
    int row_start,
    int row_end,
    int tile_size,
    int output_row_offset,
    const std::string& accumulation_precision
) {
    for (int row = row_start; row < row_end; ++row) {
        const size_t row_base = static_cast<size_t>(row) * static_cast<size_t>(cols);
        const float* matrix_row = matrix_values.data() + row_base;
        if (accumulation_precision == "fp64_accumulate") {
            output_values[static_cast<size_t>(row - output_row_offset)] = dot_product_tiled_fp64_accumulate(
                matrix_row,
                vector_values,
                cols,
                tile_size
            );
        } else {
            output_values[static_cast<size_t>(row - output_row_offset)] = dot_product_tiled_fp32(
                matrix_row,
                vector_values,
                cols,
                tile_size
            );
        }
    }
}

int run_once(
    const std::vector<float>& matrix_values,
    const std::vector<float>& vector_values,
    std::vector<float>& output_values,
    int cols,
    int row_start,
    int row_end,
    int requested_workers,
    int tile_size,
    const std::string& accumulation_precision
) {
    const int task_rows = row_end - row_start;
    const int actual_workers = std::max(1, std::min({requested_workers, task_rows}));
    if (actual_workers == 1) {
        compute_row_range(
            matrix_values,
            vector_values,
            output_values,
            cols,
            row_start,
            row_end,
            tile_size,
            row_start,
            accumulation_precision
        );
        return actual_workers;
    }

    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(actual_workers));
    for (int worker_index = 0; worker_index < actual_workers; ++worker_index) {
        const int partition_start = row_start + (task_rows * worker_index) / actual_workers;
        const int partition_end = row_start + (task_rows * (worker_index + 1)) / actual_workers;
        workers.emplace_back(
            compute_row_range,
            std::cref(matrix_values),
            std::cref(vector_values),
            std::ref(output_values),
            cols,
            partition_start,
            partition_end,
            tile_size,
            row_start,
            std::cref(accumulation_precision)
        );
    }

    for (std::thread& worker : workers) {
        worker.join();
    }
    return actual_workers;
}

std::string fnv1a64_checksum(const std::vector<float>& values);

PhaseMetrics measure_config(
    const std::vector<float>& matrix_values,
    const std::vector<float>& vector_values,
    std::vector<float>& output_values,
    int cols,
    int row_start,
    int row_end,
    int requested_workers,
    int tile_size,
    int repeats,
    const std::string& accumulation_precision
) {
    const int task_rows = row_end - row_start;
    const int warmup_workers = run_once(
        matrix_values,
        vector_values,
        output_values,
        cols,
        row_start,
        row_end,
        requested_workers,
        tile_size,
        accumulation_precision
    );

    const auto started = std::chrono::steady_clock::now();
    int actual_workers = warmup_workers;
    for (int repeat = 0; repeat < repeats; ++repeat) {
        actual_workers = run_once(
            matrix_values,
            vector_values,
            output_values,
            cols,
            row_start,
            row_end,
            requested_workers,
            tile_size,
            accumulation_precision
        );
    }
    const auto finished = std::chrono::steady_clock::now();

    const double total_seconds = std::chrono::duration<double>(finished - started).count();
    const double latency_seconds = total_seconds / static_cast<double>(repeats);
    const double effective_gflops =
        (2.0 * static_cast<double>(task_rows) * static_cast<double>(cols))
        / std::max(latency_seconds, 1e-12) / 1.0e9;

    PhaseMetrics metrics;
    metrics.actual_workers = actual_workers;
    metrics.repeats = repeats;
    metrics.wall_clock_latency_seconds = latency_seconds;
    metrics.effective_gflops = effective_gflops;
    metrics.checksum = fnv1a64_checksum(output_values);
    return metrics;
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

        const int task_rows = options.row_end - options.row_start;
        std::vector<float> output_values(static_cast<size_t>(task_rows), 0.0f);

        if (options.task_mode) {
            const PhaseMetrics metrics = measure_config(
                matrix_values,
                vector_values,
                output_values,
                options.cols,
                options.row_start,
                options.row_end,
                options.fixed_workers,
                options.fixed_tile_size,
                options.measurement_repeats,
                options.accumulation_precision
            );

            if (!options.output_path.empty()) {
                write_float32_file(options.output_path, output_values);
            }

            std::cout << "{"
                      << "\"backend\":\"cpu\","
                      << "\"mode\":\"task\","
                      << "\"requested_workers\":" << options.fixed_workers << ","
                      << "\"actual_workers\":" << metrics.actual_workers << ","
                      << "\"tile_size\":" << options.fixed_tile_size << ","
                      << "\"accumulation_precision\":\"" << options.accumulation_precision << "\","
                      << "\"row_start\":" << options.row_start << ","
                      << "\"row_end\":" << options.row_end << ","
                      << "\"iteration_count\":" << metrics.repeats << ","
                      << "\"wall_clock_latency_seconds\":" << std::fixed << std::setprecision(9)
                      << metrics.wall_clock_latency_seconds << ","
                      << "\"effective_gflops\":" << std::fixed << std::setprecision(9)
                      << metrics.effective_gflops << ","
                      << "\"checksum\":\"" << metrics.checksum << "\""
                      << "}" << std::endl;
            return 0;
        }

        std::vector<float> best_output_values(static_cast<size_t>(task_rows), 0.0f);
        TrialMetrics best_metrics;
        bool have_best_trial = false;
        int trials_run = 0;

        for (const int requested_workers : options.worker_candidates) {
            for (const int tile_size : options.tile_sizes) {
                ++trials_run;

                const PhaseMetrics autotune_metrics = measure_config(
                    matrix_values,
                    vector_values,
                    output_values,
                    options.cols,
                    options.row_start,
                        options.row_end,
                        requested_workers,
                        tile_size,
                        options.autotune_repeats,
                        options.accumulation_precision
                    );

                if (
                    !have_best_trial ||
                    autotune_metrics.wall_clock_latency_seconds < best_metrics.autotune.wall_clock_latency_seconds
                ) {
                    have_best_trial = true;
                    best_metrics.requested_workers = requested_workers;
                    best_metrics.tile_size = tile_size;
                    best_metrics.actual_workers = autotune_metrics.actual_workers;
                    best_metrics.autotune = autotune_metrics;
                }
            }
        }

        best_metrics.measurement = measure_config(
            matrix_values,
            vector_values,
            output_values,
            options.cols,
            options.row_start,
            options.row_end,
            best_metrics.requested_workers,
            best_metrics.tile_size,
            options.measurement_repeats,
            options.accumulation_precision
        );
        best_metrics.actual_workers = best_metrics.measurement.actual_workers;
        best_output_values = output_values;

        if (!options.output_path.empty()) {
            write_float32_file(options.output_path, best_output_values);
        }
        if (!have_best_trial) {
            throw std::runtime_error("CPU benchmark ran zero valid trials");
        }

        const unsigned int hardware_concurrency = std::thread::hardware_concurrency();
        std::cout << "{"
                  << "\"backend\":\"cpu\","
                  << "\"hardware_concurrency\":" << hardware_concurrency << ","
                  << "\"requested_workers\":" << best_metrics.requested_workers << ","
                  << "\"actual_workers\":" << best_metrics.actual_workers << ","
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
                  << "\"measurement_checksum\":\"" << best_metrics.measurement.checksum << "\""
                  << "}" << std::endl;
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << escape_json(exc.what()) << std::endl;
        return 1;
    }
}
