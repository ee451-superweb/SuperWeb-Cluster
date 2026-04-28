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
#include <thread>
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
    std::vector<int> worker_candidates;
    std::vector<int> tile_sizes;
    int fixed_workers = 0;
    int fixed_tile_size = 0;
    int autotune_repeats = 1;
    int measurement_repeats = 1;
    std::string accumulation_precision = "fp32";
    bool task_mode = false;
    bool verbose = false;
};

// One per-trial record for the raw_report.trials array. Mirrors the conv2d CPU
// schema so downstream aggregators can treat all CPU backends uniformly, plus a
// per-trial checksum because gemv's measure_config already computes one.
struct TrialRecord {
    std::string phase;                  // "autotune" or "measurement"
    int candidate_index = 0;
    int candidate_total = 0;
    int trial_index_within_candidate = 0;
    int repeats_for_candidate = 0;
    int requested_workers = 0;
    int actual_workers = 0;
    int tile_size = 0;
    double host_prep_seconds = 0.0;
    double host_compute_seconds = 0.0;
    double device_to_host_seconds = 0.0;
    double host_postproc_seconds = 0.0;
    double total_wall_seconds = 0.0;
    std::string checksum;
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

        // Memory-traffic model (FC-layer analogy): A is "weights", x is "input
        // activation", y is output. Compulsory traffic = one-touch lower bound.
        const double flops_per_run =
            2.0 * static_cast<double>(task_rows) * static_cast<double>(options.cols);
        const size_t bytes_input =
            static_cast<size_t>(options.cols) * sizeof(float);
        const size_t bytes_weight =
            static_cast<size_t>(task_rows) * static_cast<size_t>(options.cols) * sizeof(float);
        const size_t bytes_output =
            static_cast<size_t>(task_rows) * sizeof(float);
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
                    "[gemv cpu (macos) %s %d/%d] candidate=%d/%d (trial %d/%d) "
                    "requested_workers=%d actual_workers=%d tile_size=%d "
                    "compute=%.6fs total=%.6fs "
                    "compute_gflops=%.3f effective_gflops=%.3f\n",
                    tr.phase.c_str(), global_index, global_total,
                    tr.candidate_index + 1, tr.candidate_total,
                    tr.trial_index_within_candidate + 1, tr.repeats_for_candidate,
                    tr.requested_workers, tr.actual_workers, tr.tile_size,
                    tr.host_compute_seconds, tr.total_wall_seconds,
                    compute_gflops, effective_gflops);
            fflush(stderr);
        };

        if (options.task_mode) {
            if (options.verbose) {
                fprintf(stderr,
                        "[gemv cpu (macos) plan] phase=task_execution "
                        "iteration_count=%d fixed_workers=%d fixed_tile_size=%d "
                        "accumulation_precision=%s row_start=%d row_end=%d\n",
                        options.measurement_repeats,
                        options.fixed_workers,
                        options.fixed_tile_size,
                        options.accumulation_precision.c_str(),
                        options.row_start,
                        options.row_end);
                fflush(stderr);
            }

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

            TrialRecord tr;
            tr.phase = "measurement";
            tr.candidate_index = 0;
            tr.candidate_total = 1;
            tr.trial_index_within_candidate = 0;
            tr.repeats_for_candidate = metrics.repeats;
            tr.requested_workers = options.fixed_workers;
            tr.actual_workers = metrics.actual_workers;
            tr.tile_size = options.fixed_tile_size;
            tr.host_compute_seconds = metrics.wall_clock_latency_seconds;
            tr.total_wall_seconds = metrics.wall_clock_latency_seconds;
            tr.checksum = metrics.checksum;
            emit_verbose_trial(tr, 1, 1);

            if (!options.output_path.empty()) {
                write_float32_file(options.output_path, output_values);
            }

            const double compute_event_ms =
                metrics.wall_clock_latency_seconds * static_cast<double>(metrics.repeats) * 1000.0;

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
                      << "\"compute_event_ms\":" << std::fixed << std::setprecision(6)
                      << compute_event_ms << ","
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

        const int autotune_total =
            static_cast<int>(options.worker_candidates.size() * options.tile_sizes.size());
        int emitted_autotune = 0;
        int candidate_index = 0;
        if (options.verbose) {
            fprintf(stderr,
                    "[gemv cpu (macos) plan] phase=autotune worker_candidates=%zu "
                    "tile_size_candidates=%zu autotune_repeats=%d total_candidates=%d "
                    "accumulation_precision=%s row_start=%d row_end=%d\n",
                    options.worker_candidates.size(),
                    options.tile_sizes.size(),
                    options.autotune_repeats,
                    autotune_total,
                    options.accumulation_precision.c_str(),
                    options.row_start,
                    options.row_end);
            fflush(stderr);
        }

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

                TrialRecord tr;
                tr.phase = "autotune";
                tr.candidate_index = candidate_index;
                tr.candidate_total = autotune_total;
                tr.trial_index_within_candidate = 0;
                tr.repeats_for_candidate = autotune_metrics.repeats;
                tr.requested_workers = requested_workers;
                tr.actual_workers = autotune_metrics.actual_workers;
                tr.tile_size = tile_size;
                tr.host_compute_seconds = autotune_metrics.wall_clock_latency_seconds;
                tr.total_wall_seconds = autotune_metrics.wall_clock_latency_seconds;
                tr.checksum = autotune_metrics.checksum;
                ++emitted_autotune;
                emit_verbose_trial(tr, emitted_autotune, autotune_total);
                trials.push_back(std::move(tr));

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
                ++candidate_index;
            }
        }

        if (options.verbose) {
            fprintf(stderr,
                    "[gemv cpu (macos) plan] phase=measurement "
                    "selected_requested_workers=%d selected_tile_size=%d "
                    "measurement_repeats=%d\n",
                    best_metrics.requested_workers,
                    best_metrics.tile_size,
                    options.measurement_repeats);
            fflush(stderr);
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

        {
            TrialRecord tr;
            tr.phase = "measurement";
            tr.candidate_index = 0;
            tr.candidate_total = 1;
            tr.trial_index_within_candidate = 0;
            tr.repeats_for_candidate = best_metrics.measurement.repeats;
            tr.requested_workers = best_metrics.requested_workers;
            tr.actual_workers = best_metrics.measurement.actual_workers;
            tr.tile_size = best_metrics.tile_size;
            tr.host_compute_seconds = best_metrics.measurement.wall_clock_latency_seconds;
            tr.total_wall_seconds = best_metrics.measurement.wall_clock_latency_seconds;
            tr.checksum = best_metrics.measurement.checksum;
            emit_verbose_trial(tr, 1, 1);
            trials.push_back(std::move(tr));
        }

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
                  << "\"measurement_checksum\":\"" << best_metrics.measurement.checksum << "\","
                  << "\"flops_per_run\":" << std::fixed << std::setprecision(1) << flops_per_run << ","
                  << "\"bytes_input\":" << bytes_input << ","
                  << "\"bytes_weight\":" << bytes_weight << ","
                  << "\"bytes_output\":" << bytes_output << ","
                  << "\"bytes_kernel_compulsory_memory_traffic\":" << bytes_kernel_compulsory << ","
                  << "\"notes_schema\":\"gemv CPU (macos) backend: host_prep/device_to_host/host_postproc are zero by definition; host_compute equals the per-repeat averaged latency returned by measure_config (thread spawn+join overhead is included); memory_bandwidth model treats matrix A as weights, vector x as input activation, vector y as output, and uses compulsory one-touch DRAM traffic as a lower bound (real traffic >= compulsory).\","
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
            std::cout << "{"
                      << "\"phase\":\"" << tr.phase << "\","
                      << "\"candidate_index\":" << tr.candidate_index << ","
                      << "\"candidate_total\":" << tr.candidate_total << ","
                      << "\"trial_index_within_candidate\":" << tr.trial_index_within_candidate << ","
                      << "\"repeats_for_candidate\":" << tr.repeats_for_candidate << ","
                      << "\"requested_workers\":" << tr.requested_workers << ","
                      << "\"actual_workers\":" << tr.actual_workers << ","
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
                      << "\"pcie_d2h_bandwidth_gibps\":0.0,"
                      << "\"kernel_memory_bandwidth_gibps_compulsory_lower_bound_model\":" << kernel_bandwidth_gibps << ","
                      << "\"checksum\":\"" << tr.checksum << "\""
                      << "}";
            if (i + 1 < trials.size()) std::cout << ",";
        }
        std::cout << "]"
                  << "}" << std::endl;
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << escape_json(exc.what()) << std::endl;
        return 1;
    }
}
