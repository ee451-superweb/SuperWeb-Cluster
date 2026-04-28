#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdint>
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
    std::string library_path;
    std::string matrix_path;
    std::string vector_path;
    std::string output_path;
    int rows = 0;
    int cols = 0;
    int row_start = 0;
    int row_end = 0;
    std::vector<int> block_sizes;
    std::vector<int> tile_sizes;
    double headroom_fraction = 1.0;
    int row_chunk_size = 0;
    int autotune_repeats = 1;
    int measurement_repeats = 1;
    bool verbose = false;
};

struct PhaseMetrics {
    int repeats = 0;
    double wall_clock_latency_seconds = std::numeric_limits<double>::infinity();
    double effective_gflops = 0.0;
    double device_to_host_seconds = 0.0;  // post-compute memcpy out of shared MTLBuffer
    std::string checksum;
};

struct TrialMetrics {
    std::string implementation = "mps_matrix_vector_multiplication";
    PhaseMetrics autotune;
    PhaseMetrics measurement;
};

// One per-trial record for the raw_report.trials array. Schema mirrors the
// conv2d CPU/GPU layout so downstream aggregators share a common shape.
struct TrialRecord {
    std::string phase;                  // "autotune" or "measurement"
    int candidate_index = 0;
    int candidate_total = 0;
    int trial_index_within_candidate = 0;
    int repeats_for_candidate = 0;
    std::string implementation;
    int row_chunk_size = 0;
    double host_prep_seconds = 0.0;
    double host_compute_seconds = 0.0;
    double device_to_host_seconds = 0.0;
    double host_postproc_seconds = 0.0;
    double total_wall_seconds = 0.0;
    std::string checksum;
};

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

        if (key == "--library") {
            options.library_path = value;
        } else if (key == "--matrix") {
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
        } else if (key == "--block-sizes") {
            options.block_sizes = parse_int_list(value);
        } else if (key == "--tile-sizes") {
            options.tile_sizes = parse_int_list(value);
        } else if (key == "--headroom-fraction") {
            options.headroom_fraction = std::stod(value);
        } else if (key == "--row-chunk-size") {
            options.row_chunk_size = std::stoi(value);
        } else if (key == "--autotune-repeats") {
            options.autotune_repeats = std::stoi(value);
        } else if (key == "--measurement-repeats" || key == "--iteration-count") {
            options.measurement_repeats = std::stoi(value);
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
    if (options.block_sizes.empty() || options.tile_sizes.empty()) {
        throw std::runtime_error("block-size and tile-size candidate lists are required");
    }
    if (!(options.headroom_fraction > 0.0 && options.headroom_fraction <= 1.0)) {
        throw std::runtime_error("headroom fraction must be within (0.0, 1.0]");
    }
    if (options.row_chunk_size < 0) {
        throw std::runtime_error("row chunk size must be non-negative");
    }
    if (options.autotune_repeats <= 0 || options.measurement_repeats <= 0) {
        throw std::runtime_error("autotune and measurement repeats must be positive");
    }
    return options;
}

void read_exact_file(const std::string& path, void* destination, size_t expected_bytes) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("unable to open file: " + path);
    }

    stream.seekg(0, std::ios::end);
    const std::streamsize bytes = stream.tellg();
    stream.seekg(0, std::ios::beg);

    if (bytes < 0) {
        throw std::runtime_error("unable to determine file size: " + path);
    }
    if (static_cast<size_t>(bytes) != expected_bytes) {
        throw std::runtime_error("file size does not match expected byte count: " + path);
    }
    if (!stream.read(reinterpret_cast<char*>(destination), bytes)) {
        throw std::runtime_error("failed to read file: " + path);
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

std::string nsstring_to_string(NSString* value) {
    if (value == nil) {
        return std::string();
    }
    return std::string([value UTF8String]);
}

[[noreturn]] void throw_metal_error(const std::string& prefix, NSError* error) {
    std::ostringstream stream;
    stream << prefix;
    if (error != nil) {
        stream << ": " << nsstring_to_string([error localizedDescription]);
    }
    throw std::runtime_error(stream.str());
}

void ensure_completed(id<MTLCommandBuffer> command_buffer, const std::string& context) {
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if ([command_buffer status] != MTLCommandBufferStatusCompleted) {
        throw_metal_error(context, [command_buffer error]);
    }
}

double cooldown_seconds_for_fraction(double active_seconds, double fraction) {
    if (!(fraction > 0.0) || fraction >= 1.0 || active_seconds <= 0.0) {
        return 0.0;
    }
    return active_seconds * ((1.0 / fraction) - 1.0);
}

PhaseMetrics run_mps_gemv(
    id<MTLCommandQueue> command_queue,
    const Options& options,
    std::vector<float>& host_output,
    int repeats
) {
    @autoreleasepool {
        const size_t matrix_values =
            static_cast<size_t>(options.rows) * static_cast<size_t>(options.cols);
        const size_t vector_values = static_cast<size_t>(options.cols);
        const size_t output_rows = static_cast<size_t>(options.row_end - options.row_start);
        const size_t matrix_bytes = matrix_values * sizeof(float);
        const size_t vector_bytes = vector_values * sizeof(float);
        const size_t output_bytes = output_rows * sizeof(float);
        const size_t configured_row_chunk =
            options.row_chunk_size > 0 ? static_cast<size_t>(options.row_chunk_size) : output_rows;
        const size_t row_chunk_size = std::max<size_t>(1, std::min(output_rows, configured_row_chunk));

        id<MTLDevice> device = [command_queue device];
        id<MTLBuffer> matrix_buffer = [device newBufferWithLength:matrix_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> vector_buffer = [device newBufferWithLength:vector_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> output_buffer = [device newBufferWithLength:output_bytes options:MTLResourceStorageModeShared];
        if (matrix_buffer == nil || vector_buffer == nil || output_buffer == nil) {
            throw std::runtime_error("failed to allocate Metal buffers");
        }

        read_exact_file(options.matrix_path, [matrix_buffer contents], matrix_bytes);
        read_exact_file(options.vector_path, [vector_buffer contents], vector_bytes);
        std::memset([output_buffer contents], 0, output_bytes);

        MPSMatrixDescriptor* matrix_descriptor =
            [MPSMatrixDescriptor matrixDescriptorWithRows:static_cast<NSUInteger>(options.rows)
                                                  columns:static_cast<NSUInteger>(options.cols)
                                                 rowBytes:static_cast<NSUInteger>(options.cols * sizeof(float))
                                                 dataType:MPSDataTypeFloat32];
        MPSVectorDescriptor* input_vector_descriptor =
            [MPSVectorDescriptor vectorDescriptorWithLength:static_cast<NSUInteger>(options.cols)
                                                   dataType:MPSDataTypeFloat32];
        MPSVectorDescriptor* output_vector_descriptor =
            [MPSVectorDescriptor vectorDescriptorWithLength:static_cast<NSUInteger>(output_rows)
                                                   dataType:MPSDataTypeFloat32];

        if (matrix_descriptor == nil || input_vector_descriptor == nil || output_vector_descriptor == nil) {
            throw std::runtime_error("failed to create Metal Performance Shaders descriptors");
        }

        MPSMatrix* input_matrix = [[MPSMatrix alloc] initWithBuffer:matrix_buffer descriptor:matrix_descriptor];
        MPSVector* input_vector = [[MPSVector alloc] initWithBuffer:vector_buffer descriptor:input_vector_descriptor];
        MPSVector* output_vector = [[MPSVector alloc] initWithBuffer:output_buffer descriptor:output_vector_descriptor];
        if (input_matrix == nil || input_vector == nil || output_vector == nil) {
            throw std::runtime_error("failed to create Metal Performance Shaders matrix/vector wrappers");
        }

        auto encode_chunk = [&](
            id<MTLCommandBuffer> command_buffer,
            size_t chunk_row_offset,
            size_t chunk_row_count
        ) {
            MPSMatrixVectorMultiplication* kernel =
                [[MPSMatrixVectorMultiplication alloc] initWithDevice:device
                                                                 rows:static_cast<NSUInteger>(chunk_row_count)
                                                              columns:static_cast<NSUInteger>(options.cols)];
            if (kernel == nil) {
                throw std::runtime_error("failed to create MPSMatrixVectorMultiplication kernel");
            }
            kernel.primarySourceMatrixOrigin = MTLOriginMake(
                static_cast<NSUInteger>(options.row_start + static_cast<int>(chunk_row_offset)),
                0,
                0
            );
            kernel.secondarySourceMatrixOrigin = MTLOriginMake(0, 0, 0);
            kernel.resultMatrixOrigin = MTLOriginMake(static_cast<NSUInteger>(chunk_row_offset), 0, 0);
            [kernel encodeToCommandBuffer:command_buffer
                              inputMatrix:input_matrix
                              inputVector:input_vector
                             resultVector:output_vector];
        };

        auto run_pass = [&](bool include_cooldown, bool cooldown_after_final_chunk) -> double {
            double total_seconds = 0.0;
            for (size_t chunk_row_offset = 0; chunk_row_offset < output_rows; chunk_row_offset += row_chunk_size) {
                const size_t chunk_row_count = std::min(row_chunk_size, output_rows - chunk_row_offset);
                id<MTLCommandBuffer> chunk_buffer = [command_queue commandBuffer];
                encode_chunk(chunk_buffer, chunk_row_offset, chunk_row_count);

                const auto chunk_started = std::chrono::steady_clock::now();
                ensure_completed(chunk_buffer, "Metal command buffer failed");
                const auto chunk_finished = std::chrono::steady_clock::now();

                double chunk_seconds = std::chrono::duration<double>(chunk_finished - chunk_started).count();
                const CFTimeInterval gpu_started = [chunk_buffer GPUStartTime];
                const CFTimeInterval gpu_finished = [chunk_buffer GPUEndTime];
                if (gpu_started > 0.0 && gpu_finished > gpu_started) {
                    chunk_seconds = static_cast<double>(gpu_finished - gpu_started);
                }
                total_seconds += chunk_seconds;

                const bool final_chunk = (chunk_row_offset + chunk_row_count) >= output_rows;
                if (include_cooldown && (!final_chunk || cooldown_after_final_chunk)) {
                    const double cooldown_seconds =
                        cooldown_seconds_for_fraction(chunk_seconds, options.headroom_fraction);
                    if (cooldown_seconds > 0.0) {
                        std::this_thread::sleep_for(std::chrono::duration<double>(cooldown_seconds));
                        total_seconds += cooldown_seconds;
                    }
                }
            }
            return total_seconds;
        };

        run_pass(false, false);

        double total_seconds = 0.0;
        for (int repeat = 0; repeat < repeats; ++repeat) {
            total_seconds += run_pass(true, repeat + 1 < repeats);
        }
        const double latency_seconds = total_seconds / static_cast<double>(repeats);
        const double effective_gflops =
            (2.0 * static_cast<double>(output_rows) * static_cast<double>(options.cols))
            / std::max(latency_seconds, 1e-12) / 1.0e9;

        // Time the post-compute copy out of the shared MTLBuffer so the
        // raw_report can surface it. On unified-memory Apple Silicon this is a
        // local memcpy — not a PCIe transfer — and is typically negligible.
        const auto d2h_started = std::chrono::steady_clock::now();
        std::memcpy(host_output.data(), [output_buffer contents], output_bytes);
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
    }
}

}  // namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            const Options options = parse_args(argc, argv);
            const size_t output_rows = static_cast<size_t>(options.row_end - options.row_start);
            const size_t configured_row_chunk =
                options.row_chunk_size > 0 ? static_cast<size_t>(options.row_chunk_size) : output_rows;
            const size_t row_chunk_size = std::max<size_t>(1, std::min(output_rows, configured_row_chunk));

            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (device == nil) {
                throw std::runtime_error("MTLCreateSystemDefaultDevice returned nil");
            }

            id<MTLCommandQueue> command_queue = [device newCommandQueue];
            if (command_queue == nil) {
                throw std::runtime_error("failed to create Metal command queue");
            }

            // Memory-traffic model (FC-layer analogy): A is "weights", x is
            // "input activation", y is output. Compulsory traffic = one-touch
            // lower bound. On unified-memory Apple Silicon, PCIe bandwidth does
            // not apply; the D2H column captures the local memcpy out.
            const double flops_per_run =
                2.0 * static_cast<double>(output_rows) * static_cast<double>(options.cols);
            const size_t bytes_input =
                static_cast<size_t>(options.cols) * sizeof(float);
            const size_t bytes_weight =
                static_cast<size_t>(output_rows) * static_cast<size_t>(options.cols) * sizeof(float);
            const size_t bytes_output =
                output_rows * sizeof(float);
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
                        "[gemv metal %s %d/%d] candidate=%d/%d (trial %d/%d) "
                        "row_chunk_size=%d implementation=%s "
                        "compute=%.6fs d2h=%.6fs total=%.6fs "
                        "compute_gflops=%.3f effective_gflops=%.3f\n",
                        tr.phase.c_str(), global_index, global_total,
                        tr.candidate_index + 1, tr.candidate_total,
                        tr.trial_index_within_candidate + 1, tr.repeats_for_candidate,
                        tr.row_chunk_size, tr.implementation.c_str(),
                        tr.host_compute_seconds, tr.device_to_host_seconds, tr.total_wall_seconds,
                        compute_gflops, effective_gflops);
                fflush(stderr);
            };

            std::vector<float> best_output_values(output_rows, 0.0f);
            TrialMetrics metrics;

            if (options.verbose) {
                fprintf(stderr,
                        "[gemv metal plan] phase=autotune autotune_repeats=%d "
                        "row_chunk_size=%zu headroom_fraction=%.6f "
                        "device=\"%s\"\n",
                        options.autotune_repeats,
                        row_chunk_size,
                        options.headroom_fraction,
                        [[device name] UTF8String]);
                fflush(stderr);
            }

            metrics.autotune = run_mps_gemv(
                command_queue,
                options,
                best_output_values,
                options.autotune_repeats
            );
            {
                TrialRecord tr;
                tr.phase = "autotune";
                tr.candidate_index = 0;
                tr.candidate_total = 1;
                tr.trial_index_within_candidate = 0;
                tr.repeats_for_candidate = metrics.autotune.repeats;
                tr.implementation = metrics.implementation;
                tr.row_chunk_size = static_cast<int>(row_chunk_size);
                tr.host_compute_seconds = metrics.autotune.wall_clock_latency_seconds;
                tr.device_to_host_seconds = metrics.autotune.device_to_host_seconds;
                tr.total_wall_seconds =
                    metrics.autotune.wall_clock_latency_seconds
                    + metrics.autotune.device_to_host_seconds;
                tr.checksum = metrics.autotune.checksum;
                emit_verbose_trial(tr, 1, 1);
                trials.push_back(std::move(tr));
            }

            if (options.verbose) {
                fprintf(stderr,
                        "[gemv metal plan] phase=measurement measurement_repeats=%d\n",
                        options.measurement_repeats);
                fflush(stderr);
            }

            metrics.measurement = run_mps_gemv(
                command_queue,
                options,
                best_output_values,
                options.measurement_repeats
            );
            {
                TrialRecord tr;
                tr.phase = "measurement";
                tr.candidate_index = 0;
                tr.candidate_total = 1;
                tr.trial_index_within_candidate = 0;
                tr.repeats_for_candidate = metrics.measurement.repeats;
                tr.implementation = metrics.implementation;
                tr.row_chunk_size = static_cast<int>(row_chunk_size);
                tr.host_compute_seconds = metrics.measurement.wall_clock_latency_seconds;
                tr.device_to_host_seconds = metrics.measurement.device_to_host_seconds;
                tr.total_wall_seconds =
                    metrics.measurement.wall_clock_latency_seconds
                    + metrics.measurement.device_to_host_seconds;
                tr.checksum = metrics.measurement.checksum;
                emit_verbose_trial(tr, 1, 1);
                trials.push_back(std::move(tr));
            }

            if (!options.output_path.empty()) {
                write_float32_file(options.output_path, best_output_values);
            }

            const double compute_event_ms =
                metrics.measurement.wall_clock_latency_seconds
                * static_cast<double>(metrics.measurement.repeats) * 1000.0;

            std::cout << "{"
                      << "\"backend\":\"metal\","
                      << "\"implementation\":\"" << metrics.implementation << "\","
                      << "\"device_name\":\"" << escape_json(nsstring_to_string([device name])) << "\","
                      << "\"headroom_fraction\":" << std::fixed << std::setprecision(6)
                      << options.headroom_fraction << ","
                      << "\"row_chunk_size\":" << row_chunk_size << ","
                      << "\"autotune_repeats\":" << metrics.autotune.repeats << ","
                      << "\"measurement_repeats\":" << metrics.measurement.repeats << ","
                      << "\"trials_run\":1,"
                      << "\"compute_event_ms\":" << std::fixed << std::setprecision(6)
                      << compute_event_ms << ","
                      << "\"autotune_wall_clock_latency_seconds\":" << std::fixed << std::setprecision(9)
                      << metrics.autotune.wall_clock_latency_seconds << ","
                      << "\"autotune_effective_gflops\":" << std::fixed << std::setprecision(9)
                      << metrics.autotune.effective_gflops << ","
                      << "\"autotune_checksum\":\"" << metrics.autotune.checksum << "\","
                      << "\"measurement_wall_clock_latency_seconds\":" << std::fixed << std::setprecision(9)
                      << metrics.measurement.wall_clock_latency_seconds << ","
                      << "\"measurement_effective_gflops\":" << std::fixed << std::setprecision(9)
                      << metrics.measurement.effective_gflops << ","
                      << "\"measurement_checksum\":\"" << metrics.measurement.checksum << "\","
                      << "\"flops_per_run\":" << std::fixed << std::setprecision(1) << flops_per_run << ","
                      << "\"bytes_input\":" << bytes_input << ","
                      << "\"bytes_weight\":" << bytes_weight << ","
                      << "\"bytes_output\":" << bytes_output << ","
                      << "\"bytes_kernel_compulsory_memory_traffic\":" << bytes_kernel_compulsory << ","
                      << "\"notes_schema\":\"gemv Metal backend (MPSMatrixVectorMultiplication): host_prep_seconds is zero because the MTLBuffer uses MTLResourceStorageModeShared on unified-memory Apple Silicon (no separate H2D copy needed after read_exact_file fills the shared buffer); host_compute_seconds prefers [MTLCommandBuffer GPUStartTime/GPUEndTime] over chrono and includes the cooldown budget controlled by headroom_fraction; device_to_host_seconds is the chrono-timed post-compute local memcpy out of the shared buffer (not a PCIe transfer); host_postproc_seconds excludes checksum cost; pcie_h2d/d2h bandwidth fields are zero because Apple Silicon is unified-memory; memory_bandwidth model treats matrix A as weights, vector x as input activation, vector y as output, and uses compulsory one-touch DRAM traffic as a lower bound (real traffic >= compulsory).\","
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
                          << "\"implementation\":\"" << tr.implementation << "\","
                          << "\"row_chunk_size\":" << tr.row_chunk_size << ","
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
            std::cerr << exc.what() << std::endl;
            return 1;
        }
    }
}
