#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <dispatch/dispatch.h>
#include <mach-o/getsect.h>
#include <mach-o/ldsyms.h>

namespace {

struct Options {
    std::string library_path;
    std::string matrix_path;
    std::string vector_path;
    std::string output_path;
    int rows = 0;
    int cols = 0;
    std::vector<int> block_sizes;
    std::vector<int> tile_sizes;
    int autotune_repeats = 1;
    int measurement_repeats = 1;
};

struct PhaseMetrics {
    int repeats = 0;
    double wall_clock_latency_seconds = std::numeric_limits<double>::infinity();
    double effective_gflops = 0.0;
    std::string checksum;
};

struct TrialMetrics {
    int block_size = 0;
    int tile_size = 0;
    PhaseMetrics autotune;
    PhaseMetrics measurement;
};

struct alignas(16) FmvmParams {
    uint32_t rows = 0;
    uint32_t cols = 0;
    uint32_t simdgroup_count = 0;
    uint32_t reserved = 0;
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
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc) {
            throw std::runtime_error("missing value for command line flag");
        }

        const std::string key = argv[index];
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
        } else if (key == "--block-sizes") {
            options.block_sizes = parse_int_list(value);
        } else if (key == "--tile-sizes") {
            options.tile_sizes = parse_int_list(value);
        } else if (key == "--autotune-repeats") {
            options.autotune_repeats = std::stoi(value);
        } else if (key == "--measurement-repeats" || key == "--iteration-count") {
            // Task execution reuses the measurement loop but exposes the more
            // domain-specific name iteration-count to the runtime layer.
            options.measurement_repeats = std::stoi(value);
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
    if (options.block_sizes.empty() || options.tile_sizes.empty()) {
        throw std::runtime_error("block-size and tile-size candidate lists are required");
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

std::string function_name_for_tile_size(int tile_size) {
    switch (tile_size) {
        case 1:
            return "fmvm_row_major_tile_1";
        case 2:
            return "fmvm_row_major_tile_2";
        case 4:
            return "fmvm_row_major_tile_4";
        case 8:
            return "fmvm_row_major_tile_8";
        case 16:
            return "fmvm_row_major_tile_16";
        default:
            throw std::runtime_error("unsupported Metal tile size");
    }
}

dispatch_data_t embedded_metallib_data() {
    unsigned long section_size = 0;
    const uint8_t* section_bytes = getsectiondata(&_mh_execute_header, "__DATA", "__metallib", &section_size);
    if (section_bytes == nullptr || section_size == 0) {
        throw std::runtime_error("embedded metallib section was not found in the executable");
    }

    return dispatch_data_create(
        section_bytes,
        section_size,
        dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
        DISPATCH_DATA_DESTRUCTOR_DEFAULT
    );
}

}  // namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            const Options options = parse_args(argc, argv);
            const size_t matrix_bytes =
                static_cast<size_t>(options.rows) * static_cast<size_t>(options.cols) * sizeof(float);
            const size_t vector_bytes = static_cast<size_t>(options.cols) * sizeof(float);
            const size_t output_bytes = static_cast<size_t>(options.rows) * sizeof(float);

            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (device == nil) {
                throw std::runtime_error("MTLCreateSystemDefaultDevice returned nil");
            }

            id<MTLCommandQueue> command_queue = [device newCommandQueue];
            if (command_queue == nil) {
                throw std::runtime_error("failed to create Metal command queue");
            }

            NSError* error = nil;
            id<MTLLibrary> library = nil;
            if (!options.library_path.empty()) {
                NSString* library_path = [NSString stringWithUTF8String:options.library_path.c_str()];
                NSURL* library_url = [NSURL fileURLWithPath:library_path];
                library = [device newLibraryWithURL:library_url error:&error];
                if (library == nil) {
                    throw_metal_error("failed to load metallib", error);
                }
            } else {
                dispatch_data_t library_data = embedded_metallib_data();
                library = [device newLibraryWithData:library_data error:&error];
                if (library == nil) {
                    throw_metal_error("failed to load embedded metallib", error);
                }
            }

            std::map<int, id<MTLComputePipelineState>> pipelines;
            NSUInteger thread_execution_width = 0;
            NSUInteger max_threads_per_group = 0;
            for (const int tile_size : options.tile_sizes) {
                NSString* function_name =
                    [NSString stringWithUTF8String:function_name_for_tile_size(tile_size).c_str()];
                id<MTLFunction> function = [library newFunctionWithName:function_name];
                if (function == nil) {
                    throw std::runtime_error("failed to load Metal function for tile size");
                }

                id<MTLComputePipelineState> pipeline_state =
                    [device newComputePipelineStateWithFunction:function error:&error];
                if (pipeline_state == nil) {
                    throw_metal_error("failed to create Metal compute pipeline", error);
                }
                pipelines[tile_size] = pipeline_state;
                thread_execution_width = std::max(thread_execution_width, [pipeline_state threadExecutionWidth]);
                max_threads_per_group =
                    std::max(max_threads_per_group, [pipeline_state maxTotalThreadsPerThreadgroup]);
            }

            id<MTLBuffer> matrix_buffer = [device newBufferWithLength:matrix_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> vector_buffer = [device newBufferWithLength:vector_bytes options:MTLResourceStorageModeShared];
            id<MTLBuffer> output_buffer = [device newBufferWithLength:output_bytes options:MTLResourceStorageModeShared];
            if (matrix_buffer == nil || vector_buffer == nil || output_buffer == nil) {
                throw std::runtime_error("failed to allocate Metal buffers");
            }

            read_exact_file(options.matrix_path, [matrix_buffer contents], matrix_bytes);
            read_exact_file(options.vector_path, [vector_buffer contents], vector_bytes);

            std::vector<float> host_output(static_cast<size_t>(options.rows), 0.0f);
            std::vector<float> best_output_values(static_cast<size_t>(options.rows), 0.0f);
            TrialMetrics best_metrics;
            bool have_best_trial = false;
            int trials_run = 0;

            const NSUInteger max_threadgroup_memory = [device maxThreadgroupMemoryLength];
            const MTLSize threadgroups = MTLSizeMake(static_cast<NSUInteger>(options.rows), 1, 1);

            auto measure_config = [&](
                id<MTLComputePipelineState> pipeline_state,
                const FmvmParams& params,
                const MTLSize& threads_per_threadgroup,
                NSUInteger partial_bytes,
                int repeats
            ) -> PhaseMetrics {
                {
                    id<MTLCommandBuffer> warmup_buffer = [command_queue commandBuffer];
                    id<MTLComputeCommandEncoder> encoder = [warmup_buffer computeCommandEncoder];
                    [encoder setComputePipelineState:pipeline_state];
                    [encoder setBuffer:matrix_buffer offset:0 atIndex:0];
                    [encoder setBuffer:vector_buffer offset:0 atIndex:1];
                    [encoder setBuffer:output_buffer offset:0 atIndex:2];
                    [encoder setBytes:&params length:sizeof(params) atIndex:3];
                    [encoder setThreadgroupMemoryLength:partial_bytes atIndex:0];
                    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads_per_threadgroup];
                    [encoder endEncoding];
                    ensure_completed(warmup_buffer, "warmup Metal command buffer failed");
                }

                id<MTLCommandBuffer> timed_buffer = [command_queue commandBuffer];
                id<MTLComputeCommandEncoder> encoder = [timed_buffer computeCommandEncoder];
                [encoder setComputePipelineState:pipeline_state];
                [encoder setBuffer:matrix_buffer offset:0 atIndex:0];
                [encoder setBuffer:vector_buffer offset:0 atIndex:1];
                [encoder setBuffer:output_buffer offset:0 atIndex:2];
                [encoder setBytes:&params length:sizeof(params) atIndex:3];
                [encoder setThreadgroupMemoryLength:partial_bytes atIndex:0];
                for (int repeat = 0; repeat < repeats; ++repeat) {
                    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads_per_threadgroup];
                }
                [encoder endEncoding];

                const auto started = std::chrono::steady_clock::now();
                ensure_completed(timed_buffer, "timed Metal command buffer failed");
                const auto finished = std::chrono::steady_clock::now();

                const double total_seconds = std::chrono::duration<double>(finished - started).count();
                const double latency_seconds = total_seconds / static_cast<double>(repeats);
                const double effective_gflops =
                    (2.0 * static_cast<double>(options.rows) * static_cast<double>(options.cols))
                    / std::max(latency_seconds, 1e-12) / 1.0e9;

                std::memcpy(
                    host_output.data(),
                    [output_buffer contents],
                    host_output.size() * sizeof(float));

                PhaseMetrics metrics;
                metrics.repeats = repeats;
                metrics.wall_clock_latency_seconds = latency_seconds;
                metrics.effective_gflops = effective_gflops;
                metrics.checksum = fnv1a64_checksum(host_output);
                return metrics;
            };

            for (const int block_size : options.block_sizes) {
                if (block_size <= 0) {
                    continue;
                }
                if (static_cast<NSUInteger>(block_size) > max_threads_per_group) {
                    continue;
                }
                if (thread_execution_width > 0 && (static_cast<NSUInteger>(block_size) % thread_execution_width) != 0) {
                    continue;
                }

                const NSUInteger simdgroup_count =
                    (static_cast<NSUInteger>(block_size) + thread_execution_width - 1) / thread_execution_width;
                const NSUInteger partial_bytes = simdgroup_count * sizeof(float);
                if (partial_bytes > max_threadgroup_memory) {
                    continue;
                }

                const MTLSize threads_per_threadgroup = MTLSizeMake(static_cast<NSUInteger>(block_size), 1, 1);

                for (const int tile_size : options.tile_sizes) {
                    if (tile_size <= 0) {
                        continue;
                    }
                    ++trials_run;
                    auto pipeline_iterator = pipelines.find(tile_size);
                    if (pipeline_iterator == pipelines.end()) {
                        continue;
                    }
                    id<MTLComputePipelineState> pipeline_state = pipeline_iterator->second;

                    const FmvmParams params = {
                        static_cast<uint32_t>(options.rows),
                        static_cast<uint32_t>(options.cols),
                        static_cast<uint32_t>(simdgroup_count),
                        0u,
                    };

                    const PhaseMetrics autotune_metrics = measure_config(
                        pipeline_state,
                        params,
                        threads_per_threadgroup,
                        partial_bytes,
                        options.autotune_repeats
                    );

                    if (!have_best_trial || autotune_metrics.wall_clock_latency_seconds < best_metrics.autotune.wall_clock_latency_seconds) {
                        have_best_trial = true;
                        best_metrics.block_size = block_size;
                        best_metrics.tile_size = tile_size;
                        best_metrics.autotune = autotune_metrics;
                    }
                }
            }

            if (!have_best_trial) {
                throw std::runtime_error("Metal benchmark ran zero valid trials");
            }

            const NSUInteger best_simdgroup_count =
                (static_cast<NSUInteger>(best_metrics.block_size) + thread_execution_width - 1) / thread_execution_width;
            const NSUInteger best_partial_bytes = best_simdgroup_count * sizeof(float);
            const MTLSize best_threads_per_threadgroup = MTLSizeMake(static_cast<NSUInteger>(best_metrics.block_size), 1, 1);
            const FmvmParams best_params = {
                static_cast<uint32_t>(options.rows),
                static_cast<uint32_t>(options.cols),
                static_cast<uint32_t>(best_simdgroup_count),
                0u,
            };
            auto best_pipeline_iterator = pipelines.find(best_metrics.tile_size);
            if (best_pipeline_iterator == pipelines.end()) {
                throw std::runtime_error("best Metal pipeline was not cached");
            }
            best_metrics.measurement = measure_config(
                best_pipeline_iterator->second,
                best_params,
                best_threads_per_threadgroup,
                best_partial_bytes,
                options.measurement_repeats
            );
            best_output_values = host_output;

            if (!options.output_path.empty()) {
                write_float32_file(options.output_path, best_output_values);
            }

            std::cout << "{"
                      << "\"backend\":\"metal\","
                      << "\"device_name\":\"" << escape_json(nsstring_to_string([device name])) << "\","
                      << "\"thread_execution_width\":" << static_cast<unsigned long long>(thread_execution_width) << ","
                      << "\"max_total_threads_per_threadgroup\":"
                      << static_cast<unsigned long long>(max_threads_per_group) << ","
                      << "\"block_size\":" << best_metrics.block_size << ","
                      << "\"tile_size\":" << best_metrics.tile_size << ","
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
            std::cerr << exc.what() << std::endl;
            return 1;
        }
    }
}
