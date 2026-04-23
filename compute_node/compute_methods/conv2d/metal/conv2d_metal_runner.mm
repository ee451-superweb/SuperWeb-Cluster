#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <algorithm>
#include <chrono>
#include <cmath>
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
#include <thread>
#include <utility>
#include <vector>

@interface Conv2DMPSChunkContext : NSObject
@property(nonatomic) NSUInteger channelOffset;
@property(nonatomic) NSUInteger channelCount;
@property(nonatomic, strong) MPSGraph* graph;
@property(nonatomic, strong) MPSGraphTensor* inputTensor;
@property(nonatomic, strong) MPSGraphTensor* weightTensor;
@property(nonatomic, strong) MPSGraphTensor* outputTensor;
@end

@implementation Conv2DMPSChunkContext
@end

namespace {

enum class RunnerMode { Dispatch, Benchmark };

struct Options {
    std::string library_path;
    std::string input_path;
    std::string weight_path;
    std::string output_path;
    int h = 0;
    int w = 0;
    int c_in = 0;
    int c_out = 0;
    int k = 0;
    int pad = 0;
    int stride = 1;
    std::vector<int> block_sizes;
    std::vector<int> tile_sizes;
    bool include_preparation_in_metrics = false;
    double headroom_fraction = 1.0;
    int output_channel_batch = 0;
    int autotune_repeats = 1;
    int measurement_repeats = 1;
    RunnerMode mode = RunnerMode::Dispatch;
    bool verbose = false;
};

struct PhaseMetrics {
    int repeats = 0;
    double wall_clock_latency_seconds = std::numeric_limits<double>::infinity();
    double effective_gflops = 0.0;
    std::string checksum;
};

struct TrialMetrics {
    std::string implementation = "mpsgraph";
    PhaseMetrics autotune;
    PhaseMetrics measurement;
};

struct TrialRecord {
    std::string phase;
    int candidate_index = 0;
    int candidate_total = 0;
    int trial_index_within_candidate = 0;
    int repeats_for_candidate = 0;
    int output_channel_batch = 0;
    double host_prep_seconds = 0.0;
    double host_compute_seconds = 0.0;
    double device_to_host_seconds = 0.0;
    double host_postproc_seconds = 0.0;
    double total_wall_seconds = 0.0;
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

bool parse_bool_flag(const std::string& text) {
    if (text == "1" || text == "true" || text == "TRUE" || text == "yes" || text == "YES") {
        return true;
    }
    if (text == "0" || text == "false" || text == "FALSE" || text == "no" || text == "NO") {
        return false;
    }
    throw std::runtime_error("expected boolean flag value");
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
        index += 2;
        if (key == "--library") {
            options.library_path = value;
        } else if (key == "--input") {
            options.input_path = value;
        } else if (key == "--weight") {
            options.weight_path = value;
        } else if (key == "--output") {
            options.output_path = value;
        } else if (key == "--h") {
            options.h = std::stoi(value);
        } else if (key == "--w") {
            options.w = std::stoi(value);
        } else if (key == "--cin") {
            options.c_in = std::stoi(value);
        } else if (key == "--cout") {
            options.c_out = std::stoi(value);
        } else if (key == "--k") {
            options.k = std::stoi(value);
        } else if (key == "--pad") {
            options.pad = std::stoi(value);
        } else if (key == "--stride") {
            options.stride = std::stoi(value);
        } else if (key == "--block-sizes") {
            options.block_sizes = parse_int_list(value);
        } else if (key == "--tile-sizes") {
            options.tile_sizes = parse_int_list(value);
        } else if (key == "--include-preparation-in-metrics") {
            options.include_preparation_in_metrics = parse_bool_flag(value);
        } else if (key == "--headroom-fraction") {
            options.headroom_fraction = std::stod(value);
        } else if (key == "--output-channel-batch") {
            options.output_channel_batch = std::stoi(value);
        } else if (key == "--autotune-repeats") {
            options.autotune_repeats = std::stoi(value);
        } else if (key == "--measurement-repeats" || key == "--iteration-count") {
            options.measurement_repeats = std::stoi(value);
        } else if (key == "--mode") {
            if (value == "dispatch") {
                options.mode = RunnerMode::Dispatch;
            } else if (value == "benchmark") {
                options.mode = RunnerMode::Benchmark;
            } else {
                throw std::runtime_error("unknown --mode value: " + value);
            }
        } else {
            throw std::runtime_error("unknown flag: " + key);
        }
    }

    if (options.input_path.empty() || options.weight_path.empty()) {
        throw std::runtime_error("input and weight paths are required");
    }
    if (
        options.h <= 0 || options.w <= 0 || options.c_in <= 0 || options.c_out <= 0 || options.k <= 0
        || options.stride <= 0
    ) {
        throw std::runtime_error("invalid convolution dimensions");
    }
    if (options.block_sizes.empty()) {
        options.block_sizes.push_back(256);
    }
    if (options.tile_sizes.empty()) {
        options.tile_sizes.push_back(16);
    }
    if (!(options.headroom_fraction > 0.0 && options.headroom_fraction <= 1.0)) {
        throw std::runtime_error("headroom fraction must be within (0.0, 1.0]");
    }
    if (options.output_channel_batch < 0) {
        throw std::runtime_error("output channel batch must be non-negative");
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

std::vector<float> transpose_weight_for_hwio(
    const std::vector<float>& weight_src,
    int c_out,
    int k,
    int c_in
) {
    std::vector<float> weight_dst(weight_src.size(), 0.0f);
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

MPSShape* make_shape(std::initializer_list<NSUInteger> dims) {
    NSMutableArray<NSNumber*>* shape = [NSMutableArray arrayWithCapacity:dims.size()];
    for (const NSUInteger dim : dims) {
        [shape addObject:@(dim)];
    }
    return shape;
}

double cooldown_seconds_for_fraction(double active_seconds, double fraction) {
    if (!(fraction > 0.0) || fraction >= 1.0 || active_seconds <= 0.0) {
        return 0.0;
    }
    return active_seconds * ((1.0 / fraction) - 1.0);
}

Conv2DMPSChunkContext* build_chunk_context(
    const Options& options,
    NSUInteger channel_offset,
    NSUInteger channel_count
) {
    MPSGraph* graph = [MPSGraph new];
    graph.options = MPSGraphOptionsSynchronizeResults;

    MPSShape* input_shape = make_shape({
        1,
        static_cast<NSUInteger>(options.h),
        static_cast<NSUInteger>(options.w),
        static_cast<NSUInteger>(options.c_in),
    });
    MPSShape* weight_shape = make_shape({
        static_cast<NSUInteger>(options.k),
        static_cast<NSUInteger>(options.k),
        static_cast<NSUInteger>(options.c_in),
        static_cast<NSUInteger>(options.c_out),
    });

    MPSGraphTensor* input_tensor =
        [graph placeholderWithShape:input_shape dataType:MPSDataTypeFloat32 name:@"input"];
    MPSGraphTensor* weight_tensor =
        [graph placeholderWithShape:weight_shape dataType:MPSDataTypeFloat32 name:@"weight"];
    MPSGraphTensor* sliced_weight =
        [graph sliceTensor:weight_tensor
                 dimension:3
                     start:static_cast<NSInteger>(channel_offset)
                    length:static_cast<NSInteger>(channel_count)
                      name:@"weight_slice"];

    MPSGraphConvolution2DOpDescriptor* descriptor =
        [MPSGraphConvolution2DOpDescriptor descriptorWithStrideInX:static_cast<NSUInteger>(options.stride)
                                                         strideInY:static_cast<NSUInteger>(options.stride)
                                                   dilationRateInX:1
                                                   dilationRateInY:1
                                                            groups:1
                                                       paddingLeft:static_cast<NSUInteger>(options.pad)
                                                      paddingRight:static_cast<NSUInteger>(options.pad)
                                                        paddingTop:static_cast<NSUInteger>(options.pad)
                                                     paddingBottom:static_cast<NSUInteger>(options.pad)
                                                      paddingStyle:MPSGraphPaddingStyleExplicit
                                                        dataLayout:MPSGraphTensorNamedDataLayoutNHWC
                                                     weightsLayout:MPSGraphTensorNamedDataLayoutHWIO];
    if (descriptor == nil) {
        throw std::runtime_error("failed to construct MPSGraph convolution descriptor");
    }
    MPSGraphTensor* output_tensor =
        [graph convolution2DWithSourceTensor:input_tensor
                               weightsTensor:sliced_weight
                                  descriptor:descriptor
                                        name:@"output"];

    Conv2DMPSChunkContext* context = [Conv2DMPSChunkContext new];
    context.channelOffset = channel_offset;
    context.channelCount = channel_count;
    context.graph = graph;
    context.inputTensor = input_tensor;
    context.weightTensor = weight_tensor;
    context.outputTensor = output_tensor;
    return context;
}

void scatter_output_chunk(
    const std::vector<float>& chunk_output,
    std::vector<float>& host_output,
    size_t spatial_positions,
    size_t total_channels,
    size_t channel_offset,
    size_t channel_count
) {
    for (size_t position = 0; position < spatial_positions; ++position) {
        const float* chunk_src = chunk_output.data() + position * channel_count;
        float* dst = host_output.data() + position * total_channels + channel_offset;
        std::memcpy(dst, chunk_src, channel_count * sizeof(float));
    }
}

PhaseMetrics run_mpsgraph_conv2d(
    id<MTLCommandQueue> command_queue,
    const Options& options,
    const std::vector<float>& transposed_weight,
    std::vector<float>& host_output,
    int repeats
) {
    @autoreleasepool {
        const auto phase_started = std::chrono::steady_clock::now();
        const int out_h_int = (options.h + 2 * options.pad - options.k) / options.stride + 1;
        const int out_w_int = (options.w + 2 * options.pad - options.k) / options.stride + 1;
        if (out_h_int <= 0 || out_w_int <= 0) {
            throw std::runtime_error("invalid convolution output shape");
        }

        const size_t input_values =
            static_cast<size_t>(options.h) * static_cast<size_t>(options.w) * static_cast<size_t>(options.c_in);
        const size_t weight_values =
            static_cast<size_t>(options.k) * static_cast<size_t>(options.k) * static_cast<size_t>(options.c_in)
            * static_cast<size_t>(options.c_out);
        const size_t spatial_positions = static_cast<size_t>(out_h_int) * static_cast<size_t>(out_w_int);
        const size_t output_values = spatial_positions * static_cast<size_t>(options.c_out);
        const size_t configured_output_batch =
            options.output_channel_batch > 0 ? static_cast<size_t>(options.output_channel_batch) : static_cast<size_t>(options.c_out);
        const size_t output_channel_batch =
            std::max<size_t>(1, std::min(static_cast<size_t>(options.c_out), configured_output_batch));

        id<MTLDevice> device = [command_queue device];
        id<MTLBuffer> input_buffer =
            [device newBufferWithLength:input_values * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> weight_buffer =
            [device newBufferWithLength:weight_values * sizeof(float) options:MTLResourceStorageModeShared];
        if (input_buffer == nil || weight_buffer == nil) {
            throw std::runtime_error("failed to allocate MPSGraph input buffers");
        }

        read_exact_file(options.input_path, [input_buffer contents], input_values * sizeof(float));
        std::memcpy([weight_buffer contents], transposed_weight.data(), weight_values * sizeof(float));

        MPSShape* input_shape = make_shape({
            1,
            static_cast<NSUInteger>(options.h),
            static_cast<NSUInteger>(options.w),
            static_cast<NSUInteger>(options.c_in),
        });
        MPSShape* weight_shape = make_shape({
            static_cast<NSUInteger>(options.k),
            static_cast<NSUInteger>(options.k),
            static_cast<NSUInteger>(options.c_in),
            static_cast<NSUInteger>(options.c_out),
        });
        MPSGraphTensorData* input_data =
            [[MPSGraphTensorData alloc] initWithMTLBuffer:input_buffer
                                                    shape:input_shape
                                                 dataType:MPSDataTypeFloat32];
        MPSGraphTensorData* weight_data =
            [[MPSGraphTensorData alloc] initWithMTLBuffer:weight_buffer
                                                    shape:weight_shape
                                                 dataType:MPSDataTypeFloat32];
        if (input_data == nil || weight_data == nil) {
            throw std::runtime_error("failed to create MPSGraph tensor data");
        }

        NSMutableArray<Conv2DMPSChunkContext*>* chunk_contexts = [NSMutableArray array];
        for (size_t channel_offset = 0; channel_offset < static_cast<size_t>(options.c_out); channel_offset += output_channel_batch) {
            const size_t channel_count =
                std::min(output_channel_batch, static_cast<size_t>(options.c_out) - channel_offset);
            [chunk_contexts addObject:build_chunk_context(
                options,
                static_cast<NSUInteger>(channel_offset),
                static_cast<NSUInteger>(channel_count)
            )];
        }
        if ([chunk_contexts count] == 0) {
            throw std::runtime_error("failed to create any conv2d chunk contexts");
        }

        for (Conv2DMPSChunkContext* context in chunk_contexts) {
            NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
                context.inputTensor: input_data,
                context.weightTensor: weight_data,
            };
            (void)[context.graph runWithMTLCommandQueue:command_queue
                                                  feeds:feeds
                                          targetTensors:@[ context.outputTensor ]
                                       targetOperations:nil];
        }

        auto run_pass = [&](bool include_cooldown, bool cooldown_after_final_chunk, bool capture_output) -> double {
            double total_seconds = 0.0;
            if (capture_output) {
                std::fill(host_output.begin(), host_output.end(), 0.0f);
            }

            for (NSUInteger chunk_index = 0; chunk_index < [chunk_contexts count]; ++chunk_index) {
                Conv2DMPSChunkContext* context = chunk_contexts[chunk_index];
                NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
                    context.inputTensor: input_data,
                    context.weightTensor: weight_data,
                };

                const auto started = std::chrono::steady_clock::now();
                NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results =
                    [context.graph runWithMTLCommandQueue:command_queue
                                                    feeds:feeds
                                            targetTensors:@[ context.outputTensor ]
                                         targetOperations:nil];
                const auto finished = std::chrono::steady_clock::now();
                if (results == nil) {
                    throw std::runtime_error("MPSGraph returned no results");
                }

                const double chunk_seconds = std::chrono::duration<double>(finished - started).count();
                total_seconds += chunk_seconds;

                if (capture_output) {
                    MPSGraphTensorData* output_data = results[context.outputTensor];
                    if (output_data == nil) {
                        throw std::runtime_error("MPSGraph output tensor data was missing");
                    }
                    MPSNDArray* output_array = [output_data mpsndarray];
                    if (output_array == nil) {
                        throw std::runtime_error("failed to materialize MPSGraph output as an MPSNDArray");
                    }

                    const size_t chunk_channel_count = static_cast<size_t>(context.channelCount);
                    std::vector<float> chunk_output(spatial_positions * chunk_channel_count, 0.0f);
                    [output_array readBytes:chunk_output.data() strideBytes:nil];
                    scatter_output_chunk(
                        chunk_output,
                        host_output,
                        spatial_positions,
                        static_cast<size_t>(options.c_out),
                        static_cast<size_t>(context.channelOffset),
                        chunk_channel_count
                    );
                }

                const bool final_chunk = (chunk_index + 1) >= [chunk_contexts count];
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

        double total_seconds = 0.0;
        for (int repeat = 0; repeat < repeats; ++repeat) {
            total_seconds += run_pass(true, repeat + 1 < repeats, repeat + 1 == repeats);
        }

        double latency_seconds = total_seconds / static_cast<double>(repeats);
        if (options.include_preparation_in_metrics) {
            const auto phase_finished = std::chrono::steady_clock::now();
            const double phase_wall_clock_seconds =
                std::chrono::duration<double>(phase_finished - phase_started).count();
            latency_seconds = phase_wall_clock_seconds / static_cast<double>(repeats);
        }
        const double flops_per_run =
            2.0 * static_cast<double>(out_h_int) * static_cast<double>(out_w_int) * static_cast<double>(options.c_out)
            * static_cast<double>(options.c_in) * static_cast<double>(options.k) * static_cast<double>(options.k);

        PhaseMetrics metrics;
        metrics.repeats = repeats;
        metrics.wall_clock_latency_seconds = latency_seconds;
        metrics.effective_gflops = flops_per_run / std::max(latency_seconds, 1e-12) / 1.0e9;
        metrics.checksum = fnv1a64_checksum(host_output);
        return metrics;
    }
}

}  // namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            if (@available(macOS 11.0, *)) {
                const Options options = parse_args(argc, argv);
                const int out_h_int = (options.h + 2 * options.pad - options.k) / options.stride + 1;
                const int out_w_int = (options.w + 2 * options.pad - options.k) / options.stride + 1;
                if (out_h_int <= 0 || out_w_int <= 0) {
                    throw std::runtime_error("invalid convolution output shape");
                }

                const size_t raw_weight_values =
                    static_cast<size_t>(options.c_out) * static_cast<size_t>(options.k) * static_cast<size_t>(options.k)
                    * static_cast<size_t>(options.c_in);
                std::vector<float> weight_values(raw_weight_values, 0.0f);
                read_exact_file(options.weight_path, weight_values.data(), raw_weight_values * sizeof(float));
                const std::vector<float> transposed_weight =
                    transpose_weight_for_hwio(weight_values, options.c_out, options.k, options.c_in);

                const size_t output_values =
                    static_cast<size_t>(out_h_int) * static_cast<size_t>(out_w_int) * static_cast<size_t>(options.c_out);
                const size_t configured_output_batch =
                    options.output_channel_batch > 0 ? static_cast<size_t>(options.output_channel_batch) : static_cast<size_t>(options.c_out);
                const size_t output_channel_batch =
                    std::max<size_t>(1, std::min(static_cast<size_t>(options.c_out), configured_output_batch));

                std::vector<float> host_output(output_values, 0.0f);
                std::vector<float> best_output_values(output_values, 0.0f);
                TrialMetrics best_metrics;

                id<MTLDevice> device = MTLCreateSystemDefaultDevice();
                if (device == nil) {
                    throw std::runtime_error("MTLCreateSystemDefaultDevice returned nil");
                }
                id<MTLCommandQueue> command_queue = [device newCommandQueue];
                if (command_queue == nil) {
                    throw std::runtime_error("failed to create Metal command queue");
                }

                std::vector<TrialRecord> trial_records;
                trial_records.reserve(2);

                const double flops_per_run_plan =
                    2.0 * static_cast<double>(out_h_int) * static_cast<double>(out_w_int) *
                    static_cast<double>(options.c_out) * static_cast<double>(options.c_in) *
                    static_cast<double>(options.k) * static_cast<double>(options.k);
                const long long bytes_input_plan =
                    static_cast<long long>(options.h) * options.w * options.c_in * 4;
                const long long bytes_weight_plan =
                    static_cast<long long>(options.c_out) * options.c_in * options.k * options.k * 4;
                const long long bytes_output_plan =
                    static_cast<long long>(out_h_int) * out_w_int * options.c_out * 4;
                const long long bytes_kernel_compulsory_plan =
                    bytes_input_plan + bytes_weight_plan + bytes_output_plan;

                if (options.mode == RunnerMode::Benchmark) {
                    if (options.verbose) {
                        std::fprintf(stderr,
                            "[conv2d metal plan] phase=autotune autotune_repeats=%d output_channel_batch=%zu\n",
                            options.autotune_repeats, output_channel_batch);
                        std::fflush(stderr);
                    }

                    best_metrics.autotune = run_mpsgraph_conv2d(
                        command_queue,
                        options,
                        transposed_weight,
                        host_output,
                        options.autotune_repeats
                    );

                    {
                        TrialRecord record;
                        record.phase = "autotune";
                        record.candidate_index = 0;
                        record.candidate_total = 1;
                        record.trial_index_within_candidate = 0;
                        record.repeats_for_candidate = options.autotune_repeats;
                        record.output_channel_batch = static_cast<int>(output_channel_batch);
                        record.host_prep_seconds = 0.0;
                        record.host_compute_seconds = best_metrics.autotune.wall_clock_latency_seconds;
                        record.device_to_host_seconds = 0.0;
                        record.host_postproc_seconds = 0.0;
                        record.total_wall_seconds = best_metrics.autotune.wall_clock_latency_seconds;
                        trial_records.push_back(std::move(record));
                    }

                    if (options.verbose) {
                        std::fprintf(stderr,
                            "[conv2d metal autotune 1/1] output_channel_batch=%zu repeats=%d "
                            "per_run=%.6fs effective_gflops=%.3f\n",
                            output_channel_batch, options.autotune_repeats,
                            best_metrics.autotune.wall_clock_latency_seconds,
                            best_metrics.autotune.effective_gflops);
                        std::fprintf(stderr,
                            "[conv2d metal plan] phase=measurement measurement_repeats=%d\n",
                            options.measurement_repeats);
                        std::fflush(stderr);
                    }

                    best_metrics.measurement = run_mpsgraph_conv2d(
                        command_queue,
                        options,
                        transposed_weight,
                        host_output,
                        options.measurement_repeats
                    );

                    {
                        TrialRecord record;
                        record.phase = "measurement";
                        record.candidate_index = 0;
                        record.candidate_total = 1;
                        record.trial_index_within_candidate = 0;
                        record.repeats_for_candidate = options.measurement_repeats;
                        record.output_channel_batch = static_cast<int>(output_channel_batch);
                        record.host_prep_seconds = 0.0;
                        record.host_compute_seconds = best_metrics.measurement.wall_clock_latency_seconds;
                        record.device_to_host_seconds = 0.0;
                        record.host_postproc_seconds = 0.0;
                        record.total_wall_seconds = best_metrics.measurement.wall_clock_latency_seconds;
                        trial_records.push_back(std::move(record));
                    }

                    if (options.verbose) {
                        std::fprintf(stderr,
                            "[conv2d metal measurement 1/1] output_channel_batch=%zu repeats=%d "
                            "per_run=%.6fs effective_gflops=%.3f\n",
                            output_channel_batch, options.measurement_repeats,
                            best_metrics.measurement.wall_clock_latency_seconds,
                            best_metrics.measurement.effective_gflops);
                        std::fflush(stderr);
                    }
                } else {
                    // Dispatch: single MPSGraph submission using the
                    // benchmark-pinned config. The timing window includes
                    // D2H + scatter because MPSGraph runs to completion
                    // inside run_mpsgraph_conv2d's std::chrono bracket,
                    // which is the best we have without MTLCounterSampleBuffer
                    // support. compute_event_ms stays comparable to the
                    // benchmark's measurement latency.
                    if (options.verbose) {
                        std::fprintf(stderr,
                            "[conv2d metal plan] phase=dispatch output_channel_batch=%zu\n",
                            output_channel_batch);
                        std::fflush(stderr);
                    }

                    best_metrics.measurement = run_mpsgraph_conv2d(
                        command_queue,
                        options,
                        transposed_weight,
                        host_output,
                        1
                    );

                    {
                        TrialRecord record;
                        record.phase = "dispatch";
                        record.candidate_index = 0;
                        record.candidate_total = 1;
                        record.trial_index_within_candidate = 0;
                        record.repeats_for_candidate = 1;
                        record.output_channel_batch = static_cast<int>(output_channel_batch);
                        record.host_prep_seconds = 0.0;
                        record.host_compute_seconds = best_metrics.measurement.wall_clock_latency_seconds;
                        record.device_to_host_seconds = 0.0;
                        record.host_postproc_seconds = 0.0;
                        record.total_wall_seconds = best_metrics.measurement.wall_clock_latency_seconds;
                        trial_records.push_back(std::move(record));
                    }

                    if (options.verbose) {
                        std::fprintf(stderr,
                            "[conv2d metal dispatch 1/1] output_channel_batch=%zu "
                            "per_run=%.6fs effective_gflops=%.3f\n",
                            output_channel_batch,
                            best_metrics.measurement.wall_clock_latency_seconds,
                            best_metrics.measurement.effective_gflops);
                        std::fflush(stderr);
                    }
                }

                best_output_values = host_output;

                if (!options.output_path.empty()) {
                    write_float32_file(options.output_path, best_output_values);
                }

                const char* mode_str =
                    (options.mode == RunnerMode::Benchmark) ? "benchmark" : "dispatch";
                const double compute_event_ms_value =
                    best_metrics.measurement.wall_clock_latency_seconds * 1000.0;

                std::cout << "{"
                          << "\"mode\":\"" << mode_str << "\","
                          << "\"backend\":\"metal\","
                          << "\"implementation\":\"mpsgraph\","
                          << "\"device_name\":\"" << escape_json(nsstring_to_string([device name])) << "\","
                          << "\"headroom_fraction\":" << std::fixed << std::setprecision(6)
                          << options.headroom_fraction << ","
                          << "\"output_channel_batch\":" << output_channel_batch << ","
                          << "\"autotune_repeats\":" << best_metrics.autotune.repeats << ","
                          << "\"measurement_repeats\":" << best_metrics.measurement.repeats << ","
                          << "\"trials_run\":1,"
                          << "\"compute_event_ms\":" << std::fixed << std::setprecision(6)
                          << compute_event_ms_value << ","
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
                          << "\"flops_per_run\":" << std::fixed << std::setprecision(1) << flops_per_run_plan << ","
                          << "\"bytes_input\":" << bytes_input_plan << ","
                          << "\"bytes_weight\":" << bytes_weight_plan << ","
                          << "\"bytes_output\":" << bytes_output_plan << ","
                          << "\"bytes_kernel_compulsory_memory_traffic\":" << bytes_kernel_compulsory_plan << ","
                          << "\"notes_schema\":\"Metal backend (MPSGraph): host_prep/device_to_host/host_postproc are zero in this schema. host_compute_seconds is the amortized per-run wall clock over autotune_repeats (or measurement_repeats) MPSGraph submissions; H2D is done before the timing window and D2H + checksum happen after it. Per-phase GPU timing split is deferred pending MTLCounterSampleBuffer integration; memory_bandwidth model assumes perfect DRAM reuse (real traffic >= compulsory).\","
                          << "\"trials\":[";
                for (size_t i = 0; i < trial_records.size(); ++i) {
                    const auto& tr = trial_records[i];
                    const double compute_gflops = tr.host_compute_seconds > 0.0
                        ? (flops_per_run_plan / tr.host_compute_seconds / 1e9) : 0.0;
                    const double effective_gflops = tr.total_wall_seconds > 0.0
                        ? (flops_per_run_plan / tr.total_wall_seconds / 1e9) : 0.0;
                    const double kernel_bandwidth_gibps = tr.host_compute_seconds > 0.0
                        ? (static_cast<double>(bytes_kernel_compulsory_plan) / tr.host_compute_seconds / (1024.0 * 1024.0 * 1024.0))
                        : 0.0;
                    std::cout << (i == 0 ? "" : ",")
                              << "{"
                              << "\"phase\":\"" << tr.phase << "\","
                              << "\"candidate_index\":" << tr.candidate_index << ","
                              << "\"candidate_total\":" << tr.candidate_total << ","
                              << "\"trial_index_within_candidate\":" << tr.trial_index_within_candidate << ","
                              << "\"repeats_for_candidate\":" << tr.repeats_for_candidate << ","
                              << "\"output_channel_batch\":" << tr.output_channel_batch << ","
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
                              << "\"kernel_memory_bandwidth_gibps_compulsory_lower_bound_model\":" << kernel_bandwidth_gibps
                              << "}";
                }
                std::cout << "]}" << std::endl;
                return 0;
            }

            std::cerr << "MPSGraph conv2d requires macOS 11.0 or later" << std::endl;
            return 1;
        } catch (const std::exception& exc) {
            std::cerr << exc.what() << std::endl;
            return 1;
        }
    }
}
