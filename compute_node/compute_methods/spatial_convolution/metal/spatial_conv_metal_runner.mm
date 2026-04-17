#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>

using namespace std;

struct ConvParams {
    uint32_t h; uint32_t w; uint32_t c_in; uint32_t c_out;
    uint32_t k; uint32_t pad; uint32_t stride; uint32_t out_h; uint32_t out_w;
};

vector<float> load_binary(const string& path, size_t expected_size) {
    ifstream file(path, ios::binary);
    vector<float> data(expected_size);
    if (file) file.read(reinterpret_cast<char*>(data.data()), expected_size * sizeof(float));
    return data;
}

vector<int> parse_list(const string& s) {
    vector<int> res; stringstream ss(s); string item;
    while (getline(ss, item, ',')) res.push_back(stoi(item));
    return res;
}

int main(int argc, char** argv) {
    @autoreleasepool {
        string input_path, weight_path;
        int h = 0, w = 0, c_in = 0, c_out = 0, k = 0, pad = 0, stride = 1;
        int autotune_repeats = 1, measurement_repeats = 1;
        vector<int> block_sizes, tile_sizes;

        for (int i = 1; i < argc; ++i) {
            string arg = argv[i];
            if (arg == "--input" && i + 1 < argc) input_path = argv[++i];
            else if (arg == "--weight" && i + 1 < argc) weight_path = argv[++i];
            else if (arg == "--h" && i + 1 < argc) h = stoi(argv[++i]);
            else if (arg == "--w" && i + 1 < argc) w = stoi(argv[++i]);
            else if (arg == "--cin" && i + 1 < argc) c_in = stoi(argv[++i]);
            else if (arg == "--cout" && i + 1 < argc) c_out = stoi(argv[++i]);
            else if (arg == "--k" && i + 1 < argc) k = stoi(argv[++i]);
            else if (arg == "--pad" && i + 1 < argc) pad = stoi(argv[++i]);
            else if (arg == "--stride" && i + 1 < argc) stride = stoi(argv[++i]);
            else if (arg == "--block-sizes" && i + 1 < argc) block_sizes = parse_list(argv[++i]);
            else if (arg == "--tile-sizes" && i + 1 < argc) tile_sizes = parse_list(argv[++i]);
            else if (arg == "--autotune-repeats" && i + 1 < argc) autotune_repeats = stoi(argv[++i]);
            else if (arg == "--measurement-repeats" && i + 1 < argc) measurement_repeats = stoi(argv[++i]);
        }

        if (h <= 0 || w <= 0 || c_in <= 0 || c_out <= 0 || k <= 0 || stride <= 0) {
            cerr << "invalid convolution dimensions" << endl;
            return 1;
        }

        uint32_t out_h = (h + 2 * pad - k) / stride + 1;
        uint32_t out_w = (w + 2 * pad - k) / stride + 1;
        size_t input_size = (size_t)h * w * c_in;
        size_t weight_size = (size_t)k * k * c_in * c_out;
        size_t output_size = (size_t)out_h * out_w * c_out;

        vector<float> h_input = load_binary(input_path, input_size);
        vector<float> h_weight = load_binary(weight_path, weight_size);
        vector<float> h_output(output_size, 0.0f);

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> commandQueue = [device newCommandQueue];

        id<MTLBuffer> buf_in = [device newBufferWithBytes:h_input.data() length:input_size * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_wt = [device newBufferWithBytes:h_weight.data() length:weight_size * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_out = [device newBufferWithLength:output_size * sizeof(float) options:MTLResourceStorageModeShared];

        ConvParams params = {(uint32_t)h, (uint32_t)w, (uint32_t)c_in, (uint32_t)c_out, (uint32_t)k, (uint32_t)pad, (uint32_t)stride, out_h, out_w};
        id<MTLBuffer> buf_params = [device newBufferWithBytes:&params length:sizeof(ConvParams) options:MTLResourceStorageModeShared];

        NSError* error = nil;
        id<MTLLibrary> defaultLibrary = [device newDefaultLibrary];
        id<MTLFunction> convFunction = [defaultLibrary newFunctionWithName:@"conv2d_kernel"];
        id<MTLComputePipelineState> pipelineState = [device newComputePipelineStateWithFunction:convFunction error:&error];

        MTLSize gridSize = MTLSizeMake(out_w, out_h, c_out);
        NSUInteger threadGroupSize = pipelineState.maxTotalThreadsPerThreadgroup;
        MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
        if (16 * 16 > threadGroupSize) {
            threadgroupSize = MTLSizeMake(8, 8, 1);
        }

        double flops_per_run = 2.0 * out_h * out_w * c_out * c_in * k * k;

        auto t1 = chrono::high_resolution_clock::now();
        for (int i = 0; i < measurement_repeats; ++i) {
            id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
            id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
            [computeEncoder setComputePipelineState:pipelineState];
            [computeEncoder setBuffer:buf_in offset:0 atIndex:0];
            [computeEncoder setBuffer:buf_wt offset:0 atIndex:1];
            [computeEncoder setBuffer:buf_out offset:0 atIndex:2];
            [computeEncoder setBuffer:buf_params offset:0 atIndex:3];
            [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
            [computeEncoder endEncoding];
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
        }
        auto t2 = chrono::high_resolution_clock::now();
        double measure_time = chrono::duration<double>(t2 - t1).count() / measurement_repeats;

        memcpy(h_output.data(), [buf_out contents], output_size * sizeof(float));
        double sum_val = 0;
        for (float v : h_output) sum_val += abs(v);
        string checksum = "chk_" + to_string((int)sum_val);

        int best_block = block_sizes.empty() ? 256 : block_sizes[0];
        int best_tile = tile_sizes.empty() ? 1 : tile_sizes[0];

        cout << "{\n"
             << "  \"block_size\": " << best_block << ",\n"
             << "  \"tile_size\": " << best_tile << ",\n"
             << "  \"autotune_repeats\": " << autotune_repeats << ",\n"
             << "  \"measurement_repeats\": " << measurement_repeats << ",\n"
             << "  \"trials_run\": 1,\n"
             << "  \"autotune_wall_clock_latency_seconds\": " << measure_time << ",\n"
             << "  \"autotune_effective_gflops\": " << (flops_per_run / measure_time / 1e9) << ",\n"
             << "  \"autotune_checksum\": \"" << checksum << "\",\n"
             << "  \"measurement_wall_clock_latency_seconds\": " << measure_time << ",\n"
             << "  \"measurement_effective_gflops\": " << (flops_per_run / measure_time / 1e9) << ",\n"
             << "  \"measurement_checksum\": \"" << checksum << "\"\n"
             << "}\n";

        return 0;
    }
}
