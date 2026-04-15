/*
 * CPU Conv2D Benchmark Runner for macOS
 *
 * Key optimizations vs original:
 *   1. Weight index bug FIXED: file is [Cout,K,K,Cin], now transposed to [K,K,Cin,Cout]
 *   2. Loop order: kh -> kw -> ic -> oc (input reuse + auto-vectorizable inner loop)
 *   3. Real autotune: sweeps all worker counts, picks fastest
 *   4. Supports --start-oc / --end-oc for distributed slice computation
 *
 * Compile: clang++ -std=c++20 -O3 -ffast-math -pthread -o fmvm_cpu_macos fmvm_cpu_macos.cpp
 */

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <thread>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <functional>

using namespace std;

// ─── Helpers ──────────────────────────────────────────────────────────────────

vector<int> parse_list(const string& s) {
    vector<int> res;
    stringstream ss(s);
    string item;
    while (getline(ss, item, ',')) {
        if (!item.empty()) res.push_back(stoi(item));
    }
    return res;
}

vector<float> load_binary(const string& path, size_t expected_size) {
    ifstream file(path, ios::binary);
    if (!file) {
        cerr << "Error: Cannot open " << path << endl;
        exit(1);
    }
    vector<float> data(expected_size);
    file.read(reinterpret_cast<char*>(data.data()), expected_size * sizeof(float));
    return data;
}

// ─── Weight Transpose ─────────────────────────────────────────────────────────
// File layout:    [slice_Cout, K, K, Cin]   (contiguous in Cin)
// Compute layout: [K, K, Cin, slice_Cout]   (contiguous in Cout → auto-vectorize)

vector<float> transpose_weight(const vector<float>& w_src, int slice_c_out, int k, int c_in) {
    vector<float> w_dst(w_src.size());
    for (int oc = 0; oc < slice_c_out; ++oc) {
        for (int kh = 0; kh < k; ++kh) {
            for (int kw = 0; kw < k; ++kw) {
                for (int ic = 0; ic < c_in; ++ic) {
                    int src_idx = ((oc * k + kh) * k + kw) * c_in + ic;
                    int dst_idx = ((kh * k + kw) * c_in + ic) * slice_c_out + oc;
                    w_dst[dst_idx] = w_src[src_idx];
                }
            }
        }
    }
    return w_dst;
}

// ─── Conv2D Kernel ────────────────────────────────────────────────────────────
// Input layout:   [H, W, Cin]
// Weight layout:  [K, K, Cin, slice_Cout] (transposed)
// Output layout:  [out_H, out_W, slice_Cout]

void compute_conv2d_slice(const float* __restrict__ input,
                          const float* __restrict__ weight_t,
                          float* __restrict__ output,
                          int h, int w, int c_in,
                          int slice_c_out, int k, int pad, int stride,
                          int out_h, int out_w,
                          int row_start, int row_end) {

    for (int oh = row_start; oh < row_end; ++oh) {
        for (int ow = 0; ow < out_w; ++ow) {
            int out_base = (oh * out_w + ow) * slice_c_out;

            // Zero output for this spatial point
            for (int oc = 0; oc < slice_c_out; ++oc)
                output[out_base + oc] = 0.0f;

            for (int kh = 0; kh < k; ++kh) {
                int ih = oh * stride - pad + kh;
                if (ih < 0 || ih >= h) continue;

                for (int kw = 0; kw < k; ++kw) {
                    int iw = ow * stride - pad + kw;
                    if (iw < 0 || iw >= w) continue;

                    int in_idx = (ih * w + iw) * c_in;
                    for (int ic = 0; ic < c_in; ++ic) {
                        float in_val = input[in_idx + ic];
                        int w_base = ((kh * k + kw) * c_in + ic) * slice_c_out;

                        // Inner loop: oc contiguous in both output and weight_t
                        // → clang auto-vectorizes with NEON/AVX2
                        for (int oc = 0; oc < slice_c_out; ++oc) {
                            output[out_base + oc] += in_val * weight_t[w_base + oc];
                        }
                    }
                }
            }
        }
    }
}

void run_multithreaded(const float* input, const float* weight_t, float* output,
                       int h, int w, int c_in, int slice_c_out, int k, int pad, int stride,
                       int out_h, int out_w, int num_workers) {

    vector<thread> threads;
    int rows_per_thread = (out_h + num_workers - 1) / num_workers;

    for (int t = 0; t < num_workers; ++t) {
        int row_start = t * rows_per_thread;
        int row_end = min(row_start + rows_per_thread, out_h);
        if (row_start >= out_h) break;

        threads.emplace_back(compute_conv2d_slice,
            input, weight_t, output, h, w, c_in, slice_c_out, k, pad, stride,
            out_h, out_w, row_start, row_end);
    }
    for (auto& th : threads) th.join();
}

// ─── Timing Helper ────────────────────────────────────────────────────────────

double time_run(function<void()> fn, int repeats) {
    auto t1 = chrono::high_resolution_clock::now();
    for (int i = 0; i < repeats; ++i) fn();
    auto t2 = chrono::high_resolution_clock::now();
    return chrono::duration<double>(t2 - t1).count() / repeats;
}

// ─── Main ─────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    string input_path, weight_path;
    int h = 0, w = 0, c_in = 0, total_c_out = 0, k = 0, pad = 0, stride = 1;
    int start_oc = 0, end_oc = 0;
    vector<int> workers, tile_sizes;
    string output_path;
    int autotune_repeats = 1, measurement_repeats = 1;

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--input" && i + 1 < argc) input_path = argv[++i];
        else if (arg == "--weight" && i + 1 < argc) weight_path = argv[++i];
        else if (arg == "--h" && i + 1 < argc) h = stoi(argv[++i]);
        else if (arg == "--w" && i + 1 < argc) w = stoi(argv[++i]);
        else if (arg == "--cin" && i + 1 < argc) c_in = stoi(argv[++i]);
        else if (arg == "--cout" && i + 1 < argc) total_c_out = stoi(argv[++i]);
        else if (arg == "--k" && i + 1 < argc) k = stoi(argv[++i]);
        else if (arg == "--pad" && i + 1 < argc) pad = stoi(argv[++i]);
        else if (arg == "--stride" && i + 1 < argc) stride = stoi(argv[++i]);
        else if (arg == "--start-oc" && i + 1 < argc) start_oc = stoi(argv[++i]);
        else if (arg == "--end-oc" && i + 1 < argc) end_oc = stoi(argv[++i]);
        else if (arg == "--workers" && i + 1 < argc) workers = parse_list(argv[++i]);
        else if (arg == "--tile-sizes" && i + 1 < argc) tile_sizes = parse_list(argv[++i]);
        else if (arg == "--autotune-repeats" && i + 1 < argc) autotune_repeats = stoi(argv[++i]);
        else if (arg == "--measurement-repeats" && i + 1 < argc) measurement_repeats = stoi(argv[++i]);
        else if (arg == "--output" && i + 1 < argc) output_path = argv[++i];
    }

    if (end_oc == 0) end_oc = total_c_out;
    int slice_c_out = end_oc - start_oc;
    if (h <= 0 || w <= 0 || c_in <= 0 || total_c_out <= 0 || k <= 0 || stride <= 0 || slice_c_out <= 0) {
        cerr << "Error: invalid convolution dimensions" << endl;
        return 1;
    }

    int out_h = (h + 2 * pad - k) / stride + 1;
    int out_w = (w + 2 * pad - k) / stride + 1;
    if (out_h <= 0 || out_w <= 0) {
        cerr << "Error: invalid output shape after applying padding/kernel size/stride" << endl;
        return 1;
    }
    size_t input_size = (size_t)h * w * c_in;
    size_t weight_size = (size_t)k * k * c_in * slice_c_out;
    size_t output_size = (size_t)out_h * out_w * slice_c_out;

    vector<float> input_data = load_binary(input_path, input_size);
    vector<float> weight_raw = load_binary(weight_path, weight_size);
    vector<float> output_data(output_size, 0.0f);

    // Transpose weight: [slice_Cout, K, K, Cin] → [K, K, Cin, slice_Cout]
    vector<float> weight_t = transpose_weight(weight_raw, slice_c_out, k, c_in);

    double flops_per_run = 2.0 * out_h * out_w * slice_c_out * c_in * k * k;

    if (workers.empty()) {
        workers.push_back(max(1, (int)thread::hardware_concurrency()));
    }

    // ─── Autotune: sweep all worker counts, pick fastest ──────────────────
    int best_worker = workers[0];
    double best_autotune_time = 1e30;

    for (int w_count : workers) {
        double elapsed = time_run([&]() {
            run_multithreaded(input_data.data(), weight_t.data(), output_data.data(),
                              h, w, c_in, slice_c_out, k, pad, stride, out_h, out_w, w_count);
        }, autotune_repeats);

        if (elapsed < best_autotune_time) {
            best_autotune_time = elapsed;
            best_worker = w_count;
        }
    }

    // ─── Measurement: run with best config ────────────────────────────────
    double measure_time = time_run([&]() {
        run_multithreaded(input_data.data(), weight_t.data(), output_data.data(),
                          h, w, c_in, slice_c_out, k, pad, stride, out_h, out_w, best_worker);
    }, measurement_repeats);

    // ─── Write output file if requested ────────────────────────────────
    if (!output_path.empty()) {
        ofstream out(output_path, ios::binary);
        out.write(reinterpret_cast<const char*>(output_data.data()), output_data.size() * sizeof(float));
    }

    // ─── Checksum & Output ────────────────────────────────────────────────
    double sum_val = 0;
    for (size_t i = 0; i < output_data.size(); ++i) sum_val += abs(output_data[i]);
    string checksum = "chk_" + to_string((long long)sum_val);

    int best_tile = tile_sizes.empty() ? 1 : tile_sizes[0];

    cout << "{\n"
         << "  \"actual_workers\": " << best_worker << ",\n"
         << "  \"requested_workers\": " << workers[0] << ",\n"
         << "  \"tile_size\": " << best_tile << ",\n"
         << "  \"autotune_repeats\": " << autotune_repeats << ",\n"
         << "  \"measurement_repeats\": " << measurement_repeats << ",\n"
         << "  \"trials_run\": " << workers.size() << ",\n"
         << "  \"autotune_wall_clock_latency_seconds\": " << fixed << setprecision(9) << best_autotune_time << ",\n"
         << "  \"autotune_effective_gflops\": " << (flops_per_run / best_autotune_time / 1e9) << ",\n"
         << "  \"autotune_checksum\": \"" << checksum << "\",\n"
         << "  \"measurement_wall_clock_latency_seconds\": " << fixed << setprecision(9) << measure_time << ",\n"
         << "  \"measurement_effective_gflops\": " << (flops_per_run / measure_time / 1e9) << ",\n"
         << "  \"measurement_checksum\": \"" << checksum << "\"\n"
         << "}\n";

    return 0;
}
