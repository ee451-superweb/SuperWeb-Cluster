/*
 * CPU Conv2D Benchmark Runner for macOS
 *
 * Key optimizations vs original:
 *   1. Weight index bug FIXED: file is [Cout,K,K,Cin], now transposed to [K,K,Cin,Cout]
 *   2. Loop order: kh -> kw -> ic -> oc (input reuse + auto-vectorizable inner loop)
 *   3. Real autotune: sweeps all worker counts, picks fastest
 *   4. Supports --start-oc / --end-oc for distributed slice computation
 *
 * Compile: clang++ -std=c++20 -O3 -ffast-math -pthread -o conv2d_cpu_macos conv2d_cpu_macos.cpp
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

// ─── Timing Helpers ───────────────────────────────────────────────────────────

struct TrialRecord {
    string phase;
    int candidate_index;
    int candidate_total;
    int trial_index_within_candidate;
    int repeats_for_candidate;
    int workers;
    int tile_size;
    double host_prep_seconds;
    double host_compute_seconds;
    double device_to_host_seconds;
    double host_postproc_seconds;
    double total_wall_seconds;
};

double time_run(function<void()> fn, int repeats) {
    auto t1 = chrono::high_resolution_clock::now();
    for (int i = 0; i < repeats; ++i) fn();
    auto t2 = chrono::high_resolution_clock::now();
    return chrono::duration<double>(t2 - t1).count() / repeats;
}

static inline double seconds_between(
    const chrono::high_resolution_clock::time_point& a,
    const chrono::high_resolution_clock::time_point& b) {
    return chrono::duration<double>(b - a).count();
}

// ─── Main ─────────────────────────────────────────────────────────────────────

enum class RunnerMode { Dispatch, Benchmark };

int main(int argc, char** argv) {
    string input_path, weight_path;
    int h = 0, w = 0, c_in = 0, total_c_out = 0, k = 0, pad = 0, stride = 1;
    int start_oc = 0, end_oc = 0;
    vector<int> workers, tile_sizes;
    string output_path;
    int autotune_repeats = 1, measurement_repeats = 1;
    RunnerMode mode = RunnerMode::Dispatch;
    bool verbose = false;

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
        else if (arg == "--verbose") verbose = true;
        else if (arg == "--mode" && i + 1 < argc) {
            string mode_value = argv[++i];
            if (mode_value == "dispatch") mode = RunnerMode::Dispatch;
            else if (mode_value == "benchmark") mode = RunnerMode::Benchmark;
            else { cerr << "Error: unknown --mode value: " << mode_value << endl; return 1; }
        }
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

    // Compulsory per-run DRAM traffic (see Windows runner for model notes).
    size_t bytes_input  = input_size  * sizeof(float);
    size_t bytes_weight = weight_size * sizeof(float);
    size_t bytes_output = output_size * sizeof(float);
    size_t bytes_kernel_compulsory = bytes_input + bytes_weight + bytes_output;

    if (workers.empty()) {
        workers.push_back(max(1, (int)thread::hardware_concurrency()));
    }

    int default_tile = tile_sizes.empty() ? 1 : tile_sizes[0];
    vector<TrialRecord> trials;
    trials.reserve(workers.size() * autotune_repeats + measurement_repeats);

    auto emit_verbose_trial = [&](const TrialRecord& tr, int global_index, int global_total) {
        if (!verbose) return;
        double compute_gflops = tr.host_compute_seconds > 0.0
            ? (flops_per_run / tr.host_compute_seconds / 1e9) : 0.0;
        double effective_gflops = tr.total_wall_seconds > 0.0
            ? (flops_per_run / tr.total_wall_seconds / 1e9) : 0.0;
        fprintf(stderr,
                "[conv2d cpu %s %d/%d] candidate=%d/%d (trial %d/%d) "
                "workers=%d tile_size=%d "
                "compute=%.6fs total=%.6fs "
                "compute_gflops=%.3f effective_gflops=%.3f\n",
                tr.phase.c_str(), global_index, global_total,
                tr.candidate_index + 1, tr.candidate_total,
                tr.trial_index_within_candidate + 1, tr.repeats_for_candidate,
                tr.workers, tr.tile_size,
                tr.host_compute_seconds, tr.total_wall_seconds,
                compute_gflops, effective_gflops);
        fflush(stderr);
    };

    int best_worker = workers[0];
    double best_autotune_mean_seconds = 0.0;
    double measure_sum_seconds = 0.0;
    double measure_time = 0.0;

    if (mode == RunnerMode::Benchmark) {
        int autotune_total = (int)workers.size() * autotune_repeats;
        int emitted_autotune = 0;
        if (verbose) {
            fprintf(stderr,
                    "[conv2d cpu plan] phase=autotune worker_candidates=%zu "
                    "autotune_repeats=%d total_trials=%d\n",
                    workers.size(), autotune_repeats, autotune_total);
            fflush(stderr);
        }

        best_autotune_mean_seconds = 1e30;
        for (size_t ci = 0; ci < workers.size(); ++ci) {
            int w_count = workers[ci];
            double candidate_sum_seconds = 0.0;
            for (int r = 0; r < autotune_repeats; ++r) {
                auto t_trial_start = chrono::high_resolution_clock::now();
                auto t_compute_start = chrono::high_resolution_clock::now();
                run_multithreaded(input_data.data(), weight_t.data(), output_data.data(),
                                  h, w, c_in, slice_c_out, k, pad, stride, out_h, out_w, w_count);
                auto t_compute_end = chrono::high_resolution_clock::now();
                auto t_trial_end = chrono::high_resolution_clock::now();

                TrialRecord tr;
                tr.phase = "autotune";
                tr.candidate_index = (int)ci;
                tr.candidate_total = (int)workers.size();
                tr.trial_index_within_candidate = r;
                tr.repeats_for_candidate = autotune_repeats;
                tr.workers = w_count;
                tr.tile_size = default_tile;
                tr.host_prep_seconds        = seconds_between(t_trial_start, t_compute_start);
                tr.host_compute_seconds     = seconds_between(t_compute_start, t_compute_end);
                tr.device_to_host_seconds   = 0.0;
                tr.host_postproc_seconds    = 0.0;
                tr.total_wall_seconds       = seconds_between(t_trial_start, t_trial_end);

                candidate_sum_seconds += tr.host_compute_seconds;
                ++emitted_autotune;
                emit_verbose_trial(tr, emitted_autotune, autotune_total);
                trials.push_back(std::move(tr));
            }
            double candidate_mean_seconds = candidate_sum_seconds / max(1, autotune_repeats);
            if (candidate_mean_seconds < best_autotune_mean_seconds) {
                best_autotune_mean_seconds = candidate_mean_seconds;
                best_worker = w_count;
            }
        }

        if (verbose) {
            fprintf(stderr,
                    "[conv2d cpu plan] phase=measurement selected_workers=%d "
                    "measurement_repeats=%d\n",
                    best_worker, measurement_repeats);
            fflush(stderr);
        }

        for (int r = 0; r < measurement_repeats; ++r) {
            auto t_trial_start = chrono::high_resolution_clock::now();
            auto t_compute_start = chrono::high_resolution_clock::now();
            run_multithreaded(input_data.data(), weight_t.data(), output_data.data(),
                              h, w, c_in, slice_c_out, k, pad, stride, out_h, out_w, best_worker);
            auto t_compute_end = chrono::high_resolution_clock::now();
            auto t_trial_end = chrono::high_resolution_clock::now();

            TrialRecord tr;
            tr.phase = "measurement";
            tr.candidate_index = 0;
            tr.candidate_total = 1;
            tr.trial_index_within_candidate = r;
            tr.repeats_for_candidate = measurement_repeats;
            tr.workers = best_worker;
            tr.tile_size = default_tile;
            tr.host_prep_seconds        = seconds_between(t_trial_start, t_compute_start);
            tr.host_compute_seconds     = seconds_between(t_compute_start, t_compute_end);
            tr.device_to_host_seconds   = 0.0;
            tr.host_postproc_seconds    = 0.0;
            tr.total_wall_seconds       = seconds_between(t_trial_start, t_trial_end);

            measure_sum_seconds += tr.host_compute_seconds;
            emit_verbose_trial(tr, r + 1, measurement_repeats);
            trials.push_back(std::move(tr));
        }
        measure_time = measure_sum_seconds / max(1, measurement_repeats);
    } else {
        // Dispatch: single compute pass with the worker count the caller
        // pinned (executor passes the benchmark-selected value), timed so
        // the JSON emits a compute_event_ms aligned with the benchmark's
        // measurement window.
        if (verbose) {
            fprintf(stderr,
                    "[conv2d cpu plan] phase=dispatch selected_workers=%d\n",
                    best_worker);
            fflush(stderr);
        }
        auto t_trial_start = chrono::high_resolution_clock::now();
        auto t_compute_start = chrono::high_resolution_clock::now();
        run_multithreaded(input_data.data(), weight_t.data(), output_data.data(),
                          h, w, c_in, slice_c_out, k, pad, stride, out_h, out_w, best_worker);
        auto t_compute_end = chrono::high_resolution_clock::now();
        auto t_trial_end = chrono::high_resolution_clock::now();

        TrialRecord tr;
        tr.phase = "dispatch";
        tr.candidate_index = 0;
        tr.candidate_total = 1;
        tr.trial_index_within_candidate = 0;
        tr.repeats_for_candidate = 1;
        tr.workers = best_worker;
        tr.tile_size = default_tile;
        tr.host_prep_seconds        = seconds_between(t_trial_start, t_compute_start);
        tr.host_compute_seconds     = seconds_between(t_compute_start, t_compute_end);
        tr.device_to_host_seconds   = 0.0;
        tr.host_postproc_seconds    = 0.0;
        tr.total_wall_seconds       = seconds_between(t_trial_start, t_trial_end);

        measure_sum_seconds = tr.host_compute_seconds;
        measure_time = tr.host_compute_seconds;
        trials.push_back(std::move(tr));
    }

    // ─── Write output file if requested ────────────────────────────────
    if (!output_path.empty()) {
        ofstream out(output_path, ios::binary);
        out.write(reinterpret_cast<const char*>(output_data.data()), output_data.size() * sizeof(float));
    }

    // ─── Checksum & Output ────────────────────────────────────────────────
    double sum_val = 0;
    for (size_t i = 0; i < output_data.size(); ++i) sum_val += std::abs(output_data[i]);
    string checksum = "chk_" + to_string((long long)sum_val);

    int best_tile = default_tile;

    double autotune_gflops = best_autotune_mean_seconds > 0.0
        ? (flops_per_run / best_autotune_mean_seconds / 1e9) : 0.0;
    double measurement_gflops = measure_time > 0.0
        ? (flops_per_run / measure_time / 1e9) : 0.0;
    double compute_event_ms = measure_time * 1000.0;
    const char* mode_str = (mode == RunnerMode::Benchmark) ? "benchmark" : "dispatch";
    size_t trials_run = (mode == RunnerMode::Benchmark) ? workers.size() : 1;

    cout << "{\n"
         << "  \"mode\": \"" << mode_str << "\",\n"
         << "  \"actual_workers\": " << best_worker << ",\n"
         << "  \"requested_workers\": " << workers[0] << ",\n"
         << "  \"tile_size\": " << best_tile << ",\n"
         << "  \"autotune_repeats\": " << autotune_repeats << ",\n"
         << "  \"measurement_repeats\": " << measurement_repeats << ",\n"
         << "  \"trials_run\": " << trials_run << ",\n"
         << "  \"compute_event_ms\": " << fixed << setprecision(6) << compute_event_ms << ",\n"
         << "  \"autotune_wall_clock_latency_seconds\": " << fixed << setprecision(9) << best_autotune_mean_seconds << ",\n"
         << "  \"autotune_effective_gflops\": " << autotune_gflops << ",\n"
         << "  \"autotune_checksum\": \"" << checksum << "\",\n"
         << "  \"measurement_wall_clock_latency_seconds\": " << fixed << setprecision(9) << measure_time << ",\n"
         << "  \"measurement_effective_gflops\": " << measurement_gflops << ",\n"
         << "  \"measurement_checksum\": \"" << checksum << "\",\n"
         << "  \"flops_per_run\": " << fixed << setprecision(1) << flops_per_run << ",\n"
         << "  \"bytes_input\": " << bytes_input << ",\n"
         << "  \"bytes_weight\": " << bytes_weight << ",\n"
         << "  \"bytes_output\": " << bytes_output << ",\n"
         << "  \"bytes_kernel_compulsory_memory_traffic\": " << bytes_kernel_compulsory << ",\n"
         << "  \"notes_schema\": \"CPU backend: host_prep/device_to_host/host_postproc are zero by definition; compute includes thread spawn+join overhead; memory_bandwidth model assumes perfect DRAM reuse (real traffic >= compulsory).\",\n"
         << "  \"trials\": [\n";
    for (size_t i = 0; i < trials.size(); ++i) {
        const auto& tr = trials[i];
        double compute_gflops = tr.host_compute_seconds > 0.0
            ? (flops_per_run / tr.host_compute_seconds / 1e9) : 0.0;
        double effective_gflops = tr.total_wall_seconds > 0.0
            ? (flops_per_run / tr.total_wall_seconds / 1e9) : 0.0;
        double kernel_bandwidth_gibps = tr.host_compute_seconds > 0.0
            ? ((double)bytes_kernel_compulsory / tr.host_compute_seconds / (1024.0 * 1024.0 * 1024.0)) : 0.0;
        cout << "    {"
             << "\"phase\": \"" << tr.phase << "\", "
             << "\"candidate_index\": " << tr.candidate_index << ", "
             << "\"candidate_total\": " << tr.candidate_total << ", "
             << "\"trial_index_within_candidate\": " << tr.trial_index_within_candidate << ", "
             << "\"repeats_for_candidate\": " << tr.repeats_for_candidate << ", "
             << "\"workers\": " << tr.workers << ", "
             << "\"tile_size\": " << tr.tile_size << ", "
             << fixed << setprecision(9)
             << "\"host_prep_seconds\": " << tr.host_prep_seconds << ", "
             << "\"host_compute_seconds\": " << tr.host_compute_seconds << ", "
             << "\"device_to_host_seconds\": " << tr.device_to_host_seconds << ", "
             << "\"host_postproc_seconds\": " << tr.host_postproc_seconds << ", "
             << "\"total_wall_seconds\": " << tr.total_wall_seconds << ", "
             << setprecision(6)
             << "\"compute_gflops\": " << compute_gflops << ", "
             << "\"effective_gflops\": " << effective_gflops << ", "
             << "\"pcie_h2d_bandwidth_gibps\": 0.0, "
             << "\"pcie_d2h_bandwidth_gibps\": 0.0, "
             << "\"kernel_memory_bandwidth_gibps_compulsory_lower_bound_model\": " << kernel_bandwidth_gibps
             << "}";
        if (i + 1 < trials.size()) cout << ",";
        cout << "\n";
    }
    cout << "  ]\n"
         << "}\n";

    return 0;
}
