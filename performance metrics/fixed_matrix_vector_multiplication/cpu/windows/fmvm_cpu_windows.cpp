#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
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
    int workers = 1;
    int tile_size = 256;
    int repeats = 4;
};

// Parse the small command-line surface the Python benchmark passes in.
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
        } else if (key == "--workers") {
            options.workers = std::stoi(value);
        } else if (key == "--tile-size") {
            options.tile_size = std::stoi(value);
        } else if (key == "--repeats") {
            options.repeats = std::stoi(value);
        } else {
            throw std::runtime_error("unknown flag: " + key);
        }
    }

    if (options.matrix_path.empty() || options.vector_path.empty() || options.output_path.empty()) {
        throw std::runtime_error("matrix/vector/output paths are required");
    }
    if (options.rows <= 0 || options.cols <= 0) {
        throw std::runtime_error("rows and cols must be positive");
    }
    if (options.workers <= 0) {
        throw std::runtime_error("workers must be positive");
    }
    if (options.tile_size <= 0) {
        throw std::runtime_error("tile size must be positive");
    }
    if (options.repeats <= 0) {
        throw std::runtime_error("repeats must be positive");
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

void compute_row_range(
    const std::vector<float>& matrix_values,
    const std::vector<float>& vector_values,
    std::vector<float>& output_values,
    int rows,
    int cols,
    int row_start,
    int row_end,
    int tile_size
) {
    for (int row = row_start; row < row_end; ++row) {
        const size_t row_base = static_cast<size_t>(row) * static_cast<size_t>(cols);
        double accumulator = 0.0;

        // Tiling keeps the inner loop shape configurable so the benchmark can
        // search for a cache-friendlier chunk size on each machine.
        for (int tile_start = 0; tile_start < cols; tile_start += tile_size) {
            const int tile_end = std::min(cols, tile_start + tile_size);
            for (int col = tile_start; col < tile_end; ++col) {
                accumulator += static_cast<double>(matrix_values[row_base + static_cast<size_t>(col)])
                    * static_cast<double>(vector_values[static_cast<size_t>(col)]);
            }
        }

        output_values[static_cast<size_t>(row)] = static_cast<float>(accumulator);
    }
}

int run_once(
    const std::vector<float>& matrix_values,
    const std::vector<float>& vector_values,
    std::vector<float>& output_values,
    int rows,
    int cols,
    int requested_workers,
    int tile_size
) {
    const int actual_workers = std::max(1, std::min(requested_workers, rows));
    const int rows_per_worker = (rows + actual_workers - 1) / actual_workers;

    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(actual_workers));

    // Each worker owns a disjoint row range, so no locking is needed while
    // writing the output vector.
    for (int worker_index = 0; worker_index < actual_workers; ++worker_index) {
        const int row_start = worker_index * rows_per_worker;
        const int row_end = std::min(rows, row_start + rows_per_worker);
        if (row_start >= row_end) {
            break;
        }

        workers.emplace_back(
            compute_row_range,
            std::cref(matrix_values),
            std::cref(vector_values),
            std::ref(output_values),
            rows,
            cols,
            row_start,
            row_end,
            tile_size
        );
    }

    for (std::thread& worker : workers) {
        worker.join();
    }

    return static_cast<int>(workers.size());
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

        std::vector<float> output_values(static_cast<size_t>(options.rows), 0.0f);

        // Warm up once so the measured repeats focus more on the configured
        // worker/tile choice than on first-touch effects.
        const int actual_workers = run_once(
            matrix_values,
            vector_values,
            output_values,
            options.rows,
            options.cols,
            options.workers,
            options.tile_size
        );

        const auto started = std::chrono::steady_clock::now();
        for (int repeat = 0; repeat < options.repeats; ++repeat) {
            run_once(
                matrix_values,
                vector_values,
                output_values,
                options.rows,
                options.cols,
                options.workers,
                options.tile_size
            );
        }
        const auto finished = std::chrono::steady_clock::now();

        write_float32_file(options.output_path, output_values);

        const double total_seconds = std::chrono::duration<double>(finished - started).count();
        const double per_run_seconds = total_seconds / static_cast<double>(options.repeats);
        const unsigned int hardware_concurrency = std::thread::hardware_concurrency();

        std::cout << "{"
                  << "\"backend\":\"cpu\","
                  << "\"hardware_concurrency\":" << hardware_concurrency << ","
                  << "\"requested_workers\":" << options.workers << ","
                  << "\"actual_workers\":" << actual_workers << ","
                  << "\"tile_size\":" << options.tile_size << ","
                  << "\"repeats\":" << options.repeats << ","
                  << "\"total_seconds\":" << std::fixed << std::setprecision(9) << total_seconds << ","
                  << "\"per_run_seconds\":" << std::fixed << std::setprecision(9) << per_run_seconds
                  << "}" << std::endl;
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << escape_json(exc.what()) << std::endl;
        return 1;
    }
}
