#include <metal_stdlib>

using namespace metal;

struct FmvmParams {
    uint rows;
    uint cols;
    uint simdgroup_count;
    uint row_start;
};

template <uint TileSize>
inline void gemv_row_major_impl(
    device const float* matrix_values [[buffer(0)]],
    device const float* vector_values [[buffer(1)]],
    device float* output_values [[buffer(2)]],
    constant FmvmParams& params [[buffer(3)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threads_per_threadgroup [[threads_per_threadgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint simd_lane_index [[thread_index_in_simdgroup]],
    uint simdgroup_size [[threads_per_simdgroup]],
    threadgroup float* partial_sums [[threadgroup(0)]]
) {
    const uint output_row = threadgroup_position.x;
    if (output_row >= params.rows) {
        return;
    }
    const uint row = params.row_start + output_row;

    const uint worker_count = threads_per_threadgroup.x;
    const uint stride = worker_count * TileSize;
    const uint row_base = row * params.cols;

    float accumulator = 0.0f;
    for (uint column_base = thread_index; column_base < params.cols; column_base += stride) {
        #pragma unroll
        for (uint tile = 0; tile < TileSize; ++tile) {
            const uint column = column_base + tile * worker_count;
            if (column < params.cols) {
                accumulator += matrix_values[row_base + column] * vector_values[column];
            }
        }
    }

    const float simdgroup_total = simd_sum(accumulator);
    if (simd_lane_index == 0) {
        partial_sums[simdgroup_index] = simdgroup_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup_index == 0) {
        float final_sum = 0.0f;
        if (thread_index < params.simdgroup_count) {
            final_sum = partial_sums[thread_index];
        }
        final_sum = simd_sum(final_sum);
        if (simd_lane_index == 0) {
            output_values[output_row] = final_sum;
        }
    }
}

#define DEFINE_GEMV_ROW_MAJOR_KERNEL(TILE_SIZE) \
kernel void gemv_row_major_tile_##TILE_SIZE( \
    device const float* matrix_values [[buffer(0)]], \
    device const float* vector_values [[buffer(1)]], \
    device float* output_values [[buffer(2)]], \
    constant FmvmParams& params [[buffer(3)]], \
    uint3 threadgroup_position [[threadgroup_position_in_grid]], \
    uint thread_index [[thread_index_in_threadgroup]], \
    uint3 threads_per_threadgroup [[threads_per_threadgroup]], \
    uint simdgroup_index [[simdgroup_index_in_threadgroup]], \
    uint simd_lane_index [[thread_index_in_simdgroup]], \
    uint simdgroup_size [[threads_per_simdgroup]], \
    threadgroup float* partial_sums [[threadgroup(0)]] \
) { \
    gemv_row_major_impl<TILE_SIZE>( \
        matrix_values, \
        vector_values, \
        output_values, \
        params, \
        threadgroup_position, \
        thread_index, \
        threads_per_threadgroup, \
        simdgroup_index, \
        simd_lane_index, \
        simdgroup_size, \
        partial_sums); \
}

DEFINE_GEMV_ROW_MAJOR_KERNEL(1)
DEFINE_GEMV_ROW_MAJOR_KERNEL(2)
DEFINE_GEMV_ROW_MAJOR_KERNEL(4)
DEFINE_GEMV_ROW_MAJOR_KERNEL(8)
DEFINE_GEMV_ROW_MAJOR_KERNEL(16)
