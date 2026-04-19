#include <metal_stdlib>
using namespace metal;

constant uint kOutputChannelsPerThread = 4;

struct ConvParams {
    uint h;
    uint w;
    uint c_in;
    uint c_out;
    uint k;
    uint pad;
    uint stride;
    uint out_h;
    uint out_w;
};

kernel void conv2d_kernel(
    device const float* input [[buffer(0)]],
    device const float4* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant ConvParams& params [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint ow = gid.x;
    uint oh = gid.y;
    uint oc_group = gid.z;
    uint oc_base = oc_group * kOutputChannelsPerThread;

    if (ow >= params.out_w || oh >= params.out_h || oc_base >= params.c_out) return;

    const uint oc_groups = (params.c_out + kOutputChannelsPerThread - 1) / kOutputChannelsPerThread;
    float4 sum = float4(0.0f);
    for (uint ic = 0; ic < params.c_in; ++ic) {
        for (uint kh = 0; kh < params.k; ++kh) {
            for (uint kw = 0; kw < params.k; ++kw) {
                int ih = (int)oh * (int)params.stride - (int)params.pad + (int)kh;
                int iw = (int)ow * (int)params.stride - (int)params.pad + (int)kw;

                if (ih >= 0 && ih < (int)params.h && iw >= 0 && iw < (int)params.w) {
                    uint in_idx = (ih * params.w + iw) * params.c_in + ic;
                    // Host code repacks weights as [k, k, c_in, ceil(c_out / 4)] with
                    // each entry materialized as a float4 over neighboring output channels.
                    uint w_idx = (((kh * params.k + kw) * params.c_in + ic) * oc_groups) + oc_group;
                    sum += input[in_idx] * weight[w_idx];
                }
            }
        }
    }

    const uint out_base = (oh * params.out_w + ow) * params.c_out + oc_base;
    const uint valid_outputs = min(kOutputChannelsPerThread, params.c_out - oc_base);
    if (valid_outputs >= 1) output[out_base + 0] = sum.x;
    if (valid_outputs >= 2) output[out_base + 1] = sum.y;
    if (valid_outputs >= 3) output[out_base + 2] = sum.z;
    if (valid_outputs >= 4) output[out_base + 3] = sum.w;
}
