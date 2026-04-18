#include <metal_stdlib>
using namespace metal;

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
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant ConvParams& params [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint ow = gid.x;
    uint oh = gid.y;
    uint oc = gid.z;

    if (ow >= params.out_w || oh >= params.out_h || oc >= params.c_out) return;

    float sum = 0.0f;
    for (uint ic = 0; ic < params.c_in; ++ic) {
        for (uint kh = 0; kh < params.k; ++kh) {
            for (uint kw = 0; kw < params.k; ++kw) {
                int ih = (int)oh * (int)params.stride - (int)params.pad + (int)kh;
                int iw = (int)ow * (int)params.stride - (int)params.pad + (int)kw;

                if (ih >= 0 && ih < (int)params.h && iw >= 0 && iw < (int)params.w) {
                    uint in_idx = (ih * params.w + iw) * params.c_in + ic;
                    uint w_idx = ((kh * params.k + kw) * params.c_in + ic) * params.c_out + oc;
                    sum += input[in_idx] * weight[w_idx];
                }
            }
        }
    }
    output[(oh * params.out_w + ow) * params.c_out + oc] = sum;
}
