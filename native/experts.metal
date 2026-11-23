#include <metal_stdlib>
using namespace metal;

// Conversion arithmetic adapted from MLX v0.32.4, mlx/backend/metal/kernels/fp8.h.
// See MLX-LICENSE.txt in this directory.
inline float e4m3(uchar value) {
    ushort magnitude = value & 127;
    ushort bits = ((value & 128) << 8) | (magnitude << 7) | (((magnitude + 1) >> 7) << 14);
    return float(as_type<half>(bits)) * 256.0f;
}

inline float e8m0(uchar value) {
    uint bits = value == 0 ? 0x400000u : uint(value) << 23;
    return as_type<float>(bits);
}

constant float fp4[16] = {0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6};
constant bool nvfp4 [[function_constant(0)]];

// One SIMD group projects one row. Checkpoint rows are contiguous and packed
// least-significant nibble first. Scale groups never cross a row boundary.
template<uint width>
inline float project(device const bfloat *x, device const uchar *weights,
                     device const uchar *scales, uint row, uint lane) {
    #pragma clang fp contract(off)
    float dot = 0;
    if (nvfp4) {
        device const uint *words = (device const uint *)weights + row * (width / 8);
        for (uint word = lane; word < width / 8; word += 32) {
            uint codes = words[word];
            float sum = 0;
            for (uint k = 0; k < 8; k += 4) {
                device const bfloat *v = x + word * 8 + k;
                sum += float(v[0]) * fp4[codes & 15] + float(v[1]) * fp4[(codes >> 4) & 15]
                     + float(v[2]) * fp4[(codes >> 8) & 15] + float(v[3]) * fp4[(codes >> 12) & 15];
                codes >>= 16;
            }
            dot += sum * e4m3(scales[row * (width / 16) + word / 2]);
        }
    } else {
        // Keep the same reduction grouping as the Inkling decode kernels.
        constexpr uint chunk = width == 2304 ? 8 : 4;
        for (uint block = lane; block < width / chunk; block += 32) {
            uint offset = block * chunk;
            float sum = 0;
            for (uint k = 0; k < chunk; ++k)
                sum += float(x[offset + k]) * e4m3(weights[row * width + offset + k]);
            dot += sum * e8m0(scales[row * (width / 32) + offset / 32]);
        }
    }
    return simd_sum(dot);
}

kernel void expert_gate_up(device const bfloat *x [[buffer(0)]],
                           device const uint *indices [[buffer(1)]],
                           device const uchar *weights [[buffer(2)]],
                           device const uchar *scales [[buffer(3)]],
                           device const float *amax [[buffer(4)]],
                           device bfloat *active [[buffer(5)]],
                           constant uint &expertCount [[buffer(6)]],
                           uint3 group [[threadgroup_position_in_grid]],
                           uint simd [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
    uint slot = group.z * 8 + group.y;
    uint expert = indices[slot];
    x += group.z * 2304;
    uint channel = group.x * 4 + simd;
    if (expert >= expertCount) {
        if (lane == 0) active[slot * 1152 + channel] = 0;
        return;
    }
    uint row = expert * 2304 + 2 * channel;
    float scale = nvfp4 ? amax[expert] / 2688.f : 1.f;
    float gate = float(bfloat(project<2304>(x, weights, scales, row, lane) * scale));
    float up = float(bfloat(project<2304>(x, weights, scales, row + 1, lane) * scale));
    if (lane == 0) {
        float p = 1.f / (1.f + precise::exp(abs(gate)));
        float sigmoid = gate < 0 ? p : 1.f - p;
        active[slot * 1152 + channel] = bfloat((4.f * precise::tanh(gate / 4.f) * sigmoid)
                                                   * (25.f * precise::tanh(up / 25.f)));
    }
}

kernel void expert_down_mix(device const bfloat *active [[buffer(0)]],
                            device const uint *indices [[buffer(1)]],
                            device const uchar *weights [[buffer(2)]],
                            device const uchar *scales [[buffer(3)]],
                            device const float *amax [[buffer(4)]],
                            device bfloat *output [[buffer(5)]],
                            device const float *routing [[buffer(6)]],
                            constant uint &expertCount [[buffer(7)]],
                            uint2 group [[threadgroup_position_in_grid]],
                            uint slot [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
    #pragma clang fp contract(off)
    indices += group.y * 8;
    routing += group.y * 8;
    active += group.y * 8 * 1152;
    uint expert = indices[slot];
    threadgroup float projections[8];
    float value = 0;
    if (expert < expertCount) {
        float scale = nvfp4 ? amax[expert] / 2688.f : 1.f;
        value = float(bfloat(project<1152>(active + slot * 1152, weights, scales,
                                           expert * 2304 + group.x, lane) * scale));
    }
    if (lane == 0) projections[slot] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (slot == 0 && lane == 0) {
        float sum = 0;
        for (uint i = 0; i < 8; ++i) sum += projections[i] * routing[i];
        output[group.y * 2304 + group.x] = bfloat(sum);
    }
}
