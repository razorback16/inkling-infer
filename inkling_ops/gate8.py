"""Interleaved MXFP8 gate/up projection and SwiGLU for batch-one decode."""
import mlx.core as mx

_kernel = mx.fast.metal_kernel(
    name="inkling_mxfp8_gate", input_names=["x", "indices", "w", "scales"],
    output_names=["out"],
    source="""
        uint lane = thread_index_in_simdgroup;
        uint slot = threadgroup_position_in_grid.y;
        uint row = thread_position_in_grid.x / 32 * 2;
        uint expert = indices[slot];
        float gate = 0, up = 0;
        for (uint word = lane; word < 576; word += 32) {
            float dots[2];
            for (uint r = 0; r < 2; ++r) {
                uint n = expert * 2304 + row + r;
                uint packed = w[n * 576 + word];
                float sum = 0;
                for (uint i = 0; i < 4; ++i) {
                    ushort b = (packed >> (8 * i)) & 255;
                    ushort m = b & 127;
                    ushort bits = (m << 7) | (((m + 1) >> 7) << 14) | ((b & 128) << 8);
                    sum += float(x[word * 4 + i]) * (float(as_type<half>(bits)) * 256.f);
                }
                uint exponent = scales[n * 72 + word / 8];
                float scale = as_type<float>(exponent ? exponent << 23 : 0x400000u);
                dots[r] = sum * scale;
            }
            gate += dots[0];
            up += dots[1];
        }
        gate = float(bfloat16_t(simd_sum(gate)));
        up = float(bfloat16_t(simd_sum(up)));
        if (lane == 0) {
            float sigmoid = 1.f / (1.f + precise::exp(-gate));
            out[slot * 1152 + row / 2] = bfloat16_t(
                (precise::tanh(gate * .25f) * 4.f) * sigmoid * (precise::tanh(up / 25.f) * 25.f));
        }
    """,
)


def activate(x, ids, proj):
    return _kernel(inputs=[x, ids, proj.weight, proj.scales],
                   grid=(1152 * 32, 8, 1), threadgroup=(128, 1, 1),
                   output_shapes=[(1, 1, 8, 1, 1152)], output_dtypes=[x.dtype])[0]
