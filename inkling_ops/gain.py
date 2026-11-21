"""Apply the two rank-64 message projections and mix their source states."""
import mlx.core as mx

_kernel = mx.fast.metal_kernel(
    name="inkling_gain_forget", input_names=["x", "packed", "state", "msg"], output_names=["out"],
    source="""
        uint lane = thread_index_in_simdgroup;
        uint row = thread_position_in_grid.x / 4;
        uint col = (lane % 4) * 4;
        uint bank = thread_position_in_grid.y;
        uint base = (bank * 4608 + row) * 64;
        float f = 0, g = 0;
        for (uint tile = 0; tile < 64; tile += 16) {
            for (uint i = 0; i < 4; ++i) {
                uint k = tile + col + i;
                float v = float(x[bank * 64 + k]);
                f = fma(v, float(packed[base + k]), f);
                g = fma(v, float(packed[base + 2304 * 64 + k]), g);
            }
        }
        f += simd_shuffle_down(f, 2);
        g += simd_shuffle_down(g, 2);
        f += simd_shuffle_down(f, 1);
        g += simd_shuffle_down(g, 1);
        if ((lane % 4) == 0) {
            uint c = bank * 2304 + row;
            volatile float retained = (1.f - precise::tanh(f)) * float(state[c]);
            volatile float added = precise::tanh(g) * float(msg[c]);
            out[c] = retained + added;
        }
    """,
)


def apply(x, packed, state, messages):
    banks = state.size // 2304
    return _kernel(inputs=[x, packed, state, messages],
                   grid=(2304 * 4, banks, 1), threadgroup=(128, 1, 1),
                   output_shapes=[state.shape], output_dtypes=[mx.float32])[0]
