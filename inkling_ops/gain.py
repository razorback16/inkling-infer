"""Apply the two rank-64 message projections and mix their source states."""
import mlx.core as mx

_source = """
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
        ushort leader=ushort(lane&~3u);
        float gain_dot=simd_shuffle(g,leader);
        float gate=precise::tanh((lane&1) ? gain_dot : f);
        {
        #pragma clang fp contract(off)
        uint c = bank * 2304 + row;
        float term;
        if (lane & 1) term = gate * float(msg[c]);
        else term = (1.f - gate) * float(state[SHARED ? row : c]);
        float added = simd_shuffle(term,ushort(leader+1));
        if ((lane % 4) == 0) {
            float retained=term;
            out[c] = retained + added;
        }
        }
    """


def _make_kernel(with_sums):
    source = _source
    outputs = ["out"]
    if with_sums:
        outputs.append("squares")
        source = "threadgroup float values[128];\n" + source.replace(
            "out[c] = retained + added;", "float value = retained + added; out[c] = value; values[row%128] = value;")
        source += """
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (!simdgroup_index_in_threadgroup) {
                float sum=0;
                for (uint j=0;j<4;++j) {
                    float v=values[lane*4+j]; sum+=v*v;
                }
                sum=simd_sum(sum);
                if (!lane) squares[bank*18+threadgroup_position_in_grid.x]=sum;
            }
        """
    return mx.fast.metal_kernel(name="inkling_gain_router_norm" if with_sums else "inkling_gain_forget",
                                input_names=["x","packed","state","msg"],output_names=outputs,source=source)


_kernel = _make_kernel(False)
_with_sums = _make_kernel(True)


def apply(x, packed, state, messages, with_sums=False):
    banks = messages.size // 2304
    shapes = [messages.shape]
    if with_sums:
        shapes.append((banks,18))
    kernel = _with_sums if with_sums else _kernel
    result = kernel(inputs=[x,packed,state,messages],template=[("SHARED",state.size != messages.size)],
                    grid=(2304*4,banks,1),threadgroup=(512 if with_sums else 128,1,1),
                    output_shapes=shapes,output_dtypes=[mx.float32]*len(shapes))
    return tuple(result) if with_sums else result[0]
