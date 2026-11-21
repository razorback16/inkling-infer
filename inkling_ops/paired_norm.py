"""Normalize the two inputs of each cross-pass projection into one BF16 buffer."""
import mlx.core as mx

_kernel = mx.fast.metal_kernel(
    name="inkling_paired_rms", input_names=["state", "message"], output_names=["out"],
    source="""
        uint g = threadgroup_position_in_grid.x, bank = PREPARED ? g : g/2;
        uint tid = thread_position_in_threadgroup.x, lane = thread_index_in_simdgroup;
        uint simd = simdgroup_index_in_threadgroup;
        float sum = 0;
        for (uint d = tid * 4; d < (tid + 1) * 4; ++d) {
            float v = (!PREPARED && (g & 1)) ? float(message[bank*2304+d]) : float(state[(SHARED ? 0 : bank)*2304+d]);
            sum += v * v;
        }
        threadgroup float sums[18];
        sum = simd_sum(sum);
        if (lane == 0) sums[simd] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sum = simd_sum(lane < 18 ? sums[lane] : 0.f);
        float scale = precise::rsqrt(sum / 2304.f + 1e-6f);
        for (uint d = tid * 4; d < (tid + 1) * 4; ++d) {
            float v = (!PREPARED && (g & 1)) ? float(message[bank*2304+d]) : float(state[(SHARED ? 0 : bank)*2304+d]);
            if (PREPARED) {
                out[bank*4608+d] = T(v*scale);
                out[bank*4608+2304+d] = T(message[bank*2304+d]);
            } else {
                out[g*2304+d] = T(v*scale);
            }
        }
    """,
)


def join(x, message, dtype, prepared=False):
    n = message.size // 2304
    return _kernel(inputs=[x, message], template=[("T",dtype),("PREPARED",prepared),("SHARED",x.size != message.size)],
                   grid=(n * (1 if prepared else 2) * 576, 1, 1), threadgroup=(576, 1, 1),
                   output_shapes=[message.shape[:-1]+(4608,)], output_dtypes=[dtype])[0]
