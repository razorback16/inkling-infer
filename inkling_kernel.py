"""Single-token weighted mixture of eight Inkling expert outputs."""
import mlx.core as mx
from mlx_lm.models.inkling2 import layers

_reference = layers.SparseMoeBlock.__call__
_mix = mx.fast.metal_kernel(
    name="inkling_mix8",
    input_names=["states", "weights"],
    output_names=["out"],
    source="""
        uint c = thread_position_in_grid.x;
        float value = 0;
        for (uint e = 0; e < 8; ++e)
            value += float(states[e * 2304 + c]) * float(weights[e]);
        out[c] = static_cast<T>(value);
    """,
)


def mix(states, weights, dtype):
    return _mix(inputs=[states, weights], template=[("T", dtype)],
                grid=(2304, 1, 1), threadgroup=(128, 1, 1),
                output_shapes=[(1, 1, 2304)], output_dtypes=[dtype])[0]


_down_mix = mx.fast.metal_kernel(
    name="inkling_nvfp4_down_mix",
    input_names=["hidden", "indices", "routing", "packed", "scales", "amax"],
    output_names=["out"],
    header="""
        constant float fp4_values[16] = {
            0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6
        };

        inline float decode_scale(uchar bits) {
            ushort magnitude = bits & 127;
            ushort raw = (magnitude << 7) | (((magnitude + 1) >> 7) << 14)
                         | ((bits & 128) << 8);
            return float(as_type<half>(raw)) * 256.f;
        }
    """,
    source="""
        // Each SIMD group projects one expert into the same four channels.
        uint slot = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint channel = 4 * threadgroup_position_in_grid.x;
        uint expert = indices[slot];
        uint first_row = expert * 2304 + channel;
        float accum[4] = {};
        for (uint word = lane; word < 144; word += 32) {
            float x[8];
            for (uint k = 0; k < 8; ++k)
                x[k] = float(hidden[slot * 1152 + word * 8 + k]);
            for (uint r = 0; r < 4; ++r) {
                uint row = first_row + r;
                uint q = packed[row * 144 + word];
                float chunk = 0;
                for (uint k = 0; k < 8; k += 4) {
                    chunk += x[k] * fp4_values[q & 15]
                           + x[k+1] * fp4_values[(q >> 4) & 15]
                           + x[k+2] * fp4_values[(q >> 8) & 15]
                           + x[k+3] * fp4_values[(q >> 12) & 15];
                    q >>= 16;
                }
                accum[r] += decode_scale(scales[row * 72 + word / 2]) * chunk;
            }
        }
        threadgroup float projected[8][4];
        for (uint r = 0; r < 4; ++r) {
            float dot = simd_sum(accum[r]);
            if (lane == 0)
                projected[slot][r] = float(T(dot * (amax[expert] / 2688.f)));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (slot == 0 && lane < 4) {
            float value = 0;
            for (uint e = 0; e < 8; ++e)
                value += projected[e][lane] * float(routing[e]);
            out[channel + lane] = T(value);
        }
    """,
)


def single_token_moe(self, x):
    if x.shape != (1, 1, 2304) or self.gate.top_k != 8 or self.sharding_group is not None:
        return _reference(self, x)
    experts, weights = self.gate(x)
    down = self.experts.down_proj
    if (getattr(down, "mode", None) == "nvfp4" and down.group_size == 16
            and down.bits == 4 and down.weight.shape[1:] == (2304, 144)
            and down.global_scale is not None
            and down.global_scale.shape == (down.weight.shape[0],) and "bias" not in down
            and x.dtype == mx.bfloat16):
        gate_up = self.experts.gate_up_proj(x[..., None, None, :], experts)
        hidden = self.experts.act_fn(gate_up[..., ::2], gate_up[..., 1::2])
        return _down_mix(
            inputs=[hidden, experts, weights, down.weight, down.scales, down.global_scale],
            template=[("T", x.dtype)], grid=(576 * 256, 1, 1),
            threadgroup=(256, 1, 1), output_shapes=[x.shape], output_dtypes=[x.dtype],
        )[0]
    return mix(self.experts(x, experts), weights, x.dtype)


def install():
    layers.SparseMoeBlock.__call__ = single_token_moe
