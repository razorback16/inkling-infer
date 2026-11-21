"""Single-token weighted mixture of eight Inkling expert outputs."""
import os

import mlx.core as mx
from mlx_lm.models.inkling2 import layers

_reference = layers.SparseMoeBlock.__call__
_mix = mx.fast.metal_kernel(
    name="inkling_mix8",
    input_names=["states", "weights"],
    output_names=["out"],
    source="""
        #pragma clang fp contract(off)
        uint c = thread_position_in_grid.x;
        float value = 0;
        for (uint e = 0; e < 8; ++e) {
            float term = float(states[e * 2304 + c]) * float(weights[e]);
            value = value + term;
        }
        out[c] = static_cast<T>(value);
    """,
)


def mix(states, weights, dtype):
    return _mix(inputs=[states, weights], template=[("T", dtype)],
                grid=(2304, 1, 1), threadgroup=(128, 1, 1),
                output_shapes=[(1, 1, 2304)], output_dtypes=[dtype])[0]


def _make_down(finish):
    inputs = ["hidden", "indices", "routing", "packed", "scales", "amax"]
    outputs = ["out"]
    store = "out[channel + lane] = T(value);"
    if finish:
        inputs += ["history", "conv_w", "residual"]
        outputs += ["next_history"]
        store = """
            uint c = channel + lane;
            float current = float(T(value));
            float total = 0;
            for (uint j = 0; j < 3; ++j) {
                float old = history[j*2304+c];
                total = fma(float(W(old)), float(conv_w[c*4+j]), total);
                if (j > 0) next_history[(j-1)*2304+c] = old;
            }
            total = fma(float(W(current)), float(conv_w[c*4+3]), total);
            next_history[2*2304+c] = current;
            float filtered = float(T((float(W(total)) + current) * 0.4472135954999579f));
            out[c] = T(residual[c] + filtered);
        """
    return mx.fast.metal_kernel(
        name="inkling_finish_nvfp4_layer" if finish else "inkling_nvfp4_down_mix",
        input_names=inputs, output_names=outputs,
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
        #pragma clang fp contract(off)
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
                float term = decode_scale(scales[row * 72 + word / 2]) * chunk;
                accum[r] += term;
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
            for (uint e = 0; e < 8; ++e) {
                float term = projected[e][lane] * float(routing[e]);
                value = value + term;
            }
            STORE
        }
    """.replace("STORE", store),
    )


_down_mix = _make_down(False)
_finish_mix = _make_down(True)


def can_fuse_down(down, x):
    return (getattr(down, "mode", None) == "nvfp4" and down.group_size == 16
            and down.bits == 4 and down.weight.shape[1:] == (2304, 144)
            and down.global_scale is not None
            and down.global_scale.shape == (down.weight.shape[0],) and "bias" not in down
            and x.dtype == mx.bfloat16)


def single_token_moe(self, x):
    if x.shape != (1, 1, 2304) or self.gate.top_k != 8 or self.sharding_group is not None:
        return _reference(self, x)
    experts, weights = self.gate(x)
    down = self.experts.down_proj
    if can_fuse_down(down, x):
        from inkling_ops.expert_gate import gated_activation
        hidden = gated_activation(self.experts, x, experts)
        return _down_mix(
            inputs=[hidden, experts, weights, down.weight, down.scales, down.global_scale],
            template=[("T", x.dtype)], grid=(576 * 256, 1, 1),
            threadgroup=(256, 1, 1), output_shapes=[x.shape], output_dtypes=[x.dtype],
        )[0]
    return mix(self.experts(x, experts), weights, x.dtype)


def install():
    if mx.device_info().get("device_name") == "Apple M3 Ultra":
        os.environ.setdefault("MLX_METAL_FAST_SYNCH", "1")
    from inkling_ops import packed_vocab
    packed_vocab.install()
    from inkling_ops import attention_mask, compiled_ring, dynamic_conv, message_gate, projections, short_conv, triplet
    from inkling_ops import expert_gate
    expert_gate.install()
    from inkling_ops import loop
    loop.install()
    from inkling_ops import input_heads
    input_heads.install()
    from inkling_ops import exact_sources
    exact_sources.install()
    short_conv.install()
    message_gate.install()
    attention_mask.install()
    projections.install()
    triplet.install()
    compiled_ring.install()
    dynamic_conv.install()
    layers.SparseMoeBlock.__call__ = single_token_moe
