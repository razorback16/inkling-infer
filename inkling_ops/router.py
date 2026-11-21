"""Normalize expert inputs and score the 128 routes from the same BF16 values."""
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.kernels.common import scalar
from mlx_lm.models.inkling2 import layers

from .exact_sources import native

_kernel = mx.fast.metal_kernel(
    name="inkling_normalized_router", input_names=["x", "norm_w", "route_w", "eps"],
    output_names=["normalized", "logits"],
    source="""
        uint tid = thread_position_in_threadgroup.x, lane = thread_index_in_simdgroup;
        uint sg = simdgroup_index_in_threadgroup, base = threadgroup_position_in_grid.x * 4;
        threadgroup float sums[18], partial[8][4];
        threadgroup bfloat16_t vec[2304];
        float acc = 0;
        for (uint j = 0; j < 4; ++j) {
            float v = x[tid*4+j];
            acc += v*v;
        }
        acc = simd_sum(acc);
        if (lane == 0) sums[sg] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float inv = precise::rsqrt(simd_sum(lane < 18 ? sums[lane] : 0.f) / 2304.f + eps);
        for (uint j = 0; j < 4; ++j) {
            uint d = tid*4+j;
            bfloat16_t v = bfloat16_t((x[d]*inv)*float(norm_w[d]));
            vec[d] = v;
            if (!base) normalized[d] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 256) {
            for (uint r = 0; r < 4; ++r) {
                float dot = 0;
                for (uint start = tid*4; start < 2304; start += 1024) {
                    for (uint j = 0; j < 4; ++j)
                        dot = fma(float(vec[start+j]),float(route_w[(base+r)*2304+start+j]),dot);
                }
                for (ushort s = 16; s; s >>= 1) dot += simd_shuffle_down(dot,s);
                if (!lane) partial[sg][r] = dot;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 4) {
            float total = partial[0][tid];
            for (uint i = 1; i < 8; ++i) total += partial[i][tid];
            logits[base+tid] = total;
        }
    """,
)


def route(layer, x, dtype):
    norm, gate = layer.post_attention_layernorm, layer.mlp.gate
    if (type(norm) is not nn.RMSNorm or type(gate) is not layers.TopKRouter
            or x.dtype != mx.float32 or dtype != mx.bfloat16 or gate.weight.shape != (128,2304)):
        x = norm(x).astype(dtype)
        indices, weights = gate(x)
        return x, indices, weights
    x, logits = _kernel(inputs=[x,norm.weight,native(gate),scalar(norm.eps)],
                        grid=(32*576,1,1),threadgroup=(576,1,1),
                        output_shapes=[x.shape,(1,1,128)],output_dtypes=[dtype,mx.float32])
    indices, weights = layers.expert_routing(
        logits, gate.e_score_correction_bias, top_k=gate.top_k, multiplier=1.,
        score_eps=layers._ROUTER_SCORE_EPS,route_scale=gate.routed_scaling_factor)
    return x, indices, weights
