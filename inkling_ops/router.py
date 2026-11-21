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


_dot_only = mx.fast.metal_kernel(
    name="inkling_prenormalized_router", input_names=["x","w"], output_names=["logits"],
    source="""
        uint tid=thread_position_in_threadgroup.x, lane=thread_index_in_simdgroup;
        uint sg=simdgroup_index_in_threadgroup, base=threadgroup_position_in_grid.x*4;
        threadgroup float partial[8][4];
        for (uint r=0;r<4;++r) {
            float dot=0;
            for (uint d=tid*4;d<2304;d+=1024)
                for (uint j=0;j<4;++j) dot=fma(float(x[d+j]),float(w[(base+r)*2304+d+j]),dot);
            for (ushort n=16;n;n>>=1) dot+=simd_shuffle_down(dot,n);
            if (!lane) partial[sg][r]=dot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid<4) {
            float val=partial[0][tid];
            for (uint i=1;i<8;++i) val+=partial[i][tid];
            logits[base+tid]=val;
        }
    """,
)


_select = mx.fast.metal_kernel(
    name="inkling_parallel_rank_routes",
    input_names=["logits","bias","eps","scale"], output_names=["inds","weights"],
    source="""
        uint tid=thread_position_in_threadgroup.x, lane=thread_index_in_simdgroup;
        uint warp=simdgroup_index_in_threadgroup;
        threadgroup float biased[128], scores[128], selected[8];
        threadgroup uint special[4];
        if (tid<128) {
            float x=logits[tid];
            float p=1.f/(1.f+precise::exp(abs(x)));
            float score=x<0 ? p : 1.f-p;
            scores[tid]=score;
            float b=score+float(bias[tid]);
            biased[tid]=b;
            bool nonfinite=simd_any(!isfinite(b));
            if (!lane) special[warp]=nonfinite;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (special[0] || special[1] || special[2] || special[3]) {
            if (!warp) {
                float b[4];
                for (uint j=0;j<4;++j) b[j]=biased[lane+32*j];
                for (uint k=0;k<8;++k) {
                    float best=-INFINITY;
                    uint which=128;
                    for (uint j=0;j<4;++j) {
                        uint e=lane+32*j;
                        if (b[j]>best || (b[j]==best && e<which)) {best=b[j];which=e;}
                    }
                    float peak=simd_max(best);
                    uint winner=simd_min(best==peak ? which : 128u);
                    float keep=0;
                    for (uint j=0;j<4;++j) if (lane+32*j==winner) {
                        keep=scores[winner]; b[j]=-INFINITY;
                    }
                    float w=simd_sum(keep);
                    if (!lane) {inds[k]=winner; selected[k]=w;}
                }
            }
        } else {
            float other[4];
            for (uint j=0;j<4;++j) other[j]=biased[lane+32*j];
            for (uint i=0;i<4;++i) {
                uint e=warp+32*i;
                float query=simd_shuffle(other[i],warp);
                uint ahead=0;
                for (uint j=0;j<4;++j)
                    ahead+=uint(other[j]>query || (other[j]==query && lane+32*j<e));
                uint rank=simd_sum(ahead);
                if (!lane && rank<8) {inds[rank]=e; selected[rank]=scores[e];}
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid<8) {
            float total=0;
            for (uint i=0;i<8;++i) total=selected[i]+total;
            weights[tid]=selected[tid]/(total+eps)*scale;
        }
    """,
)


def route(layer, x, dtype, normalized=False):
    norm, gate = layer.post_attention_layernorm, layer.mlp.gate
    if (type(norm) is not nn.RMSNorm or type(gate) is not layers.TopKRouter
            or x.dtype != (dtype if normalized else mx.float32) or dtype != mx.bfloat16 or gate.weight.shape != (128,2304)):
        x = x if normalized else norm(x).astype(dtype)
        indices, weights = gate(x)
        return x, indices, weights
    if normalized:
        logits = _dot_only(inputs=[x,native(gate)],grid=(32*256,1,1),threadgroup=(256,1,1),
                           output_shapes=[(1,1,128)],output_dtypes=[mx.float32])[0]
    else:
        x, logits = _kernel(inputs=[x,norm.weight,native(gate),scalar(norm.eps)],
                            grid=(32*576,1,1),threadgroup=(576,1,1),
                            output_shapes=[x.shape,(1,1,128)],output_dtypes=[dtype,mx.float32])
    if gate.top_k == 8:
        indices, weights = _select(
            inputs=[logits,gate.e_score_correction_bias,scalar(layers._ROUTER_SCORE_EPS),
                    scalar(gate.routed_scaling_factor)], grid=(1024,1,1),threadgroup=(1024,1,1),
            output_shapes=[(1,1,8)]*2, output_dtypes=[mx.uint32,mx.float32])
    else:
        indices, weights = layers.expert_routing(
            logits, gate.e_score_correction_bias, top_k=gate.top_k, multiplier=1.,
            score_eps=layers._ROUTER_SCORE_EPS,route_scale=gate.routed_scaling_factor)
    return x, indices, weights
