"""Normalize expert inputs and score the 128 routes from the same BF16 values."""
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.kernels.common import scalar
from mlx_lm.models.inkling2 import layers


# Both projection paths finish the unbiased FP32 score and its rank key in the same lane.
_ROUTE_PAIR_HEADER = """
    template <typename B>
    inline void store_route_pair(device float* pairs, device const B* bias, uint e, float x) {
        float p=1.f/(1.f+precise::exp(abs(x)));
        float score=x<0 ? p : 1.f-p;
        pairs[2*e]=score;
        pairs[2*e+1]=score+float(bias[e]);
    }
"""

_kernel = mx.fast.metal_kernel(
    name="inkling_normalized_router", input_names=["x", "norm_w", "route_w", "eps", "squares", "bias"],
    header=_ROUTE_PAIR_HEADER,
    output_names=["normalized", "logits"],
    source="""
        uint tid=thread_position_in_threadgroup.x, lane=thread_index_in_simdgroup;
        uint sg=simdgroup_index_in_threadgroup, base=threadgroup_position_in_grid.x*ROWS;
        threadgroup float sums[18], partial[8][ROWS];
        float inv;
        if (PRECOMPUTED) {
            inv=precise::rsqrt(simd_sum(lane<18 ? squares[lane]:0.f)/2304.f+eps);
        } else {
            for (uint part=sg;part<18;part+=8) {
                float norm2=0;
                for (uint j=0;j<4;++j) {
                    float v=x[(part*32+lane)*4+j];
                    norm2+=v*v;
                }
                norm2=simd_sum(norm2);
                if (!lane) sums[part]=norm2;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            inv=precise::rsqrt(simd_sum(lane<18 ? sums[lane]:0.f)/2304.f+eps);
        }
        float dots[ROWS]={};
        for (uint d=tid*4;d<2304;d+=1024) {
            for (uint j=0;j<4;++j) {
                bfloat16_t value=bfloat16_t((float(x[d+j])*inv)*float(norm_w[d+j]));
                if (!base) normalized[d+j]=value;
                for (uint r=0;r<ROWS;++r)
                    dots[r]=fma(float(value),float(route_w[(base+r)*2304+d+j]),dots[r]);
            }
        }
        for (uint r=0;r<ROWS;++r) {
            float dot=dots[r];
            for (ushort i=16;i;i>>=1) dot+=simd_shuffle_down(dot,i);
            if (!lane) partial[sg][r]=dot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid<ROWS) {
            float total=partial[0][tid];
            for (uint i=1;i<8;++i) total+=partial[i][tid];
            if (PAIRED) store_route_pair(logits,bias,base+tid,total);
            else logits[base+tid]=total;
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


_rank_pairs = mx.fast.metal_kernel(
    name="inkling_rank_probability_pairs",input_names=["logits","eps","scale"],
    output_names=["inds","weights"],source="""
        uint tid=thread_position_in_threadgroup.x, lane=thread_index_in_simdgroup;
        uint warp=simdgroup_index_in_threadgroup;
        threadgroup float selected[8];
        float scores[4], other[4];
        bool special=false;
        for (uint j=0;j<4;++j) {
            float2 pair=((device const float2*)logits)[lane+32*j];
            scores[j]=pair.x; other[j]=pair.y;
            special |= !isfinite(pair.y);
        }
        if (simd_any(special)) {
            if (!warp) {
                float b[4];
                for (uint j=0;j<4;++j) b[j]=other[j];
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
                        keep=scores[j]; b[j]=-INFINITY;
                    }
                    float w=simd_sum(keep);
                    if (!lane) {inds[k]=winner; selected[k]=w;}
                }
            }
        } else {
            for (uint i=0;i<4;++i) {
                uint e=warp+32*i;
                float query=simd_shuffle(other[i],warp);
                uint ahead=0;
                for (uint j=0;j<4;++j)
                    ahead+=uint(other[j]>query || (other[j]==query && lane+32*j<e));
                uint rank=simd_sum(ahead);
                float score=simd_shuffle(scores[i],warp);
                if (!lane && rank<8) {inds[rank]=e; selected[rank]=score;}
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


def route(layer, x, dtype, normalized=False, logits=None, squares=None):
    norm, gate = layer.post_attention_layernorm, layer.mlp.gate
    if (type(norm) is not nn.RMSNorm or type(gate) is not layers.TopKRouter
            or x.dtype != (dtype if normalized else mx.float32) or dtype != mx.bfloat16 or gate.weight.shape != (128,2304)):
        x = x if normalized else norm(x).astype(dtype)
        indices, weights = gate(x)
        return x, indices, weights
    if normalized and logits is None:
        logits = _dot_only(inputs=[x,gate.weight],grid=(32*256,1,1),threadgroup=(256,1,1),
                           output_shapes=[(1,1,128)],output_dtypes=[mx.float32])[0]
    elif not normalized:
        paired = gate.top_k==8
        x, logits = _kernel(inputs=[x,norm.weight,gate.weight,scalar(norm.eps),x if squares is None else squares,gate.e_score_correction_bias],
                            template=[("PAIRED",paired),("PRECOMPUTED",squares is not None),("ROWS",2 if squares is not None else 4)],
                            grid=((64 if squares is not None else 32)*256,1,1),threadgroup=(256,1,1),
                            output_shapes=[x.shape,(1,1,128,2) if paired else (1,1,128)],output_dtypes=[dtype,mx.float32])
    if gate.top_k == 8:
        paired = logits.shape==(1,1,128,2)
        kernel = _rank_pairs if paired else _select
        args = [logits] if paired else [logits,gate.e_score_correction_bias]
        args.extend([scalar(layers._ROUTER_SCORE_EPS),scalar(gate.routed_scaling_factor)])
        indices, weights = kernel(inputs=args, grid=(1024,1,1),threadgroup=(1024,1,1),
            output_shapes=[(1,1,8)]*2, output_dtypes=[mx.uint32,mx.float32])
    else:
        indices, weights = layers.expert_routing(
            logits, gate.e_score_correction_bias, top_k=gate.top_k, multiplier=1.,
            score_eps=layers._ROUTER_SCORE_EPS,route_scale=gate.routed_scaling_factor)
    return x, indices, weights
