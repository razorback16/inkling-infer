"""Project and normalize a full head in a single workgroup on shared-input sliding visits."""
import mlx.core as mx
from mlx_lm.kernels.common import scalar
from mlx_lm.models.inkling2 import layers

import mlx.nn as nn


_epilogue = """
        uint bank=threadgroup_position_in_grid.x;
        bool query=bank<24, relative=bank>=24 && bank<27;
        bool key=bank>=51 && bank<59;
        uint col=sg*4+lane;
        if (bank>=27 && bank<51) {
            if (lane<4) gate[(bank-27)*128+col]=bfloat16_t(value);
            return;
        }
        uint head=query ? bank : relative ? bank-24 : key ? bank-51 : bank-59;
        uint d=head*128+col;
        if (lane<4) {
            if (relative) value*=float(rs[d]);
            else if (query) value*=float(qs[d]);
            else {
                auto hist=key ? kh : vh;
                auto weight=key ? kw : vw;
                auto window=key ? k_window : v_window;
                float total=0;
                for (uint i=0;i<3;++i) {
                    float h=hist[i*1024+d];
                    window[i*1024+d]=h;
                    total=fma(float(W(h)),float(weight[d*4+i]),total);
                }
                window[3*1024+d]=value;
                total=fma(float(W(value)),float(weight[d*4+3]),total);
                value=float(bfloat16_t((float(W(total))+value)*0.4472135954999579f));
            }
            if (relative) values[col]=value;
        }
        if (relative) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane<4) {
                float sum=0;
                for (uint j=0;j<16;++j) {
                    float x=values[(col/16)*16+j];
                    sum+=x*x;
                }
                rel[d]=bfloat16_t(value*min(1.f,rt/(precise::sqrt(sum)+1e-6f)));
            }
            return;
        }
        float squares=0;
        for (ushort j=0;j<4;++j) {
            float x=simd_shuffle(value,j);
            squares+=x*x;
        }
        if (!lane) values[sg]=squares;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float sum=simd_sum(values[lane]);
        float scale=(query || key) ? min(1.f,qt/(precise::sqrt(sum)+1e-6f)) : precise::rsqrt(sum/128.f+eps);
        if (lane<4) {
            bfloat16_t normal=bfloat16_t(value*scale);
            if (query) { q[d]=normal; scaled[d]=attention_scale*float(normal); }
            else pair[(key ? 0 : 1024)+d]=normal;
        }
"""
def _make_kernel(distinct):
    inputs=["x","query_w","rel_w","gate_w","key_w","value_w","qs","rs","qt","rt","eps","attention_scale","kh","vh","kw","vw"]
    prefix="#pragma clang fp contract(off)\nthreadgroup float values[128]; float value=0; uint sg=simdgroup_index_in_threadgroup;\n"
    if distinct:
        inputs+=["kx","vx"]
    projection="""
        uint region=threadgroup_position_in_grid.x,lane=thread_index_in_simdgroup;
        auto input=INPUT;
        auto weights=region<24 ? query_w : region<27 ? rel_w : region<51 ? gate_w : region<59 ? key_w : value_w;
        uint first_region=region<24 ? 0 : region<27 ? 24 : region<51 ? 27 : region<59 ? 51 : 59;
        uint first=(region-first_region)*128+sg*4;
        float dots[4]={};
        for (uint col=lane*4;col<2304;col+=128) {
            float4 v=float4(*(device const bfloat4*)(input+col));
            for (uint r=0;r<4;++r) {
                float4 w=float4(*(device const bfloat4*)(weights+(first+r)*2304+col));
                for (uint j=0;j<4;++j) dots[r]=fma(v[j],w[j],dots[r]);
            }
        }
        for (uint r=0;r<4;++r) {
            float dot=dots[r];
            for (ushort j=16;j;j>>=1) dot+=simd_shuffle_down(dot,j);
            float all=simd_broadcast_first(dot);
            if (lane==r) value=float(bfloat16_t(all));
        }
    """.replace("INPUT","region<51 ? x : region<59 ? kx : vx" if distinct else "x")
    return mx.fast.metal_kernel(
        name="inkling_project_recurrent_heads" if distinct else "inkling_project_complete_heads",input_names=inputs,
        output_names=["q","scaled","rel","gate","pair","k_window","v_window"],source=prefix+projection+_epilogue)


_kernel=_make_kernel(False)
_distinct=_make_kernel(True)


def compute(layer,x,conv_states,conv_mask,kv,extra=()):
    modules=(layer.q_proj,layer.r_proj,layer.gate_proj,layer.k_proj,layer.v_proj)
    if (not getattr(kv,"grouped",False) or layer.sliding_window!=1024 or conv_mask is not None
            or x.dtype!=mx.bfloat16 or any(type(p) is not nn.Linear or "bias" in p
                   or p.weight.shape!=(rows,2304) or p.weight.dtype!=x.dtype
                   for p,rows in zip(modules,(3072,384,3072,1024,1024)))
            or any(v.shape!=x.shape or v.dtype!=x.dtype for v in extra)
            or (layer.head_dim,layer.num_heads,layer.num_key_value_heads)!=(128,24,8)
            or type(layer.rel_logits_proj) is not layers.RelativeLogits or conv_states is None
            or any(type(c) is not layers.ShortConvolution or c.kernel_size!=4 for c in (layer.k_sconv,layer.v_sconv))
            or any(h is None or h.shape!=(1,3,1024) for h in conv_states[:2])):
        return None
    kw,vw=layer.k_sconv.conv1d.weight,layer.v_sconv.conv1d.weight
    if kw.dtype!=vw.dtype or kw.shape!=vw.shape:
        return None
    kernel=_distinct if extra else _kernel
    q,scaled,rel,gate,pair,kh,vh=kernel(
        inputs=[x,*(p.weight for p in modules),layer.q_l2_clip.scale,layer.rel_l2_clip.scale,scalar(layer.q_l2_clip.threshold),
                scalar(layer.rel_l2_clip.threshold),scalar(layer.eps),scalar(layer.scaling),*conv_states[:2],kw,vw,*extra],
        template=[("W",kw.dtype)],grid=(8576*8,1,1),threadgroup=(1024,1,1),
        output_shapes=[(1,24,1,128),(1,24,1,128),(1,1,24,16),(1,1,3072),(1,16,1,128),(1,4,1024),(1,4,1024)],
        output_dtypes=[mx.bfloat16,mx.float32,mx.bfloat16,mx.bfloat16,mx.bfloat16,mx.float32,mx.float32])
    conv_states[:2]=kh,vh
    kv.defer_update=True
    k,v=kv.update_packed(pair)
    return q,k,v,rel,gate,scaled
