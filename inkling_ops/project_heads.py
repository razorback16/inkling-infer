"""Project and normalize a full head in a single workgroup on shared-input sliding visits."""
import mlx.core as mx
from mlx_lm.kernels.common import scalar
from mlx_lm.models.inkling2 import layers

from .packed_rows import _dot_source
from .grouped_attn import gate_probabilities


_epilogue = """
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint segment=threadgroup_position_in_grid.x;
        if (segment>=24 && segment<27) {
            if (sg>=8 || lane>=16) return;
            uint head=(segment-24)*8+sg, base=head*16;
            float sum=0;
            for (uint j=0;j<16;++j) {
                float x=float(raw[sg*16+j])*float(rs[base+j]);
                sum+=x*x;
            }
            float a=min(1.f,rt/(precise::sqrt(sum)+1e-6f));
            rel[base+lane]=bfloat16_t((float(raw[sg*16+lane])*float(rs[base+lane]))*a);
            return;
        }
        if (sg) return;
        bool query=segment<24, key=segment>=51 && segment<59;
        uint head=query ? segment : key ? segment-51 : segment-59;
        float4 state;
        float sum=0;
        for (uint j=0;j<4;++j) {
            uint col=lane*4+j,d=head*128+col;
            float value=float(raw[col]);
            if (query) value*=float(qs[d]);
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
            state[j]=value;
            sum+=value*value;
        }
        sum=simd_sum(sum);
        float a=query || key ? min(1.f,qt/(precise::sqrt(sum)+1e-6f)) : precise::rsqrt(sum/128.f+eps);
        for (uint j=0;j<4;++j) {
            uint d=head*128+lane*4+j;
            bfloat16_t value=bfloat16_t(state[j]*a);
            if (query) { q[d]=value;scaled[d]=attention_scale*float(value); }
            else pair[(key ? 0 : 1024)+d]=value;
        }
"""
def _make_kernel(distinct):
    inputs=["x","payload","qs","rs","qt","rt","eps","attention_scale","kh","vh","kw","vw","table"]
    prefix="#pragma clang fp contract(off)\nthreadgroup bfloat16_t raw[128]; uint sg=simdgroup_index_in_threadgroup;\n"
    prefix+="bool is_gate=threadgroup_position_in_grid.x>=27 && threadgroup_position_in_grid.x<51;\n"
    projection=_dot_source.replace("out[group*4+r]=bfloat16_t(dot);","""{
                bfloat16_t value=bfloat16_t(dot);
                if (is_gate) {
                    ushort code=as_type<ushort>(value);
                    gate[(group-27*32)*4+r]=table[((uint(code)&32767u)<<1)|(code>>15)];
                } else raw[sg*4+r]=value;
            }""")+"\nif (is_gate) return;\n"
    if distinct:
        inputs+=["kx","vx"]
        prefix+="uint bank=threadgroup_position_in_grid.x; auto input=bank<51 ? x : bank<59 ? kx : vx;\n"
        projection=projection.replace("(x+d)","(input+d)")
    return mx.fast.metal_kernel(
        name="inkling_project_recurrent_heads" if distinct else "inkling_project_complete_heads",input_names=inputs,
        output_names=["q","scaled","rel","gate","pair","k_window","v_window"],source=prefix+projection+_epilogue)


_kernel=_make_kernel(False)
_distinct=_make_kernel(True)


def compute(layer,x,packed,tiles,conv_states,conv_mask,kv,extra=()):
    if (not getattr(kv,"grouped",False) or layer.sliding_window!=1024 or conv_mask is not None
            or tiles is None or tiles[0] is not packed or x.dtype!=mx.bfloat16 or packed.shape!=(8576,2304)
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
        inputs=[x,*tiles[1],layer.q_l2_clip.scale,layer.rel_l2_clip.scale,scalar(layer.q_l2_clip.threshold),
                scalar(layer.rel_l2_clip.threshold),scalar(layer.eps),scalar(layer.scaling),*conv_states[:2],kw,vw,gate_probabilities(),*extra],
        template=[("W",kw.dtype)],grid=(8576*8,1,1),threadgroup=(1024,1,1),
        output_shapes=[(1,24,1,128),(1,24,1,128),(1,1,24,16),(1,1,3072),(1,16,1,128),(1,4,1024),(1,4,1024)],
        output_dtypes=[mx.bfloat16,mx.float32,mx.bfloat16,mx.float32,mx.bfloat16,mx.float32,mx.float32])
    conv_states[:2]=kh,vh
    kv.defer_update=True
    k,v=kv.update_packed(pair)
    return q,k,v,rel,gate,scaled
