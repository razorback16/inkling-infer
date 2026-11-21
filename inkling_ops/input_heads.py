"""Normalize all three separate attention inputs in one decode launch."""
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.inkling2 import layers

_full = layers.FullDecoderLayer._self_attention
_sliding = layers.SlidingDecoderLayer._self_attention
_kernel = mx.fast.metal_kernel(
    name="inkling_three_input_norm", input_names=["q","k","v","qw","kw","vw","eps"], output_names=["qo","ko","vo"],
    source="""
        uint h = threadgroup_position_in_grid.x;
        uint tid = thread_position_in_threadgroup.x, lane = thread_index_in_simdgroup;
        uint sg = simdgroup_index_in_threadgroup;
        float4 values;
        float dot = 0;
        for (uint i=0; i<4; ++i) {
            uint d = tid*4+i;
            values[i] = h==0 ? float(q[d]) : h==1 ? float(k[d]) : float(v[d]);
            dot += values[i]*values[i];
        }
        dot = simd_sum(dot);
        threadgroup float sums[18];
        if (lane == 0) sums[sg] = dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        dot = simd_sum(lane < 18 ? sums[lane] : 0.f);
        float scale = precise::rsqrt(dot/2304.f + eps);
        for (uint i=0; i<4; ++i) {
            uint d = tid*4+i;
            float x = float(A(values[i]*scale));
            if (h==0) qo[d] = T(x*float(qw[d]));
            else if (h==1) ko[d] = T(x*float(kw[d]));
            else vo[d] = T(x*float(vw[d]));
        }
    """,
)


_from_sums = mx.fast.metal_kernel(
    name="inkling_scale_message_heads", input_names=["q","k","v","qw","kw","vw","eps","sums"],
    output_names=["qo","ko","vo"],source="""
        uint head=threadgroup_position_in_grid.y, lane=thread_index_in_simdgroup;
        uint first=thread_position_in_grid.x*4;
        float inv=precise::rsqrt(simd_sum(lane<18 ? sums[head*18+lane] : 0.f)/2304.f+eps);
        for (uint j=0;j<4;++j) {
            uint d=first+j;
            float v0=head==0 ? float(q[d]) : head==1 ? float(k[d]) : float(v[d]);
            float n=float(A(v0*inv));
            if (head==0) qo[d]=T(n*float(qw[d]));
            else if (head==1) ko[d]=T(n*float(kw[d]));
            else vo[d]=T(n*float(vw[d]));
        }
    """,
)


def attention(layer, x, inputs, cache, conv_states, conv_mask, messages):
    inverse = None
    if isinstance(inputs,tuple):
        inputs, inverse = inputs
    full = isinstance(layer, layers.FullDecoderLayer)
    ref = _full if full else _sliding
    if x.shape != (1,1,2304) or (not full and messages is None):
        return ref(layer, x, inputs, cache, conv_states, conv_mask, messages[0] if isinstance(messages,tuple) else messages)
    norms = (layer.q_input_layernorm, layer.k_input_layernorm, layer.v_input_layernorm) if full else (layer.input_layernorm,) * 3
    if not all(type(n) is nn.RMSNorm and n.eps == norms[0].eps for n in norms):
        return ref(layer, x, inputs, cache, conv_states, conv_mask, messages[0] if isinstance(messages,tuple) else messages)
    if inputs is None:
        states = (x,) * 3
    elif full:
        states = tuple(inputs[...,i,:] for i in range(3))
    else:
        states = (inputs[...,0,:],) * 3
    sums = None
    if messages is not None:
        from . import triplet
        if getattr(layer._mix_messages,"__func__",None) is triplet.mix_messages:
            source = x if inputs is None else inputs
            result = triplet.mix_messages(layer,states,messages,source.reshape(-1,1,2304),inverse,with_sums=True)
            if len(result)==2:
                states, sums = result
            else:
                states = result
        else:
            states = layer._mix_messages(states,messages)
    weights = [n.weight for n in norms]
    from mlx_lm.kernels.common import scalar
    args = [*states,*weights,scalar(norms[0].eps)]
    if sums is None:
        kernel,grid,group = _kernel,(3*576,1,1),(576,1,1)
    else:
        args.append(sums)
        kernel,grid,group = _from_sums,(576,3,1),(128,1,1)
    q,k,v = kernel(inputs=args,template=[("A",mx.result_type(states[0],weights[0])),("T",x.dtype)],
                   grid=grid,threadgroup=group,output_shapes=[x.shape]*3,output_dtypes=[x.dtype]*3)
    return layer.self_attn(q,k,v,cache=cache,conv_states=conv_states,conv_mask=conv_mask)


def install():
    layers.FullDecoderLayer._self_attention = attention
    layers.SlidingDecoderLayer._self_attention = attention
