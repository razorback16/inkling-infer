"""Finish the two prompt residuals while retaining three convolution samples."""
from contextvars import ContextVar

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache
from mlx_lm.models.inkling2 import layers

_tail_only = ContextVar("inkling_prefill_tail_only", default=False)

_finish = mx.fast.metal_kernel(
    name="inkling_prompt_convolution_residual",
    input_names=["x", "old", "weight", "residual"], output_names=["out", "tail"],
    source="""
        #pragma clang fp contract(off)
        uint c=thread_position_in_grid.x, first=thread_position_in_grid.y*4;
        float4 w=float4(((device const vec<W,4>*)weight)[c]);
        float a=0,b=0,h=0;
        if (first) {
            a=float(x[(first-3)*C+c]); b=float(x[(first-2)*C+c]); h=float(x[(first-1)*C+c]);
        } else if (HISTORY) {
            a=old[c]; b=old[C+c]; h=old[2*C+c];
        }
        for (uint t=first;t<min(first+4u,uint(L));++t) {
            uint i=t*C+c;
            float v=float(x[i]);
            float total=fma(float(W(a)),w.x,0.f);
            total=fma(float(W(b)),w.y,total);
            total=fma(float(W(h)),w.z,total);
            total=fma(float(W(v)),w.w,total);
            float filtered=float(bfloat16_t((float(W(total))+v)*0.4472135954999579f));
            out[i]=ADD ? O(float(residual[i])+filtered) : O(filtered);
            if (t+3>=L) tail[(t+3-L)*C+c]=v;
            a=b;b=h;h=v;
        }
    """,
)


def finish(x, old, conv, residual, add, dtype):
    _, length, width=x.shape
    weight=conv.conv1d.weight
    return _finish(
        inputs=[x,old if old is not None else x,weight,residual],
        template=[("L",length),("C",width),("W",weight.dtype),("O",dtype),("ADD",add),("HISTORY",old is not None)],
        grid=(width,(length+3)//4,1),threadgroup=(128,1,1),
        output_shapes=[x.shape,(1,3,width)],output_dtypes=[dtype,mx.float32],
    )


def applicable(layer, x, cache, mask):
    return (x.ndim==3 and x.shape[0]==1 and x.shape[1]>=3 and x.shape[-1]==2304
            and x.dtype==mx.bfloat16 and cache is not None and mask is None
            and type(cache[1]) is ArraysCache and cache[1].lengths is None and cache[1].left_padding is None
            and type(layer) in (layers.FullDecoderLayer,layers.SlidingDecoderLayer)
            and all(type(c) is layers.ShortConvolution and c.kernel_size==4 and "bias" not in c.conv1d
                    and c.conv1d.weight.shape==(2304,4,1)
                    for c in (layer.post_attention_convolution,layer.post_mlp_convolution))
            and all(v is None or v.shape==(1,3,2304) for v in cache[1].cache[2:]))


def run(layer, x, blocks, cache, messages):
    states = list(cache[1].cache)
    normed = layers.unweighted_rms_norm(x.astype(mx.float32))
    inputs = None if layer.attention_input is None else layer.attention_input(blocks,normed)
    token = _tail_only.set(True)
    try:
        attention = layer._self_attention(x,inputs,cache,states,None,messages)
    finally:
        _tail_only.reset(token)
    residual, states[2] = finish(attention, states[2], layer.post_attention_convolution, x,
                                 not layer.is_block_boundary, mx.float32)
    if layer.is_block_boundary:
        block=normed[None]
        blocks=block if blocks is None else mx.concatenate((blocks,block))
    hidden=layer.mlp_input(blocks,layers.unweighted_rms_norm(residual))[...,0,:]
    if messages is not None:
        hidden=layer.mlp_message_gate(hidden,messages[...,3,:])
    output=layer.mlp(layer.post_attention_layernorm(hidden).astype(x.dtype))
    y, states[3]=finish(output,states[3],layer.post_mlp_convolution,residual,True,x.dtype)
    cache[1].cache=[h if h.shape[1]==3 else layers._conv_state(h,None,3) for h in states[:2]]+states[2:]
    return y,blocks
