"""Prepare all four learned messages for every layer sharing a two-source bank."""
from contextvars import ContextVar

import mlx.core as mx
from mlx_lm.models.inkling2 import layers

_trace = ContextVar("inkling_message_bank", default=None)
_reference = layers.MessageInput.__call__
_kernel = mx.fast.metal_kernel(
    name="inkling_prepare_message_bank",
    input_names=["block","partial","queries"], output_names=["raw","normalized"],
    source="""
        uint q=threadgroup_position_in_grid.x, tid=thread_position_in_threadgroup.x;
        uint lane=thread_index_in_simdgroup, sg=simdgroup_index_in_threadgroup;
        threadgroup float parts[2][9], weights[2], sums[18];
        if (tid<288) {
            float a=0,b=0;
            for (uint d=tid; d<2304; d+=288) {
                float w=queries[q*2304+d];
                a+=block[d]*w; b+=partial[d]*w;
            }
            a=simd_sum(a); b=simd_sum(b);
            if (!lane) {parts[0][sg]=a; parts[1][sg]=b;}
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!sg) {
            float logit=-INFINITY;
            if (lane<2) {
                float dot=0;
                for (uint i=0; i<9; ++i) dot+=parts[lane][i];
                logit=dot*(1.f/48.f);
            }
            float w=metal::exp(logit-simd_max(logit));
            float sum=simd_shuffle(w,0)+simd_shuffle(w,1);
            if (lane<2) weights[lane]=w/sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float4 values; float sum=0;
        for (uint i=0; i<4; ++i) {
            uint d=tid*4+i;
            float v=0;
            v+=block[d]*weights[0]; v+=partial[d]*weights[1];
            values[i]=v; raw[q*2304+d]=v; sum+=v*v;
        }
        sum=simd_sum(sum);
        if (!lane) sums[sg]=sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float inv=precise::rsqrt(simd_sum(lane<18 ? sums[lane]:0.f)/2304.f+1e-6f);
        for (uint i=0;i<4;++i) normalized[q*2304+tid*4+i]=values[i]*inv;
    """,
)


def messages(layer, bank, index):
    entries = _trace.get()
    if (entries is None or bank[0].shape != (1,1,1,2304) or bank[1].shape != (1,1,2304)
            or layer.query.shape[1:] != (4,2304) or layer.query.dtype != mx.float32):
        return _reference(layer,bank,index)
    key = (id(layer),id(bank))
    if key not in entries:
        count = layer.query.shape[0]
        outputs = _kernel(inputs=[*bank,layer.query], grid=(count*4*576,1,1),
                          threadgroup=(576,1,1), output_shapes=[(count,1,1,4,2304)]*2,
                          output_dtypes=[mx.float32]*2)
        entries[key] = (bank, outputs)
    raw, normalized = entries[key][1]
    return raw[index], normalized[index]


def run(reference, model, x, caches):
    token = _trace.set({})
    try:
        return reference(model, x, caches, None)
    finally:
        _trace.reset(token)


def install():
    layers.MessageInput.__call__ = messages
