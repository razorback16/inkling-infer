"""Lossless block exponent coding for batch-one BF16 vocabulary projection."""
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.inkling2.model import Model

_reference = Model.__call__

_pack = mx.fast.metal_kernel(
    name="inkling_pack_vocab_exponents",
    input_names=["weight"], output_names=["mantissas", "deltas", "bases"],
    source="""
        uint lane = thread_index_in_simdgroup;
        uint i = thread_position_in_grid.x * 4;
        ushort4 raw = as_type<ushort4>(*(device const bfloat4_t*)(weight+i));
        ushort4 exps = (raw >> 7) & 255;
        ushort hi = simd_max(max(max(exps.x,exps.y), max(exps.z,exps.w)));
        ushort lo = simd_min(min(min(exps.x,exps.y), min(exps.z,exps.w)));
        if (lane==0) bases[i/128] = hi-lo>15 ? 255 : lo;
        uchar4 mant = uchar4((raw & 127) | ((raw >> 8) & 128));
        *(device uchar4*)(mantissas+i) = mant;
        ushort4 ds = (exps-lo) & 15;
        deltas[i/4] = ds.x | (ds.y<<4) | (ds.z<<8) | (ds.w<<12);
    """.replace("bfloat4_t","bfloat4"),
)
_dot = mx.fast.metal_kernel(
    name="inkling_lossless_vocab_gemv",
    input_names=["x","weight","mantissas","deltas","bases"], output_names=["out"],
    source="""
        uint lane=thread_index_in_simdgroup;
        uint first=(thread_position_in_grid.x/32)*4;
        float acc[4]={};
        for (uint k=0; k<2304; k+=128) {
            uint d=k+4*lane;
            float4 input=float4(*(device const bfloat4*)(x+d));
            for (uint r=0; r<4; ++r) {
                uint i=(first+r)*2304+d;
                ushort base=bases[i/128];
                ushort4 values;
                if (base==255) {
                    values=*(device const ushort4*)((device const ushort*)weight+i);
                } else {
                    ushort4 m=ushort4(*(device const uchar4*)(mantissas+i));
                    ushort bits=deltas[i/4];
                    ushort4 ex=ushort4(bits,bits>>4,bits>>8,bits>>12)&15;
                    values=((m&128)<<8) | ((base+ex)<<7) | (m&127);
                }
                float4 w=float4(as_type<bfloat4>(values));
                for (uint j=0;j<4;++j) acc[r]=fma(input[j],w[j],acc[r]);
            }
        }
        for (uint r=0; r<4; ++r) {
            float dot=acc[r];
            for (ushort s=16; s; s>>=1) dot+=simd_shuffle_down(dot,s);
            if (!lane) out[first+r]=bfloat16_t(dot);
        }
    """,
)


def project(x, head):
    w = head.weight
    n, k = w.shape
    saved = getattr(head, "_encoded_vocab", None)
    if saved is None or saved[0] is not w:
        packed = _pack(inputs=[w], grid=(n*k//4,1,1), threadgroup=(128,1,1),
                       output_shapes=[(n,k),(n,k//4),(n,k//128)],
                       output_dtypes=[mx.uint8,mx.uint16,mx.uint8])
        saved = (w, packed)
        object.__setattr__(head, "_encoded_vocab", saved)
    return _dot(inputs=[x,w,*saved[1]], grid=(n*8,1,1), threadgroup=(128,1,1),
                output_shapes=[(1,1,n)], output_dtypes=[x.dtype])[0]


def call(model, inputs, cache=None, input_embeddings=None, pixel_values=None):
    head = model.lm_head
    if (inputs.shape != (1,1) or pixel_values is not None or model.sharding_group is not None
            or type(head) is not nn.Linear or "bias" in head or head.weight.dtype != mx.bfloat16
            or head.weight.shape[1] != 2304 or head.weight.shape[0] % 16):
        return _reference(model,inputs,cache=cache,input_embeddings=input_embeddings,pixel_values=pixel_values)
    h = model.model.language_model(inputs,cache=cache,input_embeddings=input_embeddings)
    logits = project(h,head) if h.dtype == mx.bfloat16 else head(h)
    cap = model.args.text.final_logit_softcapping
    return (mx.tanh(logits.astype(mx.float32)/cap)*cap)[...,:model.args.text.unpadded_vocab_size]


def install():
    Model.__call__ = call
