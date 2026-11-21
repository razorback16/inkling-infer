"""Lossless block exponent coding for batch-one BF16 vocabulary projection."""
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.inkling2.model import Model

_reference = Model.__call__

_pack = mx.fast.metal_kernel(
    name="inkling_encode_four_vocab_rows",
    input_names=["weight"], output_names=["payload","bases"],
    source="""
        uint lane=thread_index_in_simdgroup, tile=thread_position_in_grid.x/32;
        uint group=tile/18, block=tile%18;
        uint4 low;
        ushort4 high;
        uchar4 base;
        for (uint row=0;row<4;++row) {
            uint i=(4*group+row)*2304+block*128+lane*4;
            ushort4 bits=*(device const ushort4*)((device const ushort*)weight+i);
            ushort4 e=(bits>>7)&255;
            ushort lo=simd_min(min(min(e.x,e.y),min(e.z,e.w)));
            ushort hi=simd_max(max(max(e.x,e.y),max(e.z,e.w)));
            base[row]=hi-lo>15 ? 255 : lo;
            low[row]=as_type<uint>(uchar4((bits<<1)|(bits>>15)));
            ushort4 n=(e-lo)&15;
            high[row]=n.x|(n.y<<4)|(n.z<<8)|(n.w<<12);
        }
        *(device uint4*)(payload+tile*192+lane*4)=low;
        *(device ushort4*)(payload+tile*192+128+lane*2)=high;
        if (!lane) bases[tile]=as_type<uint>(base);
    """,
)
_dot = mx.fast.metal_kernel(
    name="inkling_four_row_vocab_gemv", input_names=["x","weight","payload","bases"],output_names=["out"],
    source="""
        uint lane=thread_index_in_simdgroup, group=thread_position_in_grid.x/32;
        float sums[4]={};
        for (uint block=0;block<18;++block) {
            uint tile=group*18+block;
            uint4 low=*(device const uint4*)(payload+tile*192+lane*4);
            ushort4 high=*(device const ushort4*)(payload+tile*192+128+lane*2);
            uchar4 base=as_type<uchar4>(bases[tile]);
            uint d=block*128+lane*4;
            float4 v=float4(*(device const bfloat4*)(x+d));
            for (uint r=0;r<4;++r) {
                ushort4 bits;
                if (base[r]==255) {
                    bits=*(device const ushort4*)((device const ushort*)weight+(group*4+r)*2304+d);
                } else {
                    ushort h=high[r];
                    ushort4 ex=ushort4(h,h>>4,h>>8,h>>12)&15;
                    ushort4 rotated=((ex+base[r])<<8)|ushort4(as_type<uchar4>(low[r]));
                    bits=(rotated>>1)|(rotated<<15);
                }
                float4 w=float4(as_type<bfloat4>(bits));
                for (uint j=0;j<4;++j) sums[r]=fma(v[j],w[j],sums[r]);
            }
        }
        for (uint r=0;r<4;++r) {
            float dot=sums[r];
            for (ushort step=16;step;step>>=1) dot+=simd_shuffle_down(dot,step);
            if (!lane) out[group*4+r]=bfloat16_t(dot);
        }
    """,
)


def project(x, head):
    w = head.weight
    n, k = w.shape
    saved = getattr(head, "_encoded_vocab", None)
    if saved is None or saved[0] is not w:
        packed = _pack(inputs=[w], grid=(n*k//16,1,1), threadgroup=(128,1,1),
                       output_shapes=[(n*k//512,192),(n*k//512,)],
                       output_dtypes=[mx.uint32,mx.uint32])
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
