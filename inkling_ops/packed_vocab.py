"""Lossless block exponent coding for batch-one BF16 vocabulary projection."""
from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.inkling2.model import Model

_reference = Model.__call__

from . import single_row_tiles


@lru_cache(maxsize=8)
def softcap_table(cap):
    values=mx.arange(65536,dtype=mx.uint32).astype(mx.uint16).view(mx.bfloat16).astype(mx.float32)
    table=mx.tanh(values/cap)*cap
    mx.eval(table)
    return table


_expand=mx.fast.metal_kernel(name="inkling_decode_capped_logits",input_names=["logits","table"],
    output_names=["out"],source="""
        uint i=thread_position_in_grid.x;
        ushort4 code=((device const ushort4*)logits)[i];
        float4 capped=float4(table[code.x],table[code.y],table[code.z],table[code.w]);
        ((device float4*)out)[i]=capped;
    """)


def project(x, head):
    w = head.weight
    n, k = w.shape
    saved = getattr(head, "_encoded_vocab", None)
    if saved is None or saved[0] is not w:
        packed = single_row_tiles.encode(w)
        saved = (w, packed)
        object.__setattr__(head, "_encoded_vocab", saved)
    return single_row_tiles.project(x,w,saved[1])


def call(model, inputs, cache=None, input_embeddings=None, pixel_values=None):
    head = model.lm_head
    if (inputs.shape != (1,1) or pixel_values is not None or model.sharding_group is not None
            or type(head) is not nn.Linear or "bias" in head or head.weight.dtype != mx.bfloat16
            or head.weight.shape[1] != 2304 or head.weight.shape[0] % 16):
        return _reference(model,inputs,cache=cache,input_embeddings=input_embeddings,pixel_values=pixel_values)
    h = model.model.language_model(inputs,cache=cache,input_embeddings=input_embeddings)
    logits = project(h,head) if h.dtype == mx.bfloat16 else head(h)
    cap = model.args.text.final_logit_softcapping
    if logits.dtype==mx.bfloat16:
        capped=_expand(inputs=[logits,softcap_table(cap)],grid=(logits.size//4,1,1),threadgroup=(256,1,1),
                       output_shapes=[logits.shape],output_dtypes=[mx.float32])[0]
    else:
        capped=mx.tanh(logits.astype(mx.float32)/cap)*cap
    return capped[...,:model.args.text.unpadded_vocab_size]


def install():
    Model.__call__ = call
