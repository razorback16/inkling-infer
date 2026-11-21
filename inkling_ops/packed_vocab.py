"""Lossless block exponent coding for batch-one BF16 vocabulary projection."""
from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.inkling2.model import Model

_reference = Model.__call__

from . import vocab_tiles


@lru_cache(maxsize=8)
def _softcap_table(cap):
    values=mx.arange(65536,dtype=mx.uint32).astype(mx.uint16).view(mx.bfloat16).astype(mx.float32)
    table=mx.contiguous((mx.tanh(values/cap)*cap).reshape(2,32768).T)
    mx.eval(table)
    return table


def project(x, head, table=None):
    w = head.weight
    n, k = w.shape
    saved = getattr(head, "_encoded_vocab", None)
    if saved is None or saved[0] is not w:
        packed = vocab_tiles.encode(w)
        saved = (w, packed)
        object.__setattr__(head, "_encoded_vocab", saved)
    return vocab_tiles.project(x,w,saved[1],table)


def call(model, inputs, cache=None, input_embeddings=None, pixel_values=None):
    head = model.lm_head
    if (inputs.shape != (1,1) or pixel_values is not None or model.sharding_group is not None
            or type(head) is not nn.Linear or "bias" in head or head.weight.dtype != mx.bfloat16
            or head.weight.shape[1] != 2304 or head.weight.shape[0] % 16):
        return _reference(model,inputs,cache=cache,input_embeddings=input_embeddings,pixel_values=pixel_values)
    h = model.model.language_model(inputs,cache=cache,input_embeddings=input_embeddings)
    cap = model.args.text.final_logit_softcapping
    if h.dtype==mx.bfloat16:
        return project(h,head,_softcap_table(cap))[...,:model.args.text.unpadded_vocab_size]
    logits=head(h)
    return (mx.tanh(logits.astype(mx.float32)/cap)*cap)[...,:model.args.text.unpadded_vocab_size]


def install():
    Model.__call__ = call
