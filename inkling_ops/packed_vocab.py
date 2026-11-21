"""Lossless block exponent coding for batch-one BF16 vocabulary projection."""
from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.inkling2.model import Model

_reference = Model.__call__

from . import packed_rows


@lru_cache(maxsize=8)
def softcap_table(cap):
    values=mx.arange(65536,dtype=mx.uint32).astype(mx.uint16).view(mx.bfloat16).astype(mx.float32)
    table=mx.tanh(values/cap)*cap
    mx.eval(table)
    return table


def project(x, head, cap, count):
    w = head.weight
    n, k = w.shape
    saved = getattr(head, "_encoded_vocab", None)
    if saved is None or saved[0] is not w:
        packed = packed_rows.encode(w)
        saved = (w, packed)
        object.__setattr__(head, "_encoded_vocab", saved)
    return packed_rows.vocabulary(x,w,saved[1],softcap_table(cap),count)


def call(model, inputs, cache=None, input_embeddings=None, pixel_values=None):
    head = model.lm_head
    if (inputs.shape != (1,1) or pixel_values is not None or model.sharding_group is not None
            or type(head) is not nn.Linear or "bias" in head or head.weight.dtype != mx.bfloat16
            or head.weight.shape[1] != 2304 or head.weight.shape[0] % 16):
        return _reference(model,inputs,cache=cache,input_embeddings=input_embeddings,pixel_values=pixel_values)
    h = model.model.language_model(inputs,cache=cache,input_embeddings=input_embeddings)
    cap = model.args.text.final_logit_softcapping
    count = model.args.text.unpadded_vocab_size
    if h.dtype==mx.bfloat16 and 0<count<=head.weight.shape[0]:
        return project(h,head,cap,count)
    logits=head(h)
    return (mx.tanh(logits.astype(mx.float32)/cap)*cap)[...,:model.args.text.unpadded_vocab_size]


def install():
    Model.__call__ = call
