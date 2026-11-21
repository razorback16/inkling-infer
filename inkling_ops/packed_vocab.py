"""Lossless block exponent coding for batch-one BF16 vocabulary projection."""
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.inkling2.model import Model

_reference = Model.__call__

from . import packed_rows


def project(x, head):
    w = head.weight
    n, k = w.shape
    saved = getattr(head, "_encoded_vocab", None)
    if saved is None or saved[0] is not w:
        packed = packed_rows.encode(w)
        saved = (w, packed)
        object.__setattr__(head, "_encoded_vocab", saved)
    return packed_rows.project(x,w,saved[1])


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
