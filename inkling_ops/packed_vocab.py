"""Lossless block exponent coding for batch-one BF16 vocabulary projection."""
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.inkling2.model import Model, TextModel
from mlx_lm.models.inkling2.layers import BlockAttentionResidual

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
    text = model.model.language_model
    if (type(text) is TextModel and text.pipeline_size==1 and text.pipeline_rank==0
            and cache is not None and cache[0] is not None and text.args.loop_config is not None
            and type(text.norm) is nn.RMSNorm and text.norm.weight.shape==(2304,)
            and type(text.final_input) is BlockAttentionResidual and text.final_input.query.shape==(2304,)
            and text.final_input.query.dtype==mx.float32):
        hidden=text.embed(inputs) if input_embeddings is None else input_embeddings
        if hidden.shape==(1,1,2304) and hidden.dtype==mx.bfloat16:
            from .depth_norm import final_hidden
            raw,blocks=text._run_loop(hidden,cache,cache[0][1].make_mask(1))
            h=final_hidden(text,raw,blocks)
        else:
            h=text(inputs,cache=cache,input_embeddings=hidden)
    else:
        h=text(inputs,cache=cache,input_embeddings=input_embeddings)
    logits = project(h,head) if h.dtype == mx.bfloat16 else head(h)
    cap = model.args.text.final_logit_softcapping
    return (mx.tanh(logits.astype(mx.float32)/cap)*cap)[...,:model.args.text.unpadded_vocab_size]


def install():
    Model.__call__ = call
