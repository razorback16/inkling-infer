"""Losslessly pack BF16 exponent groups for the output projection at decode."""
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.inkling2.model import Model

_reference = Model.__call__
_kernel = mx.fast.metal_kernel(
    name="inkling_compact_unembed", input_names=["x", "mant", "low", "base", "wide"],
    output_names=["out"],
    source="""
        uint row = thread_position_in_grid.x / 32;
        uint lane = thread_index_in_simdgroup;
        float sum = 0;
        for (uint j = lane; j < 288; j += 32) {
            uint word = row * 288 + j;
            uint tag = base[word / 4];
            uint packed = ((const device uint *)low)[word];
            uchar4 m0 = ((const device uchar4 *)mant)[word*2];
            uchar4 m1 = ((const device uchar4 *)mant)[word*2+1];
            for (uint i = 0; i < 8; ++i) {
                uint exp = tag < 256 ? tag + ((packed >> (4*i)) & 15)
                    : wide[(tag-256)*32+(j%4)*8+i];
                uint m = i < 4 ? m0[i] : m1[i-4];
                uint bits = ((m & 128) << 24) | (exp << 23) | ((m & 127) << 16);
                sum += float(x[j*8+i]) * as_type<float>(bits);
            }
        }
        sum = simd_sum(sum);
        if (lane == 0) out[row] = bfloat16_t(sum);
    """,
)


def pack(weight):
    rows, width = weight.shape
    source = np.asarray(weight.view(mx.uint16))
    mantissa = np.empty((rows, width), np.uint8)
    low = np.empty((rows, width // 2), np.uint8)
    tags = np.empty((rows, width // 32), np.uint32)
    escapes = []
    start = 0
    for i in range(0, rows, 1024):
        bits = source[i:i+1024]
        n = len(bits)
        mantissa[i:i+n] = (bits & 127) | ((bits >> 8) & 128)
        exps = ((bits >> 7) & 255).astype(np.uint8).reshape(n, -1, 32)
        base = exps.min(axis=-1)
        large = exps.max(axis=-1) - base > 15
        tag = base.astype(np.uint32)
        exceptions = exps[large]
        tag[large] = np.arange(start+256, start+256+len(exceptions), dtype=np.uint32)
        if len(exceptions):
            escapes.append(exceptions)
        start += len(exceptions)
        tags[i:i+n] = tag
        deltas = (exps - base[...,None]).reshape(n, width)
        low[i:i+n] = (deltas[:, ::2] & 15) | (deltas[:, 1::2] << 4)
    extra = np.concatenate(escapes) if escapes else np.zeros((1,32), np.uint8)
    return tuple(mx.array(a) for a in (mantissa, low, tags, extra))


def call(model, inputs, cache=None, input_embeddings=None, pixel_values=None):
    if (inputs.shape != (1,1) or pixel_values is not None or model.sharding_group is not None
            or type(model.lm_head) is not nn.Linear or "bias" in model.lm_head
            or model.lm_head.weight.dtype != mx.bfloat16 or model.lm_head.weight.shape[1] != 2304):
        return _reference(model, inputs, cache, input_embeddings, pixel_values)
    hidden = model.model.language_model(inputs, cache=cache, input_embeddings=input_embeddings)
    w = model.lm_head.weight
    if hidden.dtype == mx.bfloat16:
        stored = getattr(model, "_compact_lm_head", None)
        if stored is None or stored[0] is not w:
            stored = (w, pack(w))
            object.__setattr__(model, "_compact_lm_head", stored)
        logits = _kernel(inputs=[hidden, *stored[1]], grid=(w.shape[0]*32,1,1),
                         threadgroup=(128,1,1), output_shapes=[(1,1,w.shape[0])],
                         output_dtypes=[hidden.dtype])[0]
    else:
        logits = model.lm_head(hidden)
    t = model.args.text
    capped = mx.tanh(logits.astype(mx.float32)/t.final_logit_softcapping) * t.final_logit_softcapping
    return capped[..., :t.unpadded_vocab_size]


def install():
    Model.__call__ = call
