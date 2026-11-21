"""Compile the two projections from each cross-pass message bottleneck together."""
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.inkling2 import layers

from .gain import apply
from .exact_sources import native

from . import paired_norm, reduce_message

_reference = layers.MessageGate.__call__


@mx.compile
def _project(hidden, message, down, paired):
    if (hidden.shape[-1] == 2304 and hidden.dtype == message.dtype == mx.float32
            and down.dtype == mx.bfloat16 and down.shape[-2:] == (64,4608)):
        reduced = reduce_message.project(hidden,message,down)
        return apply(reduced,paired,hidden,message)
    if hidden.shape[-1] == 2304 and hidden.dtype == message.dtype == mx.float32:
        joined = paired_norm.join(hidden, message, down.dtype)
    else:
        joined = mx.concatenate(
            (layers.unweighted_rms_norm(hidden), layers.unweighted_rms_norm(message)), axis=-1
        )
    reduced = joined.astype(down.dtype) @ down.T
    if reduced.shape[-1] == 64 and hidden.shape[-1] == 2304 and paired.dtype in (mx.bfloat16, mx.float32):
        return apply(reduced, paired, hidden, message)
    gates = mx.tanh(reduced.astype(mx.float32) @ paired.T)
    forget, gain = mx.split(gates, 2, axis=-1)
    return (1 - forget) * hidden + gain * message


def project(layer, hidden_states, message):
    if hidden_states.shape[:2] != (1, 1) or any(
        type(p) is not nn.Linear or "bias" in p
        for p in (layer.down_proj, layer.forget_gate_proj, layer.message_gate_proj)
    ):
        return _reference(layer, hidden_states, message)
    forget = layer.forget_gate_proj.weight
    gain = layer.message_gate_proj.weight
    cached = getattr(layer, "_paired_weights", None)
    if cached is None or cached[0] is not forget or cached[1] is not gain:
        cached = (forget, gain, mx.concatenate((native(layer.forget_gate_proj), native(layer.message_gate_proj)), axis=0))
        object.__setattr__(layer, "_paired_weights", cached)
    return _project(hidden_states, message, layer.down_proj.weight, cached[2])


def install():
    layers.MessageGate.__call__ = project
