"""Compile the two projections from each cross-pass message bottleneck together."""
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.inkling2 import layers

_reference = layers.MessageGate.__call__


def project(layer, hidden_states, message, normalized=None, reduced=None, with_sums=False):
    projections = (layer.down_proj, layer.forget_gate_proj, layer.message_gate_proj)
    if (hidden_states.ndim != 3 or hidden_states.shape != message.shape
            or hidden_states.shape[0] != 1 or hidden_states.shape[-1] != 2304
            or hidden_states.dtype != mx.float32 or message.dtype != mx.float32
            or any(type(p) is not nn.Linear or "bias" in p for p in projections)):
        return _reference(layer, hidden_states, message)
    down, forget, gain = (p.weight for p in projections)
    if (down.shape != (64, 4608) or down.dtype != mx.bfloat16
            or forget.shape != (2304, 64) or gain.shape != forget.shape
            or forget.dtype != mx.float32 or gain.dtype != mx.float32):
        return _reference(layer, hidden_states, message)

    if hidden_states.shape[1] >= 8:
        from . import prefill_message
        reduced = prefill_message.normalized_inputs(hidden_states, message) @ down.T
        return prefill_message.apply(reduced, forget, gain, hidden_states, message)
    if hidden_states.shape[1] == 1 and normalized is not None:
        from . import prepared_reduce
        from .gain import apply
        if reduced is None:
            reduced = prepared_reduce.reduce(hidden_states, normalized, (down,) * 3, widen=True)
        return apply(reduced, (forget,) * 3, (gain,) * 3, hidden_states, message, with_sums)
    return _reference(layer, hidden_states, message)


def install():
    layers.MessageGate.__call__ = project
