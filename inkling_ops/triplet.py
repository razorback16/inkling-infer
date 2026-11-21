"""Evaluate the three independent cross-pass attention gates as batched matvecs."""
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.inkling2 import layers

from .gain import apply
from .exact_sources import native

from . import paired_norm, prepared_reduce

_reference = layers.DecoderLayer._mix_messages


@mx.compile
def _apply(states, messages, down, up, normalized=None, prepared_state=None):
    if prepared_reduce.applicable(states,normalized,down,up):
        return apply(prepared_reduce.reduce(states,normalized,down,prepared_state),up,states,messages)
    norm = layers.unweighted_rms_norm
    if states.shape[-1] == 2304 and states.dtype == messages.dtype == mx.float32:
        joined = paired_norm.join(states, messages if normalized is None else normalized, down.dtype, normalized is not None)
    else:
        joined = mx.concatenate((norm(states), norm(messages)), axis=-1).astype(down.dtype)
    reduced = joined @ down.swapaxes(-1, -2)
    if reduced.shape[-1] == 64 and states.shape[-1] == 2304 and up.dtype in (mx.bfloat16, mx.float32):
        return apply(reduced, up, states, messages)
    gates = mx.tanh(reduced.astype(mx.float32) @ up.swapaxes(-1, -2))
    forget, gain = mx.split(gates, 2, axis=-1)
    return (1 - forget) * states + gain * messages


def mix_messages(layer, states, messages, packed_states=None, prepared_state=None):
    normalized = None
    if isinstance(messages,tuple):
        messages, normalized = messages
    if states[0].shape[:2] != (1, 1):
        return _reference(layer, states, messages)
    gates = (layer.q_message_gate, layer.k_message_gate, layer.v_message_gate)
    projections = tuple(p for gate in gates for p in
                        (gate.down_proj, gate.forget_gate_proj, gate.message_gate_proj))
    if any(type(p) is not nn.Linear or "bias" in p for p in projections):
        return _reference(layer, states, messages)
    weights = tuple(p.weight for p in projections)
    cache = getattr(layer, "_message_triplet", None)
    if cache is None or not all(a is b for a, b in zip(weights, cache[0])):
        down = mx.stack(weights[0::3])
        up = mx.stack([mx.concatenate((native(g.forget_gate_proj), native(g.message_gate_proj))) for g in gates])
        cache = (weights, down, up)
        object.__setattr__(layer, "_message_triplet", cache)
    x = (packed_states if packed_states is not None and packed_states.dtype==mx.float32
         and messages.dtype==mx.float32 and packed_states.shape[-1]==2304
         else mx.concatenate(states,axis=0))
    msg = messages.reshape(4, 1, -1)[:3]
    prepared = None if normalized is None else normalized.reshape(4,1,-1)[:3]
    outputs = _apply(x, msg, cache[1], cache[2], prepared, prepared_state)
    return tuple(mx.split(outputs, 3, axis=0))


def install():
    layers.DecoderLayer._mix_messages = mix_messages
