"""Compile the two projections from each cross-pass message bottleneck together."""
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.inkling2 import layers

from .gain import apply
from .exact_sources import native

from . import paired_norm, prepared_reduce

_reference = layers.MessageGate.__call__


@mx.compile
def _project(hidden, message, down, paired, normalized=None):
    if prepared_reduce.applicable(hidden,normalized,down,paired):
        return apply(prepared_reduce.reduce(hidden,normalized,down,widen=True),paired,hidden,message)
    if hidden.shape[-1] == 2304 and hidden.dtype == message.dtype == mx.float32:
        joined = paired_norm.join(hidden, message if normalized is None else normalized, down.dtype, normalized is not None)
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


@mx.compile
def _prompt_bottleneck(hidden,message,weight):
    states=mx.concatenate((layers.unweighted_rms_norm(hidden),layers.unweighted_rms_norm(message)),axis=-1)
    return states.astype(weight.dtype) @ weight.T


@mx.compile
def _prompt_message(hidden, message, down, paired):
    states=mx.concatenate((layers.unweighted_rms_norm(hidden),layers.unweighted_rms_norm(message)),axis=-1)
    reduced=(states.astype(down.dtype) @ down.T).astype(mx.float32)
    forget,gain=mx.split(mx.tanh(reduced @ paired.T),2,axis=-1)
    return (1-forget)*hidden+gain*message


def project(layer, hidden_states, message, normalized=None, reduced=None, with_sums=False):
    if (hidden_states.ndim==3 and hidden_states.shape[1]>=8 and hidden_states.shape==message.shape
            and hidden_states.shape[-1]==2304 and hidden_states.dtype==message.dtype==mx.float32
            and all(type(p) is nn.Linear and "bias" not in p
                    for p in (layer.down_proj,layer.forget_gate_proj,layer.message_gate_proj))):
        forget=layer.forget_gate_proj.weight
        gain=layer.message_gate_proj.weight
        if forget.shape==gain.shape==(2304,64) and forget.dtype==gain.dtype==mx.float32:
            down=layer.down_proj.weight
            if hidden_states.shape[0]==1 and down.dtype==mx.bfloat16 and down.shape==(64,4608):
                from . import prefill_message
                saved=getattr(layer,"_prompt_gate_fragments",None)
                if saved is None or saved[0] is not forget or saved[1] is not gain:
                    saved=(forget,gain,prefill_message.pack(forget,gain))
                    object.__setattr__(layer,"_prompt_gate_fragments",saved)
                reduced=_prompt_bottleneck(hidden_states,message,down)
                return prefill_message.apply(reduced,saved[2],hidden_states,message)
            saved=getattr(layer,"_prompt_gate_rows",None)
            if saved is None or saved[0] is not forget or saved[1] is not gain:
                saved=(forget,gain,mx.concatenate((forget,gain)))
                object.__setattr__(layer,"_prompt_gate_rows",saved)
            return _prompt_message(hidden_states,message,layer.down_proj.weight,saved[2])
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
    if reduced is not None and cached[2].shape==(4608,64) and cached[2].dtype in (mx.bfloat16,mx.float32):
        return apply(reduced,cached[2],hidden_states,message,with_sums)
    return _project(hidden_states, message, layer.down_proj.weight, cached[2], normalized)


def install():
    layers.MessageGate.__call__ = project
