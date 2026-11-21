"""Project the three messaging banks directly from their checkpoint tensors."""
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.inkling2 import layers

from . import prepared_reduce
from .gain import apply
_reference = layers.DecoderLayer._mix_messages


def mix_messages(layer, states, messages, packed_states=None, inverse=None, with_sums=False):
    raw, normalized = messages if isinstance(messages,tuple) else (messages,None)
    gates=(layer.q_message_gate, layer.k_message_gate, layer.v_message_gate)
    if (normalized is None or states[0].shape!=(1,1,2304) or states[0].dtype!=mx.float32
            or any(type(p) is not nn.Linear or "bias" in p or p.weight.shape!=shape or p.weight.dtype!=dtype
                   for g in gates for p,shape,dtype in ((g.down_proj,(64,4608),mx.bfloat16),
                       (g.forget_gate_proj,(2304,64),mx.float32),(g.message_gate_proj,(2304,64),mx.float32)))):
        return _reference(layer,states,raw)
    x=packed_states if packed_states is not None else mx.concatenate(states,axis=0)
    msg=raw.reshape(4,1,2304)[:3]
    norm=normalized.reshape(4,1,2304)[:3]
    reduced=prepared_reduce.reduce(x,norm,[g.down_proj.weight for g in gates],inverse,widen=True)
    result=apply(reduced,[g.forget_gate_proj.weight for g in gates],
                 [g.message_gate_proj.weight for g in gates],x,msg,with_sums)
    if with_sums:
        outputs,sums=result
        return tuple(mx.split(outputs,3,axis=0)),sums
    return tuple(mx.split(result,3,axis=0))


def install():
    layers.DecoderLayer._mix_messages = mix_messages
