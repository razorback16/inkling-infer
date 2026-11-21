"""Finish NVFP4 expert mixing, temporal convolution and residual in one launch."""
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.inkling2 import layers

from . import postmix, depth_norm, expert_gate, router

_reference = layers.DecoderLayer.__call__


def decode(layer, hidden_states, block_states, *, cache=None, conv_mask=None, messages=None):
    from inkling_kernel import _finish_mix, can_fuse_down
    from .compiled_ring import _DeviceRing
    normalized_messages = None
    if isinstance(messages,tuple):
        messages, normalized_messages = messages
    moe = layer.mlp
    post = layer.post_mlp_convolution
    down = moe.experts.down_proj
    fused_down = can_fuse_down(down, hidden_states)
    if (hidden_states.shape != (1,1,2304) or hidden_states.dtype != mx.bfloat16 or conv_mask is not None
            or cache is None or type(cache[0]) is not _DeviceRing or type(post) is not layers.ShortConvolution
            or post.kernel_size != 4 or moe.sharding_group is not None or moe.gate.top_k != 8
            or (not fused_down and getattr(down, "mode", None) != "mxfp8")):
        return _reference(layer, hidden_states, block_states, cache=cache, conv_mask=conv_mask, messages=messages)
    history = list(cache[1].cache)
    normalized_inputs = (depth_norm.attention_inputs(layer,block_states,hidden_states)
                         if messages is None and layer.attention_input is not None else None)
    if normalized_inputs is not None:
        states, normed = normalized_inputs
        attention = layer.self_attn(*states, cache=cache, conv_states=history, conv_mask=None)
    else:
        if layer.attention_input is None:
            normed = layers.unweighted_rms_norm(hidden_states.astype(mx.float32))
            inputs = None
        else:
            inputs, normed = depth_norm.evaluate(layer.attention_input, block_states, hidden_states)
        payload = (messages, normalized_messages) if normalized_messages is not None else messages
        attention = layer._self_attention(hidden_states,inputs,cache,history,None,payload)
    if layer.is_block_boundary:
        block = normed[None]
        block_states = block if block_states is None else mx.concatenate((block_states, block))
    prenormalized = False
    if type(layer.post_attention_convolution) is layers.ShortConvolution and layer.post_attention_convolution.kernel_size == 4:
        prenormalized = messages is None and type(layer.post_attention_layernorm) is nn.RMSNorm
        mixed, residual, history[2] = depth_norm.post_attention(
            layer,block_states,attention,history[2],hidden_states,normalize=prenormalized)
    else:
        filtered, history[2] = layer.post_attention_convolution(attention,history[2],None)
        residual = filtered.astype(mx.float32)
        if not layer.is_block_boundary:
            residual = hidden_states.astype(mx.float32) + residual
        mixed, _ = depth_norm.evaluate(layer.mlp_input, block_states, residual)
    x = mixed[...,0,:]
    if normalized_messages is not None and type(layer.mlp_message_gate) is layers.MessageGate:
        from . import message_gate
        x = message_gate.project(layer.mlp_message_gate,x,messages[...,3,:],normalized_messages[...,3,:])
    elif messages is not None:
        x = layer.mlp_message_gate(x,messages[...,3,:])
    x, selected, routing = router.route(layer, x, hidden_states.dtype, normalized=prenormalized)
    active = expert_gate.gated_activation(moe.experts,x,selected)
    w = post.conv1d.weight
    if fused_down:
        out, next_history = _finish_mix(
            inputs=[active, selected, routing, down.weight, down.scales, down.global_scale,
                    history[3], w, residual],
            template=[("T", x.dtype), ("W",w.dtype)], grid=(576*256,1,1), threadgroup=(256,1,1),
            output_shapes=[x.shape, (1,3,2304)], output_dtypes=[x.dtype,mx.float32])
    else:
        states = down(active, selected).squeeze(-2)
        out, next_history = postmix.finish(states, routing, history[3], w, residual)
    cache[1].cache = [layers._conv_state(h,None,3) for h in history[:3]] + [next_history]
    return out, block_states
