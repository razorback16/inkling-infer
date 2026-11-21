"""Pack attention projections with a common one-token input."""
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.inkling2 import layers

from .head_norm import normalize
from . import dynamic_conv, rel_mask

_sliding = layers.SlidingAttention.__call__
_full = layers.FullAttention.__call__


def projection(layer, modules, key):
    weights = tuple(module.weight for module in modules)
    cached = getattr(layer, key, None)
    if cached is None or not all(a is b for a, b in zip(weights, cached[0])):
        boundaries = []
        total = 0
        for weight in weights[:-1]:
            total += weight.shape[0]
            boundaries.append(total)
        cached = (weights, mx.concatenate(weights), boundaries)
        object.__setattr__(layer, key, cached)
    return cached[1:]


def attend(layer, hidden_states, key_hidden_states, value_hidden_states, *, cache, conv_states, conv_mask):
    full = isinstance(layer, layers.FullAttention)
    same = hidden_states is key_hidden_states and hidden_states is value_hidden_states
    common = (layer.q_proj, layer.r_proj, layer.gate_proj, layer.sink_proj)
    modules = common + (layer.k_proj, layer.v_proj) if same else common
    if hidden_states.shape[:2] != (1, 1) or any(type(m) is not nn.Linear or "bias" in m for m in modules):
        reference = _full if full else _sliding
        return reference(layer, hidden_states, key_hidden_states, value_hidden_states,
                          cache=cache, conv_states=conv_states, conv_mask=conv_mask)
    packed, boundaries = projection(layer, modules, "_all_qkv" if same else "_common_q")
    parts = mx.split(hidden_states @ packed.T, boundaries, axis=-1)
    query, relative, gate, sinks = parts[:4]
    paired = (full and not same and conv_mask is None
              and all(type(c) is layers.DynamicShortConvolution and c.kernel_size == 4
                      and type(c.weight_proj) is nn.Linear and "bias" not in c.weight_proj
                      for c in (layer.k_sconv, layer.v_sconv))
              and all(type(p) is nn.Linear and "bias" not in p for p in (layer.k_proj, layer.v_proj))
              and all(h is not None and h.shape == (1, 3, layer.num_key_value_heads * layer.head_dim)
                      for h in conv_states[:2]))
    if same:
        key, value = parts[4:]
    elif paired:
        kp, kb = projection(layer, (layer.k_proj, layer.k_sconv.weight_proj), "_key_filter")
        vp, vb = projection(layer, (layer.v_proj, layer.v_sconv.weight_proj), "_value_filter")
        key, kd = mx.split(key_hidden_states @ kp.T, kb, axis=-1)
        value, vd = mx.split(value_hidden_states @ vp.T, vb, axis=-1)
    else:
        key = layer.k_proj(key_hidden_states)
        value = layer.v_proj(value_hidden_states)
    if full:
        if paired:
            key, conv_states[0] = dynamic_conv.filtered(layer.k_sconv, key, conv_states[0], kd)
            value, conv_states[1] = dynamic_conv.filtered(layer.v_sconv, value, conv_states[1], vd)
        else:
            key, conv_states[0] = layer.k_sconv(key, key_hidden_states, conv_states[0], conv_mask)
            value, conv_states[1] = layer.v_sconv(value, value_hidden_states, conv_states[1], conv_mask)
    else:
        key, conv_states[0] = layer.k_sconv(key, conv_states[0], conv_mask)
        value, conv_states[1] = layer.v_sconv(value, conv_states[1], conv_mask)
    if layer.head_dim == 128 and relative.shape[-1] // layer.num_heads <= 128:
        query, key, value, rel = normalize(layer, query, key, value, relative)
    else:
        query = layer.q_l2_clip(query.reshape(1, 1, layer.num_heads, -1))
        key = layer.k_l2_clip(key.reshape(1, 1, layer.num_key_value_heads, -1))
        value = value.reshape(1, 1, layer.num_key_value_heads, -1)
        value = layers.unweighted_rms_norm(value.astype(mx.float32), layer.eps).astype(value.dtype)
        rel = layer.rel_l2_clip(relative.reshape(1, 1, layer.num_heads, -1))
    kv = cache[0] if cache is not None else None
    if full:
        offset = kv.offset if kv is not None else 0
        pos = layers._positions(offset, 1).astype(mx.float32)
        tau = 1.0 + layer.log_scaling_alpha * mx.log(mx.maximum((pos + 1) / layer.log_scaling_n_floor, 1.0))
        query = (query.astype(mx.float32) * tau[..., None, None]).astype(query.dtype)
        rel = (rel.astype(mx.float32) * tau[..., None, None]).astype(rel.dtype)
    q, k, v = (x.transpose(0, 2, 1, 3) for x in (query, key, value))
    if kv is not None:
        k, v = kv.update_and_fetch(k, v)
    if (kv is not None and layer.sliding_window and k.shape[2] >= 1024 and q.shape[-1] == 128
            and type(layer.rel_logits_proj) is layers.RelativeLogits and rel.dtype == q.dtype == mx.bfloat16):
        from . import grouped_attn
        return layer.o_proj(grouped_attn.attend(layer,q,k,v,rel,sinks,kv,gate))
    if kv is not None and type(layer.rel_logits_proj) is layers.RelativeLogits and rel.dtype == q.dtype:
        output = rel_mask.attend(layer, q, k, v, rel, sinks, kv)
    else:
        output = layer._attention(q, k, v, layer.rel_logits_proj(rel), sinks, kv)
    output = output.reshape(1, 1, -1).astype(mx.float32)
    return layer.o_proj((output * mx.sigmoid(gate.astype(mx.float32))).astype(gate.dtype))


def install():
    layers.SlidingAttention.__call__ = attend
    layers.FullAttention.__call__ = attend
