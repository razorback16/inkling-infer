"""Pack attention projections with a common one-token input."""
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.inkling2 import layers

from .head_norm import normalize
from . import rel_mask, filtered_kv, packed_rows

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


def prepare_query_weights(model):
    arrays = []
    for index, block in enumerate(model.layers):
        layer = block.self_attn
        common = (layer.q_proj,layer.r_proj,layer.gate_proj)
        variants = []
        if isinstance(layer,layers.SlidingAttention):
            variants.append(("_all_qkv",common+(layer.k_proj,layer.v_proj)))
        if isinstance(layer,layers.FullAttention) or index in model.args.loop_core:
            variants.append(("_common_q",common))
        for name, modules in variants:
            if any(type(p) is not nn.Linear or "bias" in p or p.weight.dtype!=mx.bfloat16
                   or p.weight.shape[1]!=2304 for p in modules):
                continue
            weight,_ = projection(layer,modules,name)
            saved = getattr(layer,name+"_tiles",None)
            if weight.shape[0]%16 or (saved is not None and saved[0] is weight):
                continue
            encoded = packed_rows.encode(weight)
            object.__setattr__(layer,name+"_tiles",(weight,encoded))
            arrays.extend(encoded)
    if arrays:
        mx.eval(arrays)


def attend(layer, hidden_states, key_hidden_states, value_hidden_states, *, cache, conv_states, conv_mask):
    full = isinstance(layer, layers.FullAttention)
    same = hidden_states is key_hidden_states and hidden_states is value_hidden_states
    common = (layer.q_proj, layer.r_proj, layer.gate_proj)
    modules = common + (layer.k_proj, layer.v_proj) if same else common
    if hidden_states.shape[:2] != (1, 1) or any(type(m) is not nn.Linear or "bias" in m for m in modules):
        reference = _full if full else _sliding
        return reference(layer, hidden_states, key_hidden_states, value_hidden_states,
                          cache=cache, conv_states=conv_states, conv_mask=conv_mask)
    name = "_all_qkv" if same else "_common_q"
    packed, boundaries = projection(layer,modules,name)
    tiles = getattr(layer,name+"_tiles",None)
    compact = tiles is not None and tiles[0] is packed and hidden_states.dtype==mx.bfloat16
    sink=layer.sink_proj
    if (compact and type(sink) is nn.Linear and "bias" not in sink and sink.weight.dtype==mx.bfloat16
            and sink.weight.shape==(layer.num_heads,2304) and layer.num_heads%4==0):
        projected,sinks=packed_rows.project_with_sinks(hidden_states,packed,tiles[1],sink.weight)
    else:
        projected=packed_rows.project(hidden_states,packed,tiles[1]) if compact else hidden_states @ packed.T
        sinks=sink(hidden_states)
    parts = mx.split(projected,boundaries,axis=-1)
    query, relative, gate = parts[:3]
    filtered = (None if full or same else
                filtered_kv.apply(layer,key_hidden_states,value_hidden_states,conv_states,conv_mask))
    if filtered is not None:
        key,value,conv_states[0],conv_states[1] = filtered
    elif same:
        key, value = parts[3:]
    else:
        key = layer.k_proj(key_hidden_states)
        value = layer.v_proj(value_hidden_states)
    fused_norm = (same and not full and conv_mask is None and conv_states is not None and layer.head_dim == 128
                  and relative.shape[-1] // layer.num_heads <= 128
                  and key.dtype == value.dtype == query.dtype
                  and all(type(c) is layers.ShortConvolution and c.kernel_size == 4
                          for c in (layer.k_sconv, layer.v_sconv))
                  and layer.k_sconv.conv1d.weight.dtype == layer.v_sconv.conv1d.weight.dtype
                  and all(h is not None and h.shape == (1,3,key.shape[-1]) for h in conv_states[:2]))
    if full:
        key, conv_states[0] = layer.k_sconv(key, key_hidden_states, conv_states[0], conv_mask)
        value, conv_states[1] = layer.v_sconv(value, value_hidden_states, conv_states[1], conv_mask)
    elif filtered is None and not fused_norm:
        key, conv_states[0] = layer.k_sconv(key, conv_states[0], conv_mask)
        value, conv_states[1] = layer.v_sconv(value, conv_states[1], conv_mask)
    pair = None
    if fused_norm:
        filters = [*conv_states[:2],layer.k_sconv.conv1d.weight,layer.v_sconv.conv1d.weight]
        query, pair, rel, conv_states[:2] = normalize(layer,query,key,value,relative,filters=filters)
        key,value = mx.split(pair,2,axis=2)
    elif layer.head_dim == 128 and relative.shape[-1] // layer.num_heads <= 128:
        query, key, value, rel, pair = normalize(layer, query, key, value, relative, return_pair=True)
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
    if pair is not None and getattr(kv, "packed", None) is not None and hasattr(kv, "update_packed"):
        kv.defer_update = (getattr(kv,"grouped",False) and layer.sliding_window==1024
                           and q.shape[-1]==128 and type(layer.rel_logits_proj) is layers.RelativeLogits
                           and rel.dtype==q.dtype==mx.bfloat16)
        k, v = kv.update_packed(pair.transpose(0,2,1,3))
    elif kv is not None:
        k, v = kv.update_and_fetch(k, v)
    if (kv is not None and 1024 <= k.shape[2] <= 8192 and q.shape[-1] == 128
            and type(layer.rel_logits_proj) is layers.RelativeLogits and rel.dtype == q.dtype == mx.bfloat16):
        from . import grouped_attn
        features=grouped_attn.attend(layer,q,k,v,rel,sinks,kv,gate)
        from . import filtered_output
        out=filtered_output.apply(layer.o_proj,features,getattr(kv,"output_filter",None))
        return layer.o_proj(features) if out is None else out
    if kv is not None and type(layer.rel_logits_proj) is layers.RelativeLogits and rel.dtype == q.dtype:
        output = rel_mask.attend(layer, q, k, v, rel, sinks, kv)
    else:
        output = layer._attention(q, k, v, layer.rel_logits_proj(rel), sinks, kv)
    output = output.reshape(1, 1, -1).astype(mx.float32)
    return layer.o_proj((output * mx.sigmoid(gate.astype(mx.float32))).astype(gate.dtype))


def install():
    layers.SlidingAttention.__call__ = attend
    layers.FullAttention.__call__ = attend
