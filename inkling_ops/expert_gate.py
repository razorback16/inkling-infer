"""Decode interleaved NVFP4 projections and activate each routed expert."""
from functools import lru_cache

import mlx.core as mx
from mlx_lm.models.inkling2 import layers

_reference = layers.Experts.__call__
_header = """
        constant float levels[16] = {
            0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6
        };
        inline float fp8_scale(uchar bits) {
            ushort mag = bits & 127;
            ushort raw = (mag << 7) | (((mag + 1) >> 7) << 14) | ((bits & 128) << 8);
            return float(as_type<half>(raw)) * 256.f;
        }
    """
_source = """
        #pragma clang fp contract(off)
        uint lane = thread_index_in_simdgroup;
        uint slot = threadgroup_position_in_grid.y;
        uint expert = ids[slot];
        uint first = threadgroup_position_in_grid.x * 16 + simdgroup_index_in_threadgroup * 4;
        float sums[4] = {};
        for (uint word = lane; word < 288; word += 32) {
            float v[8];
            for (uint i = 0; i < 8; ++i) v[i] = float(x[word * 8 + i]);
            for (uint r = 0; r < 4; ++r) {
                uint row = expert * 2304 + first + r;
                uint bits = packed[row * 288 + word];
                float partial = 0.f;
                for (uint i = 0; i < 8; i += 4) {
                    partial += v[i] * levels[bits & 15] + v[i+1] * levels[(bits >> 4) & 15]
                             + v[i+2] * levels[(bits >> 8) & 15] + v[i+3] * levels[(bits >> 12) & 15];
                    bits >>= 16;
                }
                float term = partial * fp8_scale(scales[row * 144 + word / 2]);
                sums[r] += term;
            }
        }
        for (uint i = 0; i < 4; ++i) sums[i] = simd_sum(sums[i]);
        if (lane < 2) {
            float scale = amax[expert] / 2688.f;
            ushort g = as_type<ushort>(bfloat16_t(sums[lane*2]*scale));
            ushort u = as_type<ushort>(bfloat16_t(sums[lane*2+1]*scale));
            out[slot*1152+first/2+lane] = bfloat16_t(gate_lut[g]*up_lut[u]);
        }
    """


def make_kernel(interleaved):
    inputs = ["x", "ids", "packed", "amax", "gate_lut", "up_lut"]
    source = _source
    if interleaved:
        # A tile holds 128 FP4 words followed by sixteen packed scale words.
        source = source.replace("uint bits = packed[row * 288 + word];",
            "uint tile = ((row/4)*9 + word/32)*144; "
            "uint bits = packed[tile + r*32 + lane];").replace(
            "scales[row * 144 + word / 2]",
            "uchar(packed[tile+128+r*4+lane/8] >> (((lane/2)%4)*8))")
    else:
        inputs.insert(3, "scales")
    return mx.fast.metal_kernel(
        name="inkling_interleaved_gate" if interleaved else "inkling_stock_order_gate_up_fp4",
        input_names=inputs, output_names=["out"], header=_header, source=source)


_kernel = make_kernel(False)
_interleaved = make_kernel(True)


def prepare(model):
    for layer in model.layers:
        proj = layer.mlp.experts.gate_up_proj
        if (getattr(proj,"mode",None) != "nvfp4" or proj.group_size != 16 or proj.bits != 4
                or proj.weight.shape[1:] != (2304,288) or proj.scales.dtype != mx.uint8
                or proj.weight.dtype != mx.uint32 or proj.scales.shape != (proj.weight.shape[0],2304,144)):
            continue
        weight, scales = proj.weight, proj.scales
        saved = getattr(proj,"_weight_scale_tiles",None)
        if saved is not None and saved[0] is weight and saved[1] is scales:
            continue
        groups = weight.shape[0]*576
        words = weight.reshape(groups,4,9,32).transpose(0,2,1,3).reshape(groups,9,128)
        scale_words = scales.view(mx.uint32).reshape(groups,4,9,4).transpose(0,2,1,3).reshape(groups,9,16)
        tiles = mx.concatenate([words,scale_words],axis=-1)
        mx.eval(tiles)
        object.__setattr__(proj,"_weight_scale_tiles",(weight,scales,tiles))



@lru_cache(maxsize=1)
def activation_tables():
    values = mx.arange(65536, dtype=mx.uint32).astype(mx.uint16).view(mx.bfloat16).astype(mx.float32)
    gates = (mx.tanh(values / 4)*4) * mx.sigmoid(values)
    ups = mx.tanh(values / 25)*25
    mx.eval(gates, ups)
    return gates, ups


def gated_activation(layer, x, indices):
    proj = layer.gate_up_proj
    if (x.shape != (1, 1, 2304) or x.dtype != mx.bfloat16 or indices.shape != (1, 1, 8)
            or getattr(proj, "mode", None) != "nvfp4" or proj.group_size != 16
            or proj.bits != 4 or proj.weight.shape[1:] != (2304, 288) or "bias" in proj
            or proj.global_scale is None or proj.global_scale.shape != (proj.weight.shape[0],)
            or layer.act_fn.gate_softcap != 4 or layer.act_fn.up_softcap != 25):
        pair = proj(x[..., None, None, :], indices)
        return layer.act_fn(pair[..., 0::2], pair[..., 1::2])
    saved = getattr(proj,"_weight_scale_tiles",None)
    use_tiles = saved is not None and saved[0] is proj.weight and saved[1] is proj.scales
    kernel = _interleaved if use_tiles else _kernel
    data = [saved[2]] if use_tiles else [proj.weight, proj.scales]
    active = kernel(
        inputs=[x, indices, *data, proj.global_scale, *activation_tables()],
        grid=(144 * 128, 8, 1), threadgroup=(128, 1, 1),
        output_shapes=[(1, 1, 8, 1, 1152)], output_dtypes=[x.dtype],
    )[0]
    return active


def experts(layer, x, indices):
    if x.shape != (1, 1, 2304) or indices.shape != (1, 1, 8):
        return _reference(layer, x, indices)
    return layer.down_proj(gated_activation(layer, x, indices), indices).squeeze(-2)


def install():
    activation_tables()
    layers.Experts.__call__ = experts
