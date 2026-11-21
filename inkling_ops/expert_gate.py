"""Decode interleaved NVFP4 projections and activate each routed expert."""
from functools import lru_cache

import mlx.core as mx
from mlx_lm.models.inkling2 import layers

_reference = layers.Experts.__call__
_kernel = mx.fast.metal_kernel(
    name="inkling_stock_order_gate_up_fp4",
    input_names=["x", "ids", "packed", "scales", "amax", "activation"],
    output_names=["out"],
    header="""
        constant float levels[16] = {
            0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6
        };
        inline float fp8_scale(uchar bits) {
            ushort mag = bits & 127;
            ushort raw = (mag << 7) | (((mag + 1) >> 7) << 14) | ((bits & 128) << 8);
            return float(as_type<half>(raw)) * 256.f;
        }
    """,
    source="""
        #pragma clang fp contract(off)
        uint lane = thread_index_in_simdgroup;
        uint slot = threadgroup_position_in_grid.y;
        uint expert = ids[slot];
        uint first = threadgroup_position_in_grid.x * 16 + simdgroup_index_in_threadgroup * 4;
        float sums[4] = {};
        for (uint word = lane; word < 288; word += 32) {
            float v[8];
            for (uint i = 0; i < 8; ++i) v[i] = float(x[word * 8 + i]);
            uint group = (expert * 2304 + first) / 4;
            uint4 words;
            uchar4 factors;
            if (INTERLEAVED) {
                words = ((device const uint4*)packed)[group * 288 + word];
                factors = ((device const uchar4*)scales)[group * 144 + word / 2];
            }
            for (uint r = 0; r < 4; ++r) {
                uint row = expert * 2304 + first + r;
                uint bits = INTERLEAVED ? words[r] : packed[row * 288 + word];
                float partial = 0.f;
                for (uint i = 0; i < 8; i += 4) {
                    partial += v[i] * levels[bits & 15] + v[i+1] * levels[(bits >> 4) & 15]
                             + v[i+2] * levels[(bits >> 8) & 15] + v[i+3] * levels[(bits >> 12) & 15];
                    bits >>= 16;
                }
                float term = partial * fp8_scale(INTERLEAVED ? factors[r] : scales[row * 144 + word / 2]);
                sums[r] += term;
            }
        }
        for (uint i = 0; i < 4; ++i) sums[i] = simd_sum(sums[i]);
        if (lane < 2) {
            float scale = amax[expert] / 2688.f;
            ushort g = as_type<ushort>(bfloat16_t(sums[lane*2]*scale));
            ushort u = as_type<ushort>(bfloat16_t(sums[lane*2+1]*scale));
            uint gi=((uint(g)&32767)<<2)|((g>>15)<<1);
            uint ui=((uint(u)&32767)<<2)|((u>>15)<<1);
            out[slot*1152+first/2+lane]=bfloat16_t(activation[gi]*activation[ui+1]);
        }
    """,
)


@lru_cache(maxsize=1)
def activation_tables():
    values = mx.arange(65536, dtype=mx.uint32).astype(mx.uint16).view(mx.bfloat16).astype(mx.float32)
    gates = (mx.tanh(values / 4)*4) * mx.sigmoid(values)
    ups = mx.tanh(values / 25)*25
    table=mx.contiguous(mx.stack((gates,ups),axis=1).reshape(2,32768,2).transpose(1,0,2))
    mx.eval(table)
    return table


def prepare_weights(model):
    arrays = []
    for layer in model.layers:
        proj = layer.mlp.experts.gate_up_proj
        if (getattr(proj, "mode", None) != "nvfp4" or proj.weight.shape[1:] != (2304, 288)
                or proj.scales.shape[1:] != (2304, 144)):
            continue
        saved = getattr(proj, "_interleaved_gate", None)
        if saved is not None and saved[0] is proj.weight and saved[1] is proj.scales:
            continue
        packed = []
        for tensor in (proj.weight, proj.scales):
            experts, rows, width = tensor.shape
            tiles = tensor.reshape(experts, rows // 4, 4, width).transpose(0, 1, 3, 2)
            packed.append(mx.contiguous(tiles))
        object.__setattr__(proj, "_interleaved_gate", (proj.weight, proj.scales, packed))
        arrays.extend(packed)
    if arrays:
        mx.eval(arrays)


def gated_activation(layer, x, indices):
    proj = layer.gate_up_proj
    if (x.shape != (1, 1, 2304) or x.dtype != mx.bfloat16 or indices.shape != (1, 1, 8)
            or getattr(proj, "mode", None) != "nvfp4" or proj.group_size != 16
            or proj.bits != 4 or proj.weight.shape[1:] != (2304, 288) or "bias" in proj
            or proj.global_scale is None or proj.global_scale.shape != (proj.weight.shape[0],)
            or layer.act_fn.gate_softcap != 4 or layer.act_fn.up_softcap != 25):
        pair = proj(x[..., None, None, :], indices)
        return layer.act_fn(pair[..., 0::2], pair[..., 1::2])
    saved = getattr(proj, "_interleaved_gate", None)
    tiled = saved is not None and saved[0] is proj.weight and saved[1] is proj.scales
    w, scales = saved[2] if tiled else (proj.weight, proj.scales)
    active = _kernel(
        inputs=[x, indices, w, scales, proj.global_scale, activation_tables()],
        template=[("INTERLEAVED", tiled)],
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
