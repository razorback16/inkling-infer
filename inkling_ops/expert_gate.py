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
_gate_source = """
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

_gate_inputs = ["x", "ids", "packed", "scales", "amax", "gate_lut", "up_lut"]
_kernel = mx.fast.metal_kernel(
    name="inkling_stock_order_gate_up_fp4", input_names=_gate_inputs, output_names=["out"],
    header=_header, source=_gate_source)
_compact = mx.fast.metal_kernel(
    name="inkling_five_bit_scale_gate", input_names=[*_gate_inputs,"lower","upper"],
    output_names=["out"], header=_header, source=_gate_source.replace(
        "float term = partial * fp8_scale(scales[row * 144 + word / 2]);",
        """
        uint index = row * 144 + word / 2;
        uint low = (lower[index / 2] >> (4 * (index % 2))) & 15;
        uint high = (upper[index / 8] >> (index % 8)) & 1;
        uint code = low | (high << 4);
        uchar scale_bits = code == 31 ? scales[index] : uchar(96 + code);
        float term = partial * fp8_scale(scale_bits);
        """))
_pack_scales = mx.fast.metal_kernel(
    name="inkling_encode_expert_scales", input_names=["scales"], output_names=["lower","upper"],
    source="""
        uint g = thread_position_in_grid.x;
        uint low = 0, high = 0;
        for (uint j=0; j<8; ++j) {
            uint raw = scales[g*8+j];
            uint code = raw >= 96 && raw < 127 ? raw-96 : 31;
            low |= (code & 15) << (4*j);
            high |= (code >> 4) << j;
        }
        lower[g]=low;
        upper[g]=uchar(high);
    """)


def prepare(model):
    for layer in model.layers:
        proj = layer.mlp.experts.gate_up_proj
        if (getattr(proj,"mode",None) != "nvfp4" or proj.scales.dtype != mx.uint8
                or proj.scales.shape[1:] != (2304,144)):
            continue
        scales=proj.scales
        saved=getattr(proj,"_scale_planes",None)
        if saved is None or saved[0] is not scales:
            groups=scales.size//8
            low,high=_pack_scales(inputs=[scales],grid=(groups,1,1),threadgroup=(128,1,1),
                                  output_shapes=[(groups,)]*2,output_dtypes=[mx.uint32,mx.uint8])
            low=low.view(mx.uint8)
            mx.eval(low,high)
            object.__setattr__(proj,"_scale_planes",(scales,low,high))



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
    saved=getattr(proj,"_scale_planes",None)
    compact=saved is not None and saved[0] is proj.scales
    kernel=_compact if compact else _kernel
    inputs=[x, indices, proj.weight, proj.scales, proj.global_scale, *activation_tables()]
    if compact:
        inputs.extend(saved[1:])
    active = kernel(
        inputs=inputs,
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
