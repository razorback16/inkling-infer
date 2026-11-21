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
            for (uint r = 0; r < 4; ++r) {
                uint row = expert * 2304 + first + r;
                uint bits = packed[STRIPED ? ((row/2*18+word/16)*2+row%2)*16+word%16 : row*288+word];
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
            uint gi=((uint(g)&32767)<<2)|((g>>15)<<1);
            uint ui=((uint(u)&32767)<<2)|((u>>15)<<1);
            out[slot*1152+first/2+lane]=bfloat16_t(activation[gi]*activation[ui+1]);
        }
    """,
)


_mx_gate = mx.fast.metal_kernel(
    name="inkling_mxfp8_gate_pairs", input_names=["x","ids","w","scales","codes","activation"],
    output_names=["out"], source="""
        #pragma clang fp contract(off)
        uint lane = thread_index_in_simdgroup, slot = threadgroup_position_in_grid.y;
        uint first = thread_position_in_grid.x / 32 * 4;
        uint row = ids[slot] * 2304 + first;
        float dots[4] = {};
        for (uint block = lane; block < 288; block += 32) {
            uint4 inputs = ((device const uint4*)x)[block];
            float4 a = float4(as_type<bfloat4>(inputs.xy));
            float4 b = float4(as_type<bfloat4>(inputs.zw));
            for (uint r = 0; r < 4; ++r) {
                uint2 packed = ((device const uint2*)w)[(row+r)*288 + block];
                uchar4 lo = as_type<uchar4>(packed.x), hi = as_type<uchar4>(packed.y);
                float sum = 0;
                for (uint j = 0; j < 4; ++j) sum += a[j] * codes[lo[j]];
                for (uint j = 0; j < 4; ++j) sum += b[j] * codes[hi[j]];
                dots[r] += sum * codes[256 + uint(scales[(row+r)*72 + block/4])];
            }
        }
        for (uint r = 0; r < 4; ++r) dots[r] = simd_sum(dots[r]);
        if (lane < 2) {
            ushort g = as_type<ushort>(bfloat16_t(dots[lane*2]));
            ushort u = as_type<ushort>(bfloat16_t(dots[lane*2+1]));
            uint gi=((uint(g)&32767)<<2)|((g>>15)<<1);
            uint ui=((uint(u)&32767)<<2)|((u>>15)<<1);
            out[slot*1152+first/2+lane] = bfloat16_t(activation[gi]*activation[ui+1]);
        }
    """,
)


@lru_cache(maxsize=1)
def mx_codes():
    bits = mx.arange(256, dtype=mx.int32).astype(mx.uint8)
    unit = mx.full((256,8), 0x38383838, mx.uint32)
    scale = mx.dequantize(unit, bits[:,None], group_size=32, bits=8, mode="mxfp8", dtype=mx.float32)
    codes = mx.concatenate((mx.from_fp8(bits,mx.float32),scale[:,0]))
    mx.eval(codes)
    return codes


@lru_cache(maxsize=1)
def activation_tables():
    values = mx.arange(65536, dtype=mx.uint32).astype(mx.uint16).view(mx.bfloat16).astype(mx.float32)
    gates = (mx.tanh(values / 4)*4) * mx.sigmoid(values)
    ups = mx.tanh(values / 25)*25
    table=mx.contiguous(mx.stack((gates,ups),axis=1).reshape(2,32768,2).transpose(1,0,2))
    mx.eval(table)
    return table


def prepare_stripes(model):
    buffers = []
    for layer in model.layers:
        experts = layer.mlp.experts
        for proj, cols in ((experts.gate_up_proj, 288), (experts.down_proj, 144)):
            if (getattr(proj, "mode", None) != "nvfp4" or proj.weight.dtype != mx.uint32
                    or proj.weight.shape[1:] != (2304, cols)):
                continue
            raw = proj.weight
            saved = getattr(proj, "_striped_nvfp4", None)
            if saved is None or saved[0] is not raw:
                matrix = raw.reshape(raw.shape[0],1152,2,cols//16,16)
                striped = mx.contiguous(matrix.transpose(0,1,3,2,4))
                object.__setattr__(proj, "_striped_nvfp4", (raw,striped))
                buffers.append(striped)
    if buffers:
        mx.eval(buffers)


def decode_weights(proj):
    saved = getattr(proj, "_striped_nvfp4", None)
    if saved is not None and saved[0] is proj.weight:
        return saved[1], True
    return proj.weight, False


def gated_activation(layer, x, indices):
    proj = layer.gate_up_proj
    if (x.shape == (1,1,2304) and x.dtype == mx.bfloat16 and indices.shape == (1,1,8)
            and getattr(proj, "mode", None) == "mxfp8" and proj.group_size == 32 and proj.bits == 8
            and proj.weight.shape[1:] == (2304,576) and proj.scales.shape[1:] == (2304,72)
            and proj.weight.dtype == mx.uint32 and proj.scales.dtype == mx.uint8
            and proj.global_scale is None and "bias" not in proj and type(layer.act_fn) is layers.SoftcappedSwiGLU
            and layer.act_fn.gate_softcap == 4 and layer.act_fn.up_softcap == 25):
        return _mx_gate(inputs=[x,indices,proj.weight,proj.scales,mx_codes(),activation_tables()],
                        grid=(576*32,8,1),threadgroup=(64,1,1),
                        output_shapes=[(1,1,8,1,1152)],output_dtypes=[x.dtype])[0]
    if (x.shape != (1, 1, 2304) or x.dtype != mx.bfloat16 or indices.shape != (1, 1, 8)
            or getattr(proj, "mode", None) != "nvfp4" or proj.group_size != 16
            or proj.bits != 4 or proj.weight.shape[1:] != (2304, 288) or "bias" in proj
            or proj.global_scale is None or proj.global_scale.shape != (proj.weight.shape[0],)
            or layer.act_fn.gate_softcap != 4 or layer.act_fn.up_softcap != 25):
        pair = proj(x[..., None, None, :], indices)
        return layer.act_fn(pair[..., 0::2], pair[..., 1::2])
    weight, striped = decode_weights(proj)
    active = _kernel(
        inputs=[x, indices, weight, proj.scales, proj.global_scale, activation_tables()],
        template=[("STRIPED",striped)],
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
    mx_codes()
    layers.Experts.__call__ = experts
