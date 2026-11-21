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
        inline float activation_value(device const float* table,ushort code,uint field) {
            uint mag=uint(code)&32767, which=(uint(code)>>15)*2+field;
            if (mag-103u*128u<33u*128u) return table[(mag-103u*128u)*4+which];
            uint e=mag>>7, page=32, adjustment=0;
            if (e<6) page=33+uint(max(int(e)-2,0));
            else if (e<103) { page=0; adjustment=(e-103u)<<23; }
            else if (e==255) page=37;
            uint bits=as_type<uint>(table[(page*128+(mag&127))*4+which]);
            return as_type<float>(bits+adjustment);
        }
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
            if (COMPACT) {
                out[slot*1152+first/2+lane]=bfloat16_t(activation_value(activation,g,0)*activation_value(activation,u,1));
            } else {
            uint gi=((uint(g)&32767)<<2)|((g>>15)<<1);
            uint ui=((uint(u)&32767)<<2)|((u>>15)<<1);
            out[slot*1152+first/2+lane]=bfloat16_t(activation[gi]*activation[ui+1]);
            }
        }
    """,
)


@lru_cache(maxsize=1)
def activation_tables():
    values = mx.arange(65536, dtype=mx.uint32).astype(mx.uint16).view(mx.bfloat16).astype(mx.float32)
    gates = (mx.tanh(values / 4)*4) * mx.sigmoid(values)
    ups = mx.tanh(values / 25)*25
    table=mx.contiguous(mx.stack((gates,ups),axis=1).reshape(2,32768,2).transpose(1,0,2))
    pages=table.reshape(256,128,4)
    short=mx.concatenate((pages[103:136],pages[mx.array([0,3,4,5,255])]))
    ex=mx.arange(256,dtype=mx.int32)
    index=mx.where(ex<6,33+mx.maximum(ex-2,0),mx.where(ex==255,37,mx.clip(ex-103,0,32)))
    delta=mx.where((ex>=6)&(ex<103),ex-103,0).astype(mx.uint32)<<23
    restored=short.view(mx.uint32)[index]+delta.reshape(256,1,1)
    if mx.array_equal(restored,pages.view(mx.uint32)).item():
        mx.eval(short)
        return short,True
    return table,False


def gated_activation(layer, x, indices):
    proj = layer.gate_up_proj
    if (x.shape != (1, 1, 2304) or x.dtype != mx.bfloat16 or indices.shape != (1, 1, 8)
            or getattr(proj, "mode", None) != "nvfp4" or proj.group_size != 16
            or proj.bits != 4 or proj.weight.shape[1:] != (2304, 288) or "bias" in proj
            or proj.global_scale is None or proj.global_scale.shape != (proj.weight.shape[0],)
            or layer.act_fn.gate_softcap != 4 or layer.act_fn.up_softcap != 25):
        pair = proj(x[..., None, None, :], indices)
        return layer.act_fn(pair[..., 0::2], pair[..., 1::2])
    table,compact=activation_tables()
    active = _kernel(
        inputs=[x, indices, proj.weight, proj.scales, proj.global_scale, table], template=[("COMPACT",compact)],
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
