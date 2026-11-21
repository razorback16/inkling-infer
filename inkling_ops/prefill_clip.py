"""Warp-local BF16 head clipping for long prompt tensors."""
import mlx.core as mx
from mlx_lm.kernels.common import scalar
from mlx_lm.models.inkling2 import layers

_scaled = layers.ScaledL2Clip.__call__
_unscaled = layers.L2Clip.__call__
_kernel = mx.fast.metal_kernel(
    name="inkling_prompt_clip",input_names=["x","scale","threshold"],output_names=["out"],
    source="""
        #pragma clang fp contract(off)
        uint lane=thread_index_in_simdgroup, row=thread_position_in_grid.x/32;
        uint base=row*D, head=row%H;
        float values[4], squares=0;
        if (D==16) {
            for (uint j=0;j<16;++j) {
                float v=float(x[base+j])*float(scale[head*16+j]);
                squares+=v*v;
            }
            if (lane<16) {
                float v=float(x[base+lane])*float(scale[head*16+lane]);
                out[base+lane]=bfloat16_t(v*min(1.f,threshold/(precise::sqrt(squares)+1e-6f)));
            }
        } else {
            for (uint j=0;j<4;++j) {
                uint col=lane*4+j;
                float v=float(x[base+col]);
                if (SCALED) v*=float(scale[head*D+col]);
                values[j]=v;
                squares+=v*v;
            }
            squares=simd_sum(squares);
            float a=min(1.f,threshold/(precise::sqrt(squares)+1e-6f));
            for (uint j=0;j<4;++j) out[base+lane*4+j]=bfloat16_t(values[j]*a);
        }
    """,
)


def clip(layer,x):
    scaled=type(layer) is layers.ScaledL2Clip
    reference=_scaled if scaled else _unscaled
    if (x.ndim!=4 or x.shape[1]<8 or x.dtype!=mx.bfloat16 or x.shape[-1] not in (16,128)
            or (not scaled and x.shape[-1]==16)):
        return reference(layer,x)
    d=x.shape[-1]
    h=x.shape[-2]
    if scaled and layer.scale.size!=h*d:
        return reference(layer,x)
    return _kernel(inputs=[x,layer.scale if scaled else x,scalar(layer.threshold)],
                   template=[("SCALED",scaled),("H",h),("D",d)],grid=(x.size//d*32,1,1),
                   threadgroup=(128,1,1),output_shapes=[x.shape],output_dtypes=[x.dtype])[0]
