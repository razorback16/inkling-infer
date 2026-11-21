"""Project and four-tap filter both distinct sliding key/value inputs in one dispatch."""
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.inkling2 import layers

_source = """
        uint lane=thread_index_in_simdgroup, bank=threadgroup_position_in_grid.y;
        uint first=(thread_position_in_grid.x/32)*4;
        auto x=bank ? vx : kx; auto w=bank ? vw : kw;
        float4 acc=0;
        for (uint d=lane*4;d<2304;d+=128) {
            float4 a=float4(*(device const bfloat4*)(x+d));
            for (uint r=0;r<4;++r) {
                float4 b=float4(*(device const bfloat4*)(w+(first+r)*2304+d));
                for (uint j=0;j<4;++j) acc[r]=fma(a[j],b[j],acc[r]);
            }
        }
        auto conv=bank ? vc : kc; auto hist=bank ? vh : kh;
        auto output=bank ? value : key; auto window=bank ? next_v : next_k;
        for (uint r=0;r<4;++r) {
            float dot=acc[r];
            for (ushort i=16;i;i>>=1) dot+=simd_shuffle_down(dot,i);
            if (!lane) {
                uint c=first+r;
                float v=float(bfloat16_t(dot)), total=0;
                for (uint j=0;j<3;++j) {
                    float old=hist[j*1024+c];
                    window[j*1024+c]=old;
                    total=fma(float(W(old)),float(conv[c*4+j]),total);
                }
                window[3*1024+c]=v;
                total=fma(float(W(v)),float(conv[c*4+3]),total);
                output[c]=bfloat16_t((float(W(total))+v)*0.4472135954999579f);
            }
        }
    """
_inputs=["kx","vx","kw","vw","kc","vc","kh","vh"]
_outputs=["key","value","next_k","next_v"]
_project=mx.fast.metal_kernel(name="inkling_project_filtered_kv",input_names=_inputs,
                              output_names=_outputs,source=_source)
_tiled_source=_source.replace("bank=threadgroup_position_in_grid.y","bank=simdgroup_index_in_threadgroup/2")
_tiled_source=_tiled_source.replace("uint first=(thread_position_in_grid.x/32)*4;",
    "uint group=threadgroup_position_in_grid.x, pair=simdgroup_index_in_threadgroup%2; "
    "uint first=group*8+pair*4;")
_tiled_source=_tiled_source.replace("auto w=bank ? vw : kw;","")
_tiled_source=_tiled_source.replace("w+(first+r)*2304+d",
    "tiles+(((((group*18+d/128)*2+bank)*2+pair)*32+lane)*4+r)*4")
_tiled=mx.fast.metal_kernel(name="inkling_tiled_filtered_kv",input_names=["kx","vx","tiles",*_inputs[4:]],
                            output_names=_outputs,source=_tiled_source)


def prepare(model):
    for i in model.args.loop_core:
        layer=model.layers[i].self_attn
        if not isinstance(layer,layers.SlidingAttention):
            continue
        kp,vp=layer.k_proj,layer.v_proj
        if any(type(p) is not nn.Linear or "bias" in p or p.weight.dtype!=mx.bfloat16
               or p.weight.shape!=(1024,2304) for p in (kp,vp)):
            continue
        kw,vw=kp.weight,vp.weight
        saved=getattr(layer,"_paired_kv_tiles",None)
        if saved is not None and saved[0] is kw and saved[1] is vw:
            continue
        blocks=mx.stack((kw,vw)).reshape(2,128,2,4,18,32,4)
        tiles=mx.contiguous(blocks.transpose(1,4,0,2,5,3,6))
        mx.eval(tiles)
        object.__setattr__(layer,"_paired_kv_tiles",(kw,vw,tiles))





def apply(layer,k,v,histories,mask):
    if (mask is not None or k.shape != (1,1,2304) or v.shape != k.shape or k.dtype != v.dtype or k.dtype != mx.bfloat16
            or any(type(p) is not nn.Linear or "bias" in p or p.weight.shape != (1024,2304)
                   or p.weight.dtype != k.dtype for p in (layer.k_proj,layer.v_proj))
            or any(type(c) is not layers.ShortConvolution or c.kernel_size != 4
                   for c in (layer.k_sconv,layer.v_sconv))
            or any(h is None or h.shape != (1,3,1024) for h in histories[:2])):
        return None
    kc,vc=layer.k_sconv.conv1d.weight,layer.v_sconv.conv1d.weight
    if kc.dtype != vc.dtype:
        return None
    kw,vw=layer.k_proj.weight,layer.v_proj.weight
    saved=getattr(layer,"_paired_kv_tiles",None)
    use_tiles=saved is not None and saved[0] is kw and saved[1] is vw
    kernel=_tiled if use_tiles else _project
    weights=[saved[2]] if use_tiles else [kw,vw]
    return kernel(inputs=[k,v,*weights,kc,vc,*histories[:2]],
                    template=[("W",kc.dtype)],grid=(128*128,1,1) if use_tiles else (256*32,2,1),threadgroup=(128,1,1),
                    output_shapes=[(1,1,1024)]*2+[(1,4,1024)]*2,
                    output_dtypes=[k.dtype]*2+[mx.float32]*2)
