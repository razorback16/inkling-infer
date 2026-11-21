"""Project and four-tap filter both distinct sliding key/value inputs in one dispatch."""
import mlx.core as mx
import mlx.nn as nn
from . import packed_rows
from mlx_lm.models.inkling2 import layers

_project = mx.fast.metal_kernel(
    name="inkling_project_filtered_kv",
    input_names=["kx","vx","kw","vw","kc","vc","kh","vh"],
    output_names=["key","value","next_k","next_v"],
    source="""
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
    """,
)


_rows = mx.fast.metal_kernel(
    name="inkling_filter_two_encoded_projections",
    input_names=["kx","vx","kw","vw","kc","vc","kh","vh","kp","vp","kb","vb","kd","vd"],
    output_names=["key","value","next_k","next_v"],
    source="""
        uint lane=thread_index_in_simdgroup, row=thread_position_in_grid.x/32;
        uint bank=threadgroup_position_in_grid.y;
        auto x=bank ? vx : kx;
        auto weight=bank ? vw : kw;
        auto payload=bank ? vp : kp;
        auto bases=bank ? vb : kb;
    """ + packed_rows.single_dot + """
        if (!lane) {
            auto conv=bank ? vc : kc;
            auto history=bank ? vh : kh;
            auto window=bank ? next_v : next_k;
            auto output=bank ? value : key;
            float current=float(bfloat16_t(dot));
            window[3*1024+row]=current;
            if (DYNAMIC) {
                auto coefs=bank ? vd : kd;
                volatile float sum=0;
                for (uint lag=0;lag<4;++lag) {
                    float val=current;
                    if (lag) {
                        uint i=(3-lag)*1024+row;
                        val=history[i]; window[i]=val;
                    }
                    float w=float(conv[lag*1024+row])+float(coefs[(row/128)*4+lag]);
                    volatile float product=val*w;
                    sum=sum+product;
                }
                output[row]=bfloat16_t((sum+current)/3.f);
            } else {
                float sum=0;
                for (uint t=0;t<3;++t) {
                    float val=history[t*1024+row]; window[t*1024+row]=val;
                    sum=fma(float(W(val)),float(conv[row*4+t]),sum);
                }
                sum=fma(float(W(current)),float(conv[row*4+3]),sum);
                output[row]=bfloat16_t((float(W(sum))+current)*0.4472135954999579f);
            }
        }
    """,
)


def prepare(model):
    arrays=[]
    for block in model.layers:
        layer=block.self_attn
        if not isinstance(layer,layers.FullAttention):
            continue
        for proj in (layer.k_proj,layer.v_proj):
            w=proj.weight
            if type(proj) is nn.Linear and w.shape==(1024,2304) and w.dtype==mx.bfloat16:
                saved=getattr(proj,"_kv_rows",None)
                if saved is None or saved[0] is not w:
                    encoded=packed_rows.encode_single(w)
                    object.__setattr__(proj,"_kv_rows",(w,encoded))
                    arrays.extend(encoded)
    if arrays:
        mx.eval(arrays)


def apply(layer,k,v,histories,mask):
    dynamic=isinstance(layer,layers.FullAttention)
    convtype=layers.DynamicShortConvolution if dynamic else layers.ShortConvolution
    if (mask is not None or k.shape != (1,1,2304) or v.shape != k.shape or k.dtype != v.dtype or k.dtype != mx.bfloat16
            or any(type(p) is not nn.Linear or "bias" in p or p.weight.shape != (1024,2304)
                   or p.weight.dtype != k.dtype for p in (layer.k_proj,layer.v_proj))
            or any(type(c) is not convtype or c.kernel_size != 4 or (dynamic and c.num_heads!=8)
                   for c in (layer.k_sconv,layer.v_sconv))
            or any(h is None or h.shape != (1,3,1024) for h in histories[:2])):
        return None
    kc,vc=((c.static_weight if dynamic else c.conv1d.weight) for c in (layer.k_sconv,layer.v_sconv))
    saved=[getattr(p,"_kv_rows",None) for p in (layer.k_proj,layer.v_proj)]
    if kc.dtype==vc.dtype and all(t is not None and t[0] is p.weight for t,p in zip(saved,(layer.k_proj,layer.v_proj))):
        (kp,kb),(vp,vb)=(t[1] for t in saved)
        kd,vd=(layer.k_sconv.weight_proj(k),layer.v_sconv.weight_proj(v)) if dynamic else (k,v)
        return _rows(inputs=[k,v,layer.k_proj.weight,layer.v_proj.weight,kc,vc,*histories[:2],kp,vp,kb,vb,kd,vd],
                     template=[("COLS",2304),("W",kc.dtype),("DYNAMIC",dynamic)],grid=(1024*32,2,1),
                     threadgroup=(128,1,1),output_shapes=[(1,1,1024)]*2+[(1,4,1024)]*2,
                     output_dtypes=[k.dtype]*2+[mx.float32]*2)
    if dynamic:
        return None
    if kc.dtype != vc.dtype:
        return None
    return _project(inputs=[k,v,layer.k_proj.weight,layer.v_proj.weight,kc,vc,*histories[:2]],
                    template=[("W",kc.dtype)],grid=(256*32,2,1),threadgroup=(128,1,1),
                    output_shapes=[(1,1,1024)]*2+[(1,4,1024)]*2,
                    output_dtypes=[k.dtype]*2+[mx.float32]*2)
