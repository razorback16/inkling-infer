"""Evaluate paired key/value filters and their small coefficient projections."""
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.inkling2 import layers

_small = mx.fast.metal_kernel(
    name="inkling_small_attention_proj",input_names=["q","k","v","sw","kw","vw"],output_names=["s","kd","vd"],
    source="""
        uint row=threadgroup_position_in_grid.x*4;
        uint tid=thread_position_in_threadgroup.x,lane=thread_index_in_simdgroup,sg=simdgroup_index_in_threadgroup;
        auto x = row<H ? q : row<H+KOUT ? k : v;
        auto w = row<H ? sw : row<H+KOUT ? kw : vw;
        uint base = row<H ? row : row<H+KOUT ? row-H : row-H-KOUT;
        float d[4]={};
        for (uint col=tid*4; col<2304; col+=1024) {
            for (uint j=0;j<4;++j) {
                float value=float(x[col+j]);
                for (uint r=0;r<4;++r) d[r]=fma(value,float(w[(base+r)*2304+col+j]),d[r]);
            }
        }
        threadgroup float partial[8][4];
        for (uint r=0;r<4;++r) {
            float t=d[r];
            for (ushort b=16;b;b>>=1) t+=simd_shuffle_down(t,b);
            if (!lane) partial[sg][r]=t;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid<4) {
            float t=partial[0][tid];
            for (uint j=1;j<8;++j) t+=partial[j][tid];
            if (row<H) s[base+tid]=T(t);
            else if (row<H+KOUT) kd[base+tid]=T(t);
            else vd[base+tid]=T(t);
        }
    """,
)
_body = """
    uint c=thread_position_in_grid.x;
    bool key=thread_position_in_grid.y==0;
    auto w=key ? kw : vw;
    float x=key ? float(k[c]) : float(v[c]);
    float y;
    FILTER
    if (key) ko[c]=T(y);
    else vo[c]=T(y);
"""
_static = """
    float sum=0;
    for (uint j=0;j<3;++j) {
        float old=key ? kh[j*C+c] : vh[j*C+c];
        if (key) kbuf[j*C+c]=old; else vbuf[j*C+c]=old;
        sum=fma(float(W(old)),float(w[c*4+j]),sum);
    }
    if (key) kbuf[3*C+c]=x; else vbuf[3*C+c]=x;
    sum=fma(float(W(x)),float(w[c*4+3]),sum);
    y=(float(W(sum))+x)*0.4472135954999579f;
"""
_dynamic = """
    uint head=c/(C/H);
    volatile float sum=0;
    for (uint lag=0;lag<4;++lag) {
        uint idx=(3-lag)*C+c;
        float old=lag ? (key ? kh[idx] : vh[idx]) : x;
        if (key) kbuf[idx]=old; else vbuf[idx]=old;
        float dyn=key ? float(kd[head*4+lag]) : float(vd[head*4+lag]);
        volatile float prod=old*(float(w[lag*C+c])+dyn);
        sum=sum+prod;
    }
    y=(sum+x)/3.f;
"""


def _build(dynamic):
    names=["k","v","kh","vh","kw","vw"]
    return mx.fast.metal_kernel(
        name="inkling_paired_dynamic_filter" if dynamic else "inkling_paired_static_filter",
        input_names=names+["kd","vd"] if dynamic else names, output_names=["ko","vo","kbuf","vbuf"],
        source=_body.replace("FILTER",_dynamic if dynamic else _static))
_filters=(_build(False),_build(True))


def coefficients(layer,x,k,v):
    mods=(layer.sink_proj,layer.k_sconv.weight_proj,layer.v_sconv.weight_proj)
    if (x.shape!=(1,1,2304) or any(z.dtype!=mx.bfloat16 for z in (x,k,v))
            or any(type(m) is not nn.Linear or "bias" in m or m.weight.dtype!=x.dtype
                   or m.weight.shape[0]%4 or m.weight.shape[0]>144 for m in mods)):
        return None
    sizes=[m.weight.shape[0] for m in mods]
    return _small(inputs=[x,k,v,*[m.weight for m in mods]],template=[("T",x.dtype),("H",sizes[0]),("KOUT",sizes[1])],
                  grid=(sum(sizes)//4*256,1,1),threadgroup=(256,1,1),
                  output_shapes=[(1,1,n) for n in sizes],output_dtypes=[x.dtype]*3)


def apply(layer,k,v,history,coefs):
    full = coefs is not None
    kc,vc=layer.k_sconv,layer.v_sconv
    kw,vw=(kc.static_weight,vc.static_weight) if full else (kc.conv1d.weight,vc.conv1d.weight)
    extras=[coefs[1],coefs[2]] if full else []
    width=k.shape[-1]
    ko,vo,history[0],history[1]=_filters[full](
        inputs=[k,v,*history[:2],kw,vw,*extras],
        template=[("T",k.dtype),("W",kw.dtype),("C",width),("H",kc.num_heads if full else 1)],
        grid=(width,2,1),threadgroup=(128,1,1),
        output_shapes=[k.shape,v.shape,(1,4,width),(1,4,width)],output_dtypes=[k.dtype,v.dtype,mx.float32,mx.float32])
    return ko,vo
