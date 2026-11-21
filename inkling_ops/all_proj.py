"""Batch the large and small attention projections using their native reductions."""
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.inkling2 import layers

_kernel = mx.fast.metal_kernel(
    name="inkling_mixed_width_projections",input_names=["xq","xk","xv","w"],output_names=["out"],
    source="""
        uint tg=threadgroup_position_in_grid.x, sg=simdgroup_index_in_threadgroup;
        uint lane=thread_index_in_simdgroup, tid=thread_position_in_threadgroup.x;
        bool small = tg >= LARGE/32;
        uint row = small ? LARGE+(tg-LARGE/32)*4 : tg*32+sg*4;
        auto x = small ? (row<SK ? xq : row<SV ? xk : xv) : (row<BK ? xq : row<BV ? xk : xv);
        uint col = (small ? tid : lane)*4, stride = small ? 1024 : 128;
        float dots[4]={};
        for (uint i=col; i<2304; i+=stride) {
            for (uint j=0;j<4;++j) {
                float a=float(x[i+j]);
                for (uint r=0;r<4;++r) dots[r]=fma(a,float(w[(row+r)*2304+i+j]),dots[r]);
            }
        }
        threadgroup float partial[8][4];
        for (uint r=0;r<4;++r) {
            float d=dots[r];
            for (ushort s=16;s;s>>=1) d+=simd_shuffle_down(d,s);
            if (!lane) {
                if (small) partial[sg][r]=d;
                else out[row+r]=bfloat16_t(d);
            }
        }
        if (small) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid<4) {
                float d=partial[0][tid];
                for (uint i=1;i<8;++i) d+=partial[i][tid];
                out[row+tid]=bfloat16_t(d);
            }
        }
    """,
)


def project(layer, x, k, v, with_filters):
    mods = (layer.q_proj,layer.r_proj,layer.gate_proj,layer.k_proj,layer.v_proj,layer.sink_proj)
    if with_filters:
        mods += (layer.k_sconv.weight_proj, layer.v_sconv.weight_proj)
    if (any(t.dtype!=mx.bfloat16 or t.shape!=(1,1,2304) for t in (x,k,v)) or
            any(type(m) is not nn.Linear or "bias" in m or m.weight.dtype!=mx.bfloat16 for m in mods)):
        return None
    weights = tuple(m.weight for m in mods)
    name = "_mixed_filter_weights" if with_filters else "_mixed_attention_weights"
    saved = getattr(layer,name,None)
    if saved is None or not all(a is b for a,b in zip(weights,saved[0])):
        lengths=[w.shape[0] for w in weights]
        if (any(s%4 for s in lengths) or sum(lengths[:5])%32
                or any(s<=144 for s in lengths[:5]) or any(s>144 for s in lengths[5:])):
            return None
        ends=[]
        n=0
        for length in lengths:
            n+=length
            ends.append(n)
        saved=(weights,mx.concatenate(weights),ends)
        object.__setattr__(layer,name,saved)
    _,packed,ends=saved
    large=ends[4]; n=ends[-1]
    groups=large//32+(n-large)//4
    out=_kernel(inputs=[x,k,v,packed],
                template=[("LARGE",large),("BK",ends[2]),("BV",ends[3]),("SK",ends[5]),
                          ("SV",ends[6] if with_filters else n)],
                grid=(groups*256,1,1),threadgroup=(256,1,1),
                output_shapes=[(1,1,n)],output_dtypes=[x.dtype])[0]
    return mx.split(out,ends[:-1],axis=-1)
