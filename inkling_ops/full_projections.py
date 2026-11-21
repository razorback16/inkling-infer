"""Joint content, filter and sink projections for single-token global attention."""
import mlx.core as mx
import mlx.nn as nn

_kernel=mx.fast.metal_kernel(
    name="inkling_global_five_projections",input_names=["q","k","v","sw","kw","vw","kd","vd"],
    output_names=["ko","vo","kc","vc","sinks"],source="""
        uint group=threadgroup_position_in_grid.x,tid=thread_position_in_threadgroup.x;
        uint sg=simdgroup_index_in_threadgroup,lane=thread_index_in_simdgroup;
        threadgroup float partial[8][4];
        if (group<256) {
            uint bank=group/128,row=(group%128)*8+sg;
            auto x=bank ? v : k;
            auto w=bank ? vw : kw;
            float dot=0;
            for (uint d=lane*4;d<2304;d+=128) {
                float4 a=float4(*(device const bfloat4*)(x+d));
                float4 b=float4(*(device const bfloat4*)(w+row*2304+d));
                for (uint j=0;j<4;++j) dot=fma(a[j],b[j],dot);
            }
            for (ushort step=16;step;step>>=1) dot+=simd_shuffle_down(dot,step);
            if (!lane) {
                if (bank) vo[row]=bfloat16_t(dot); else ko[row]=bfloat16_t(dot);
            }
        } else {
            uint g=group-256,base=(g%8)*4;
            auto x=g<8 ? k : g<16 ? v : q;
            auto w=g<8 ? kd : g<16 ? vd : sw;
            float dots[4]={};
            for (uint d=tid*4;d<2304;d+=1024) {
                for (uint j=0;j<4;++j)
                    for (uint r=0;r<4;++r) dots[r]=fma(float(x[d+j]),float(w[(base+r)*2304+d+j]),dots[r]);
            }
            for (uint r=0;r<4;++r) {
                float dot=dots[r];
                for (ushort step=16;step;step>>=1) dot+=simd_shuffle_down(dot,step);
                if (!lane) partial[sg][r]=dot;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid<4) {
                float dot=partial[0][tid];
                for (uint i=1;i<8;++i) dot+=partial[i][tid];
                if (g<8) kc[base+tid]=bfloat16_t(dot);
                else if (g<16) vc[base+tid]=bfloat16_t(dot);
                else sinks[base+tid]=bfloat16_t(dot);
            }
        }
    """)


def apply(layer,q,k,v):
    if any(x.shape!=(1,1,2304) or x.dtype!=mx.bfloat16 for x in (q,k,v)):
        return None
    mods=(layer.sink_proj,layer.k_proj,layer.v_proj,layer.k_sconv.weight_proj,layer.v_sconv.weight_proj)
    widths=(24,1024,1024,32,32)
    if any(type(m) is not nn.Linear or "bias" in m or m.weight.shape!=(rows,2304)
           or m.weight.dtype!=mx.bfloat16 for m,rows in zip(mods,widths)):
        return None
    return _kernel(inputs=[q,k,v,*[m.weight for m in mods]],
                   grid=(278*256,1,1),threadgroup=(256,1,1),
                   output_shapes=[(1,1,n) for n in (1024,1024,32,32,24)],
                   output_dtypes=[mx.bfloat16]*5)
