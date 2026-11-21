"""Apply both rank-64 projections and their message update in one prompt kernel."""
import mlx.core as mx

_pack=mx.fast.metal_kernel(
    name="inkling_prompt_message_fragments",input_names=["forget","gain"],output_names=["weights"],
    source="""
        uint at=thread_position_in_grid.x, lane=at%32, column_tile=at/256, k_tile=(at/32)%8;
        uint fm=((lane/4)&4)+(lane/2)%4,fn=((lane/4)&2)*2+(lane%2)*2;
        uint i=(column_tile*8+fn)*64+k_tile*8+fm;
        ((device float4*)weights)[at]=float4(forget[i],forget[i+64],gain[i],gain[i+64]);
    """,
)

_apply=mx.fast.metal_kernel(
    name="inkling_prompt_message_mma",input_names=["x","weights","hidden","msg"],output_names=["out"],
    ensure_row_contiguous=False,source="""
        #pragma clang fp contract(off)
        uint lane=thread_index_in_simdgroup, bank=thread_position_in_grid.x/32;
        uint fm=((lane/4)&4)+(lane/2)%4,fn=((lane/4)&2)*2+(lane%2)*2;
        uint row=threadgroup_position_in_grid.y*8+fm;
        uint valid=min(row,uint(L-1));
        simdgroup_float8x8 f[4],g[4];
        for (short j=0;j<4;++j) {
            f[j]=make_filled_simdgroup_matrix<float,8,8>(0.f);
            g[j]=make_filled_simdgroup_matrix<float,8,8>(0.f);
        }
        for (short k=0;k<8;++k) {
            simdgroup_float8x8 a;
            float2 v=float2(*(device const bfloat2*)(x+valid*x_strides[1]+k*8+fn));
            a.thread_elements()[0]=v.x; a.thread_elements()[1]=v.y;
            for (short j=0;j<4;++j) {
                float4 w=((device const float4*)weights)[((bank*4+j)*8+k)*32+lane];
                simdgroup_float8x8 b,h;
                b.thread_elements()[0]=w.x;b.thread_elements()[1]=w.y;
                h.thread_elements()[0]=w.z;h.thread_elements()[1]=w.w;
                simdgroup_multiply_accumulate(f[j],a,b,f[j]);
                simdgroup_multiply_accumulate(g[j],a,h,g[j]);
            }
        }
        if (row<L) {
            for (short j=0;j<4;++j) {
                uint col=(bank*4+j)*8+fn;
                for (ushort v=0;v<2;++v) {
                    float retained=(1.f-precise::tanh(f[j].thread_elements()[v]))*hidden[row*hidden_strides[1]+col+v];
                    float added=precise::tanh(g[j].thread_elements()[v])*msg[row*msg_strides[1]+col+v];
                    out[row*2304+col+v]=retained+added;
                }
            }
        }
    """,
)


def pack(forget,gain):
    return _pack(inputs=[forget,gain],grid=(2304*64//2,1,1),threadgroup=(128,1,1),
                 output_shapes=[(2304//8,8,32,4)],output_dtypes=[mx.float32])[0]


def apply(x,weights,hidden,message):
    l=x.shape[1]
    return _apply(inputs=[x,weights,hidden,message],template=[("L",l)],grid=(2304,(l+7)//8,1),
                  threadgroup=(128,1,1),output_shapes=[hidden.shape],output_dtypes=[mx.float32])[0]
