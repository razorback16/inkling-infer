"""Apply both rank-64 projections and their message update in one prompt kernel."""
import mlx.core as mx

_paired_rms=mx.fast.metal_kernel(name="inkling_prompt_pair_rms",input_names=["x","m"],output_names=["out"],ensure_row_contiguous=False,source="""
   uint group=threadgroup_position_in_grid.x, row=group>>1, side=group&1;
   uint tid=thread_position_in_threadgroup.x, lane=thread_index_in_simdgroup, sg=simdgroup_index_in_threadgroup;
   auto input = side ? m : x;
   uint base = row * (side ? m_strides[1] : x_strides[1]);
   uint stride = side ? m_strides[2] : x_strides[2];
   float4 v;
   for (uint i=0; i<4; ++i) v[i]=input[base+(4*tid+i)*stride];
   float sum=0;
   for (short i=0;i<4;++i) sum+=v[i]*v[i];
   threadgroup float sums[18];
   sum=simd_sum(sum);
   if (lane==0) sums[sg]=sum;
   threadgroup_barrier(mem_flags::mem_threadgroup);
   float scale=precise::rsqrt(simd_sum(lane<18 ? sums[lane] : 0.f)/2304.f+1e-6f);
   ((device bfloat4*)out)[group*576+tid]=bfloat4(v*scale);
""")
def normalized_inputs(x,m):
    return _paired_rms(inputs=[x,m],grid=(2*x.shape[1]*576,1,1),threadgroup=(576,1,1),output_shapes=[x.shape[:-1]+(4608,)],output_dtypes=[mx.bfloat16])[0]

_apply=mx.fast.metal_kernel(
    name="inkling_prompt_message_mma",input_names=["x","forget","gain","hidden","msg"],output_names=["out"],
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
                uint channel=(bank*4+j)*8+fn, index=k*8+fm;
                uint f_addr=channel*forget_strides[0]+index*forget_strides[1];
                uint g_addr=channel*gain_strides[0]+index*gain_strides[1];
                simdgroup_float8x8 b,h;
                b.thread_elements()[0]=forget[f_addr];b.thread_elements()[1]=forget[f_addr+forget_strides[0]];
                h.thread_elements()[0]=gain[g_addr];h.thread_elements()[1]=gain[g_addr+gain_strides[0]];
                simdgroup_multiply_accumulate(f[j],a,b,f[j]);
                simdgroup_multiply_accumulate(g[j],a,h,g[j]);
            }
        }
        if (row<L) {
            for (short j=0;j<4;++j) {
                uint col=(bank*4+j)*8+fn;
                for (ushort v=0;v<2;++v) {
                    float retained=(1.f-precise::tanh(f[j].thread_elements()[v]))*hidden[row*hidden_strides[1]+(col+v)*hidden_strides[2]];
                    float added=precise::tanh(g[j].thread_elements()[v])*msg[row*msg_strides[1]+(col+v)*msg_strides[2]];
                    out[row*2304+col+v]=retained+added;
                }
            }
        }
    """,
)


def apply(x,forget,gain,hidden,message):
    l=x.shape[1]
    return _apply(inputs=[x,forget,gain,hidden,message],template=[("L",l)],grid=(2304,(l+7)//8,1),
                  threadgroup=(128,1,1),output_shapes=[hidden.shape],output_dtypes=[mx.float32])[0]
