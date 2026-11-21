"""Project normalized hidden/message pairs to the rank-64 bottleneck directly."""
import mlx.core as mx

_kernel = mx.fast.metal_kernel(
    name="inkling_message_reduce",input_names=["state","msg","weight"],output_names=["out"],
    source="""
        uint tid = thread_position_in_threadgroup.x, lane = thread_index_in_simdgroup;
        uint sg = simdgroup_index_in_threadgroup, bank = threadgroup_position_in_grid.y;
        uint row = threadgroup_position_in_grid.x*4;
        threadgroup float sums[2][18], partial[8][4];
        threadgroup bfloat16_t joined[4608];
        float sx=0, sm=0;
        for (uint j=0;j<4;++j) {
            uint d=bank*2304+tid*4+j;
            float x=state[d], m=msg[d];
            sx+=x*x; sm+=m*m;
        }
        sx=simd_sum(sx); sm=simd_sum(sm);
        if (!lane) { sums[0][sg]=sx; sums[1][sg]=sm; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sx=precise::rsqrt(simd_sum(lane<18?sums[0][lane]:0.f)/2304.f+1.e-6f);
        sm=precise::rsqrt(simd_sum(lane<18?sums[1][lane]:0.f)/2304.f+1.e-6f);
        for (uint j=0;j<4;++j) {
            uint d=tid*4+j;
            joined[d]=bfloat16_t(state[bank*2304+d]*sx);
            joined[2304+d]=bfloat16_t(msg[bank*2304+d]*sm);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid<256) {
            float acc[4]={};
            for (uint i=tid*4; i<4608; i+=1024) {
                for (uint j=0;j<4;++j) {
                    float v=float(joined[i+j]);
                    for (uint r=0;r<4;++r)
                        acc[r]=fma(v,float(weight[(bank*64+row+r)*4608+i+j]),acc[r]);
                }
            }
            for (uint r=0;r<4;++r) {
                float d=acc[r];
                for (ushort i=16;i;i>>=1) d+=simd_shuffle_down(d,i);
                if (!lane) partial[sg][r]=d;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid<4) {
            float d=partial[0][tid];
            for (uint i=1;i<8;++i) d+=partial[i][tid];
            out[bank*64+row+tid]=bfloat16_t(d);
        }
    """,
)


def project(states,msg,down):
    banks=states.size//2304
    return _kernel(inputs=[states,msg,down],grid=(16*576,banks,1),threadgroup=(576,1,1),
                   output_shapes=[states.shape[:-1]+(64,)],output_dtypes=[mx.bfloat16])[0]
