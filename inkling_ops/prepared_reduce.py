"""Compute each message bottleneck directly from its state and prepared message."""
import mlx.core as mx

_kernel=mx.fast.metal_kernel(
    name="inkling_prepared_message_bottleneck",
    input_names=["state","message","weight","inverse"],output_names=["out"],
    source="""
        uint tid=thread_position_in_threadgroup.x,lane=thread_index_in_simdgroup;
        uint sg=simdgroup_index_in_threadgroup,bank=threadgroup_position_in_grid.y;
        uint row=threadgroup_position_in_grid.x*ROWS;
        uint state_base=SHARED ? 0 : bank*2304;
        threadgroup float norm[18],parts[8][ROWS];
        float inv;
        if (PRECOMPUTED) {
            inv=inverse[SHARED ? 0 : bank];
        } else {
            for (uint g=sg;g<18;g+=8) {
                float norm2=0;
                for (uint i=0;i<4;++i) {
                    float x=state[state_base+(g*32+lane)*4+i]; norm2+=x*x;
                }
                norm2=simd_sum(norm2);
                if (!lane) norm[g]=norm2;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            inv=precise::rsqrt(simd_sum(lane<18 ? norm[lane] : 0.f)/2304.f+1e-6f);
        }
        float dots[ROWS]={};
        for (uint d=tid*4;d<4608;d+=1024) {
            for (uint j=0;j<4;++j) {
                float raw=d<2304 ? state[state_base+d+j]*inv : float(message[bank*2304+d+j-2304]);
                float x=float(bfloat16_t(raw));
                for (uint r=0;r<ROWS;++r) dots[r]=fma(x,float(weight[(bank*64+row+r)*4608+d+j]),dots[r]);
            }
        }
        for (uint r=0;r<ROWS;++r) {
            float dot=dots[r];
            for (ushort j=16;j;j>>=1) dot+=simd_shuffle_down(dot,j);
            if (!lane) parts[sg][r]=dot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid<ROWS) {
            float dot=parts[0][tid];
            for (uint i=1;i<8;++i) dot+=parts[i][tid];
            out[bank*64+row+tid]=bfloat16_t(dot);
        }
    """,
)


def applicable(state, msg, down, up):
    return (msg is not None and state.dtype==mx.float32 and msg.dtype in (mx.bfloat16,mx.float32) and state.shape[-1]==2304
            and down.dtype==mx.bfloat16 and down.shape[-2:]==(64,4608)
            and up.shape[-2:]==(4608,64) and up.dtype in (mx.bfloat16,mx.float32))


def reduce(state,msg,down,inverse=None,widen=False):
    banks=msg.size//2304
    rows=2 if inverse is not None else 4
    return _kernel(inputs=[state,msg,down,state if inverse is None else inverse],
                   template=[("ROWS",rows),("SHARED",state.size!=msg.size),("PRECOMPUTED",inverse is not None)],
                   grid=(64//rows*256,banks,1),threadgroup=(256,1,1),
                   output_shapes=[msg.shape[:-1]+(64,)],output_dtypes=[mx.float32 if widen else mx.bfloat16])[0]
