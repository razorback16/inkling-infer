"""Project four NVFP4 down rows per SIMD group, scheduling adjacent rows per expert."""
import mlx.core as mx

_project = mx.fast.metal_kernel(
    name="inkling_expert_ordered_down",input_names=["hidden","ids","w","scales","amax"],output_names=["out"],
    header="""
        constant float levels[16]={0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6};
        inline float block_scale(uchar value) {
            ushort m=value&127;
            ushort bits=(m<<7)|(((m+1)>>7)<<14)|((value&128)<<8);
            return float(as_type<half>(bits))*256.f;
        }
    """,
    source="""
        uint lane = thread_index_in_simdgroup, slot = threadgroup_position_in_grid.y;
        uint expert = ids[slot], col = (thread_position_in_grid.x/32)*4;
        float sums[4]={};
        for (uint word=lane; word<144; word+=32) {
            float x[8];
            for (uint j=0;j<8;++j) x[j]=float(hidden[slot*1152+word*8+j]);
            for (uint r=0;r<4;++r) {
                uint row=expert*2304+col+r, q=w[row*144+word];
                float chunk=0;
                for (uint j=0;j<8;j+=4) {
                    chunk+=x[j]*levels[q&15]+x[j+1]*levels[(q>>4)&15]
                        +x[j+2]*levels[(q>>8)&15]+x[j+3]*levels[(q>>12)&15];
                    q>>=16;
                }
                sums[r]+=block_scale(scales[row*72+word/2])*chunk;
            }
        }
        for (uint r=0;r<4;++r) {
            float value=simd_sum(sums[r])*(amax[expert]/2688.f);
            if (!lane) out[slot*2304+col+r]=bfloat16_t(value);
        }
    """,
)


def project(hidden, ids, down):
    return _project(inputs=[hidden,ids,down.weight,down.scales,down.global_scale],
                    grid=(576*32,8,1),threadgroup=(128,1,1),
                    output_shapes=[(1,1,8,2304)],output_dtypes=[mx.bfloat16])[0]
