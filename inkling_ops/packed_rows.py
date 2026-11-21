"""Lossless four-row BF16 tiles for 2304-channel decode projections."""
import mlx.core as mx

_ranges = mx.fast.metal_kernel(
    name="inkling_row_exponent_range",input_names=["weight"],output_names=["bases"],
    source="""
        uint lane=thread_index_in_simdgroup, row=thread_position_in_grid.x/32;
        ushort peak=0;
        for (uint d=lane*4;d<2304;d+=128) {
            ushort4 bits=*(device const ushort4*)((device const ushort*)weight+row*2304+d);
            ushort4 e=(bits>>7)&255;
            peak=max(peak,max(max(e.x,e.y),max(e.z,e.w)));
        }
        peak=simd_max(peak);
        if (!lane) bases[row]=uchar(max(int(peak)-15,0));
    """,
)


_pack = mx.fast.metal_kernel(
    name="inkling_encode_four_rows",
    input_names=["weight","bases"], output_names=["payload"],
    source="""
        uint lane=thread_index_in_simdgroup, tile=thread_position_in_grid.x/32;
        uint group=tile/18, block=tile%18;
        uint4 low;
        ushort4 high;
        uchar4 base=*(device const uchar4*)(bases+group*4);
        for (uint row=0;row<4;++row) {
            uint i=(4*group+row)*2304+block*128+lane*4;
            ushort4 bits=*(device const ushort4*)((device const ushort*)weight+i);
            ushort4 e=(bits>>7)&255;
            ushort4 delta=e-ushort(base[row]);
            bool4 escapes=delta>15;
            uchar4 bytes=select(uchar4((bits<<1)|(bits>>15)),uchar4(0),escapes);
            low[row]=as_type<uint>(bytes);
            ushort4 n=select(delta,ushort4(0),escapes);
            high[row]=n.x|(n.y<<4)|(n.z<<8)|(n.w<<12);
        }
        *(device uint4*)(payload+tile*192+lane*4)=low;
        *(device ushort4*)(payload+tile*192+128+lane*2)=high;

    """,
)
_dot_source = """
        uint lane=thread_index_in_simdgroup, group=thread_position_in_grid.x/32;
        uchar4 base=*(device const uchar4*)(bases+group*4);
        float sums[4]={};
        for (uint block=0;block<18;++block) {
            uint tile=group*18+block;
            uint4 low=*(device const uint4*)(payload+tile*192+lane*4);
            ushort4 high=*(device const ushort4*)(payload+tile*192+128+lane*2);
            uint d=block*128+lane*4;
            float4 v=float4(*(device const bfloat4*)(x+d));
            for (uint r=0;r<4;++r) {
                ushort h=high[r];
                ushort4 ex=ushort4(h,h>>4,h>>8,h>>12)&15;
                ushort4 code=(ex<<8)|ushort4(as_type<uchar4>(low[r]));
                ushort4 rotated=code+(ushort(base[r])<<8);
                ushort4 bits=(rotated>>1)|(rotated<<15);
                if (any(code==0)) {
                    bits=*(device const ushort4*)((device const ushort*)weight+(group*4+r)*2304+d);
                }
                float4 w=float4(as_type<bfloat4>(bits));
                for (uint j=0;j<4;++j) sums[r]=fma(v[j],w[j],sums[r]);
            }
        }
        for (uint r=0;r<4;++r) {
            float dot=sums[r];
            for (ushort step=16;step;step>>=1) dot+=simd_shuffle_down(dot,step);
            if (!lane) out[group*4+r]=bfloat16_t(dot);
        }
    """
_dot = mx.fast.metal_kernel(
    name="inkling_four_row_gemv", input_names=["x","weight","payload","bases"],output_names=["out"],
    source=_dot_source,
)

_sink_source = """
        uint group=threadgroup_position_in_grid.x;
        if (group>=MAIN_GROUPS) {
            uint tid=thread_position_in_threadgroup.x,lane=thread_index_in_simdgroup;
            uint sg=simdgroup_index_in_threadgroup, first=(group-MAIN_GROUPS)*4;
            threadgroup float partial[8][4];
            for (uint part=sg;part<8;part+=4) {
                float accum[4]={};
                for (uint d=(part*32+lane)*4;d<2304;d+=1024) {
                    for (uint j=0;j<4;++j) {
                        float v=float(x[d+j]);
                        for (uint r=0;r<4;++r)
                            accum[r]=fma(v,float(sink_weight[(first+r)*2304+d+j]),accum[r]);
                    }
                }
                for (uint r=0;r<4;++r) {
                    float dot=accum[r];
                    for (ushort stride=16;stride;stride>>=1) dot+=simd_shuffle_down(dot,stride);
                    if (!lane) partial[part][r]=dot;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid<4) {
                float dot=partial[0][tid];
                for (uint i=1;i<8;++i) dot+=partial[i][tid];
                sinks[first+tid]=bfloat16_t(dot);
            }
            return;
        }
"""
_with_sinks = mx.fast.metal_kernel(
    name="inkling_four_row_gemv_sinks", input_names=["x","weight","payload","bases","sink_weight"],output_names=["out","sinks"],
    source="{"+_sink_source+"}"+_dot_source,
)


def project_with_sinks(x,weight,encoded,sink_weight):
    rows,heads=weight.shape[0],sink_weight.shape[0]
    return _with_sinks(inputs=[x,weight,*encoded,sink_weight],template=[("MAIN_GROUPS",rows//16)],
                       grid=((rows//16+heads//4)*128,1,1),threadgroup=(128,1,1),
                       output_shapes=[(1,1,rows),(1,1,heads)],output_dtypes=[x.dtype,x.dtype])



def encode(weight):
    count=weight.size
    rows=weight.shape[0]
    bases=_ranges(inputs=[weight],grid=(rows*32,1,1),threadgroup=(128,1,1),
                  output_shapes=[(rows,)],output_dtypes=[mx.uint8])[0]
    payload=_pack(inputs=[weight,bases],grid=(count//16,1,1),threadgroup=(128,1,1),
                 output_shapes=[(count//512,192)],output_dtypes=[mx.uint32])[0]
    return payload,bases


def project(x,weight,encoded):
    rows=weight.shape[0]
    return _dot(inputs=[x,weight,*encoded],grid=(rows*8,1,1),threadgroup=(128,1,1),
                output_shapes=[(1,1,rows)],output_dtypes=[x.dtype])[0]
