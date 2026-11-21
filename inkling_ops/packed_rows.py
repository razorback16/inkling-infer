"""Lossless four-row BF16 tiles for decode projections."""
import mlx.core as mx

_pack = mx.fast.metal_kernel(
    name="inkling_encode_four_rows",
    input_names=["weight"], output_names=["payload","bases"],
    source="""
        uint lane=thread_index_in_simdgroup, tile=thread_position_in_grid.x/32;
        uint group=tile/(COLS/128), block=tile%(COLS/128);
        uint4 low;
        ushort4 high;
        uchar4 base;
        for (uint row=0;row<4;++row) {
            uint i=(4*group+row)*COLS+block*128+lane*4;
            ushort4 bits=*(device const ushort4*)((device const ushort*)weight+i);
            ushort4 e=(bits>>7)&255;
            ushort lo=simd_min(min(min(e.x,e.y),min(e.z,e.w)));
            ushort hi=simd_max(max(max(e.x,e.y),max(e.z,e.w)));
            base[row]=hi-lo>15 ? 255 : lo;
            low[row]=as_type<uint>(uchar4((bits<<1)|(bits>>15)));
            ushort4 n=(e-lo)&15;
            high[row]=n.x|(n.y<<4)|(n.z<<8)|(n.w<<12);
        }
        *(device uint4*)(payload+tile*192+lane*4)=low;
        *(device ushort4*)(payload+tile*192+128+lane*2)=high;
        if (!lane) bases[tile]=as_type<uint>(base);
    """,
)
accumulate = """
        float sums[4]={};
        for (uint block=0;block<COLS/128;++block) {
            uint tile=group*(COLS/128)+block;
            uint4 low=*(device const uint4*)(payload+tile*192+lane*4);
            ushort4 high=*(device const ushort4*)(payload+tile*192+128+lane*2);
            uchar4 base=as_type<uchar4>(bases[tile]);
            uint d=block*128+lane*4;
            float4 v=float4(*(device const bfloat4*)(x+d));
            for (uint r=0;r<4;++r) {
                ushort4 bits;
                if (base[r]==255) {
                    bits=*(device const ushort4*)((device const ushort*)weight+(group*4+r)*COLS+d);
                } else {
                    ushort h=high[r];
                    ushort4 ex=ushort4(h,h>>4,h>>8,h>>12)&15;
                    ushort4 rotated=((ex+base[r])<<8)|ushort4(as_type<uchar4>(low[r]));
                    bits=(rotated>>1)|(rotated<<15);
                }
                float4 w=float4(as_type<bfloat4>(bits));
                for (uint j=0;j<4;++j) sums[r]=fma(v[j],w[j],sums[r]);
            }
        }
    """

_dot = mx.fast.metal_kernel(
    name="inkling_four_row_gemv", input_names=["x","weight","payload","bases"],output_names=["out"],
    source="""
        uint lane=thread_index_in_simdgroup, group=thread_position_in_grid.x/32;
    """ + accumulate + """
        for (uint r=0;r<4;++r) {
            float dot=sums[r];
            for (ushort step=16;step;step>>=1) dot+=simd_shuffle_down(dot,step);
            if (!lane) out[group*4+r]=bfloat16_t(dot);
        }
    """,
)


def encode(weight):
    count=weight.size
    return _pack(inputs=[weight],template=[("COLS",weight.shape[1])],grid=(count//16,1,1),threadgroup=(128,1,1),
                 output_shapes=[(count//512,192),(count//512,)],
                 output_dtypes=[mx.uint32,mx.uint32])


def project(x,weight,encoded):
    rows=weight.shape[0]
    return _dot(inputs=[x,weight,*encoded],template=[("COLS",weight.shape[1])],grid=(rows*8,1,1),threadgroup=(128,1,1),
                output_shapes=[(1,1,rows)],output_dtypes=[x.dtype])[0]
