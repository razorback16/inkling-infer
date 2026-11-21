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
    input_names=["weight","bases"], output_names=["payload","flags"],
    source="""
        uint lane=thread_index_in_simdgroup, tile=thread_position_in_grid.x/32;
        uint group=tile/18, block=tile%18;
        uint4 low;
        ushort4 high;
        bool needs_raw=false;
        uchar4 base=*(device const uchar4*)(bases+group*4);
        for (uint row=0;row<4;++row) {
            uint i=(4*group+row)*2304+block*128+lane*4;
            ushort4 bits=*(device const ushort4*)((device const ushort*)weight+i);
            ushort4 e=(bits>>7)&255;
            ushort4 delta=e-ushort(base[row]);
            bool4 escapes=delta>15;
            needs_raw = needs_raw || any(escapes);
            uchar4 bytes=uchar4((bits<<1)|(bits>>15));
            low[row]=as_type<uint>(bytes);
            ushort4 n=delta&15;
            high[row]=n.x|(n.y<<4)|(n.z<<8)|(n.w<<12);
        }
        uint mask=simd_sum(uint(needs_raw)<<lane);
        if (!lane) flags[tile]=mask;
        *(device uint4*)(payload+tile*192+lane*4)=low;
        *(device ushort4*)(payload+tile*192+128+lane*2)=high;

    """,
)
_dot = mx.fast.metal_kernel(
    name="inkling_four_row_gemv", input_names=["x","weight","payload","bases","flags"],output_names=["out"],
    source="""
        uint lane=thread_index_in_simdgroup, group=thread_position_in_grid.x/32;
        uchar4 base=*(device const uchar4*)(bases+group*4);
        float sums[4]={};
        for (uint block=0;block<18;++block) {
            uint tile=group*18+block;
            uint4 low=*(device const uint4*)(payload+tile*192+lane*4);
            ushort4 high=*(device const ushort4*)(payload+tile*192+128+lane*2);
            uint d=block*128+lane*4;
            float4 v=float4(*(device const bfloat4*)(x+d));
            if (flags[tile] & (1u<<lane)) {
                for (uint r=0;r<4;++r) {
                    float4 w=float4(*(device const bfloat4*)(weight+(group*4+r)*2304+d));
                    for (uint j=0;j<4;++j) sums[r]=fma(v[j],w[j],sums[r]);
                }
            } else {
                for (uint r=0;r<4;++r) {
                    ushort h=high[r];
                    ushort4 ex=ushort4(h,h>>4,h>>8,h>>12)&15;
                    ushort4 code=(ex<<8)|ushort4(as_type<uchar4>(low[r]));
                    ushort4 rot=code+(ushort(base[r])<<8);
                    float4 w=float4(as_type<bfloat4>((rot>>1)|(rot<<15)));
                    for (uint j=0;j<4;++j) sums[r]=fma(v[j],w[j],sums[r]);
                }
            }
        }
        for (uint r=0;r<4;++r) {
            float dot=sums[r];
            for (ushort step=16;step;step>>=1) dot+=simd_shuffle_down(dot,step);
            if (!lane) out[group*4+r]=bfloat16_t(dot);
        }
    """,
)


def encode(weight):
    count=weight.size
    rows=weight.shape[0]
    bases=_ranges(inputs=[weight],grid=(rows*32,1,1),threadgroup=(128,1,1),
                  output_shapes=[(rows,)],output_dtypes=[mx.uint8])[0]
    payload,flags=_pack(inputs=[weight,bases],grid=(count//16,1,1),threadgroup=(128,1,1),
                 output_shapes=[(count//512,192),(count//512,)],output_dtypes=[mx.uint32,mx.uint32])
    return payload,bases,flags


def project(x,weight,encoded):
    rows=weight.shape[0]
    return _dot(inputs=[x,weight,*encoded],grid=(rows*8,1,1),threadgroup=(128,1,1),
                output_shapes=[(1,1,rows)],output_dtypes=[x.dtype])[0]
