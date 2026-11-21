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
        bool raw=false;
        uint4 low;
        ushort4 high;
        uchar4 base=*(device const uchar4*)(bases+group*4);
        for (uint row=0;row<4;++row) {
            uint i=(4*group+row)*2304+block*128+lane*4;
            ushort4 bits=*(device const ushort4*)((device const ushort*)weight+i);
            ushort4 e=(bits>>7)&255;
            ushort4 delta=e-ushort(base[row]);
            bool4 escapes=delta>15;
            raw |= any(escapes);
            uchar4 bytes=uchar4((bits<<1)|(bits>>15));
            low[row]=as_type<uint>(bytes);
            ushort4 n=delta&15;
            high[row]=n.x|(n.y<<4)|(n.z<<8)|(n.w<<12);
        }
        bool any_raw=simd_any(raw);
        if (!lane) flags[tile]=uint(any_raw)<<block;
        *(device uint4*)(payload+tile*192+lane*4)=low;
        *(device ushort4*)(payload+tile*192+128+lane*2)=high;

    """,
)
_dot = mx.fast.metal_kernel(
    name="inkling_four_row_gemv", input_names=["x","weight","payload","bases","flags"],output_names=["out"],
    source="""
        uint lane=thread_index_in_simdgroup, group=thread_position_in_grid.x/32;
        uchar4 base=*(device const uchar4*)(bases+group*4);
        uint raw_tiles=flags[group];
        float sums[4]={};
        for (uint block=0;block<18;++block) {
            uint tile=group*18+block;
            uint d=block*128+lane*4;
            float4 v=float4(*(device const bfloat4*)(x+d));
            if (raw_tiles & (1u<<block)) {
                for (uint r=0;r<4;++r) {
                    float4 w=float4(*(device const bfloat4*)(weight+(group*4+r)*2304+d));
                    for (uint j=0;j<4;++j) sums[r]=fma(v[j],w[j],sums[r]);
                }
                continue;
            }
            uint4 low=*(device const uint4*)(payload+tile*192+lane*4);
            ushort4 high=*(device const ushort4*)(payload+tile*192+128+lane*2);
            for (uint r=0;r<4;++r) {
                ushort h=high[r];
                ushort4 ex=ushort4(h,h>>4,h>>8,h>>12)&15;
                ushort4 code=(ex<<8)|ushort4(as_type<uchar4>(low[r]));
                ushort4 rotated=code+(ushort(base[r])<<8);
                ushort4 bits=(rotated>>1)|(rotated<<15);
                float4 w=float4(as_type<bfloat4>(bits));
                for (uint j=0;j<4;++j) sums[r]=fma(v[j],w[j],sums[r]);
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
                 output_shapes=[(count//512,192),(rows//4,18)],output_dtypes=[mx.uint32,mx.uint32])
    return payload,bases,mx.sum(flags,axis=1)


def project(x,weight,encoded):
    rows=weight.shape[0]
    return _dot(inputs=[x,weight,*encoded],grid=(rows*8,1,1),threadgroup=(128,1,1),
                output_shapes=[(1,1,rows)],output_dtypes=[x.dtype])[0]

_pack11 = mx.fast.metal_kernel(
    name="inkling_encode_sparse_exponents",
    input_names=["weight","bases"], output_names=["payload","flags","lane_bits","codes","counts"],
    source="""
        uint lane=thread_index_in_simdgroup, tile=thread_position_in_grid.x/32;
        uint group=tile/18, block=tile%18;
        bool raw=false;
        ushort corrections=0;
        uint4 low;
        ushort4 high;
        uchar4 base=*(device const uchar4*)(bases+group*4);
        for (uint row=0;row<4;++row) {
            uint i=(4*group+row)*2304+block*128+lane*4;
            ushort4 bits=*(device const ushort4*)((device const ushort*)weight+i);
            ushort4 e=(bits>>7)&255;
            ushort4 delta=e-ushort(base[row]);
            bool4 escapes=delta>15;
            raw |= any(escapes);
            uchar4 bytes=uchar4((bits<<1)|(bits>>15));
            low[row]=as_type<uint>(bytes);
            ushort4 n=delta&15;
            ushort4 h=(n^8)>>3;
            corrections|=(h.x|(h.y<<1)|(h.z<<2)|(h.w<<3))<<(4*row);
            n &= 7;
            high[row]=n.x|(n.y<<3)|(n.z<<6)|(n.w<<9);
        }
        bool any_raw=simd_any(raw);
        if (!lane) flags[tile]=uint(any_raw)<<block;
        uint lanes=simd_or(corrections && !any_raw ? (1u<<lane) : 0u);
        if (!lane) {
            lane_bits[tile]=lanes;
            counts[tile]=popcount(lanes);
        }
        codes[tile*32+lane]=corrections;
        *(device uint4*)(payload+tile*176+lane*4)=low;
        payload[tile*176+128+lane]=uint(high.x)|(uint(high.y)<<12)|(uint(high.z)<<24);
        ((device ushort*)(payload+tile*176+160))[lane]=(high.z>>8)|(high.w<<4);
    """,
)
_dot11 = mx.fast.metal_kernel(
    name="inkling_sparse_exponent_dot", input_names=["x","weight","payload","bases","flags","headers","rare"],output_names=["out"],
    source="""
        uint lane=thread_index_in_simdgroup, group=thread_position_in_grid.x/32;
        uchar4 base=*(device const uchar4*)(bases+group*4);
        uint raw_tiles=flags[group];
        float sums[4]={};
        for (uint block=0;block<18;++block) {
            uint tile=group*18+block;
            uint d=block*128+lane*4;
            float4 v=float4(*(device const bfloat4*)(x+d));
            if (raw_tiles & (1u<<block)) {
                for (uint r=0;r<4;++r) {
                    float4 w=float4(*(device const bfloat4*)(weight+(group*4+r)*2304+d));
                    for (uint j=0;j<4;++j) sums[r]=fma(v[j],w[j],sums[r]);
                }
                continue;
            }
            uint4 low=*(device const uint4*)(payload+tile*176+lane*4);
            uint first=payload[tile*176+128+lane];
            uint last=((device const ushort*)(payload+tile*176+160))[lane];
            ushort4 high=ushort4(first,first>>12,(first>>24)|(last<<8),last>>4);
            uint2 header=*(device const uint2*)(headers+2*tile);
            uint mask=header.x, corrections=0;
            if (mask & (1u<<lane)) corrections=rare[header.y+popcount(mask&((1u<<lane)-1u))];
            for (uint r=0;r<4;++r) {
                ushort h=high[r];
                ushort4 ex=ushort4(h,h>>3,h>>6,h>>9)&7;
                ushort c=ushort(corrections>>(r*4));
                ex|=(((ushort4(c,c>>1,c>>2,c>>3)&1)^1)<<3);
                ushort4 code=(ex<<8)|ushort4(as_type<uchar4>(low[r]));
                ushort4 rotated=code+(ushort(base[r])<<8);
                ushort4 bits=(rotated>>1)|(rotated<<15);
                float4 w=float4(as_type<bfloat4>(bits));
                for (uint j=0;j<4;++j) sums[r]=fma(v[j],w[j],sums[r]);
            }
        }
        for (uint r=0;r<4;++r) {
            float dot=sums[r];
            for (ushort step=16;step;step>>=1) dot+=simd_shuffle_down(dot,step);
            if (!lane) out[group*4+r]=bfloat16_t(dot);
        }
    """,
)



_compact_bits = mx.fast.metal_kernel(
    name="inkling_compact_exponent_corrections",input_names=["lanes","codes","end"],
    output_names=["headers","rare"],
    source="""
        uint lane=thread_index_in_simdgroup,tile=thread_position_in_grid.x/32;
        uint mask=lanes[tile],start=end[tile]-popcount(mask);
        if (!lane) *(device uint2*)(headers+2*tile)=uint2(mask,start);
        if (mask & (1u<<lane))
            rare[start+popcount(mask&((1u<<lane)-1u))]=codes[tile*32+lane];
    """)


def encode_sparse(w):
    rows=w.shape[0]
    tiles=w.size//512
    bases=_ranges(inputs=[w],grid=(rows*32,1,1),threadgroup=(128,1,1),
                  output_shapes=[(rows,)],output_dtypes=[mx.uint8])[0]
    data,flags,lanes,codes,counts=_pack11(inputs=[w,bases],grid=(tiles*32,1,1),threadgroup=(128,1,1),
        output_shapes=[(tiles,176),(rows//4,18),(tiles,),(tiles,32),(tiles,)],
        output_dtypes=[mx.uint32,mx.uint32,mx.uint32,mx.uint16,mx.uint32])
    end=mx.cumsum(counts)
    size=max(1,int(end[-1].item()))
    headers,rare=_compact_bits(inputs=[lanes,codes,end],grid=(tiles*32,1,1),threadgroup=(128,1,1),
                               output_shapes=[(tiles,2),(size,)],output_dtypes=[mx.uint32,mx.uint16])
    return data,bases,mx.sum(flags,axis=1),headers,rare


def sparse_project(x,weight,encoded):
    rows=weight.shape[0]
    return _dot11(inputs=[x,weight,*encoded],grid=(rows*8,1,1),threadgroup=(128,1,1),
                  output_shapes=[(1,1,rows)],output_dtypes=[x.dtype])[0]
