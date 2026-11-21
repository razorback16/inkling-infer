"""Store each vocabulary coefficient in twelve bits plus exact overflow nibbles."""
import mlx.core as mx


_formats = mx.fast.metal_kernel(
    name="inkling_vocab_exponent_formats", input_names=["weight"],
    output_names=["formats", "counts"], source="""
        uint row=thread_position_in_grid.x/32, lane=thread_index_in_simdgroup;
        ushort4 exponents[18];
        ushort highest=0;
        for (uint t=0;t<18;++t) {
            ushort4 bits=((device const ushort4*)weight)[row*576+t*32+lane];
            ushort4 e=(bits>>8)&127;
            exponents[t]=e;
            highest=max(highest,max(max(e.x,e.y),max(e.z,e.w)));
        }
        ushort base=ushort(max(0,int(simd_max(highest))-7));
        uint mask=0;
        for (uint t=0;t<18;++t)
            if (simd_any(any(exponents[t]-base>7))) mask|=1u<<t;
        if (!lane) {
            formats[row]=(uint(base)<<24)|mask;
            counts[row]=popcount(mask);
        }
    """,
)

_pack = mx.fast.metal_kernel(
    name="inkling_vocab_twelve_bit_tiles", input_names=["weight","formats","offsets"],
    output_names=["tiles","overflow"], source="""
        uint lane=thread_index_in_simdgroup, group=thread_position_in_grid.x/32;
        uint row=group/18, tile=group%18;
        uint addr=((row/4*18+tile)*4+row%4)*48;
        ushort4 bits=((device const ushort4*)weight)[group*32+lane];
        uint fmt=formats[row], mask=fmt&0x3ffffu;
        ushort4 e=(((bits>>8)&127)-ushort(fmt>>24))&127;
        tiles[addr+lane]=as_type<uint>(uchar4(bits));
        ushort4 n=(e&7)|((bits>>12)&8);
        ((device ushort*)(tiles+addr+32))[lane]=n.x|(n.y<<4)|(n.z<<8)|(n.w<<12);
        if (mask&(1u<<tile)) {
            uint index=offsets[row]+popcount(mask&((1u<<tile)-1));
            n=e>>3;
            ((device ushort*)overflow)[index*32+lane]=n.x|(n.y<<4)|(n.z<<8)|(n.w<<12);
        }
    """,
)

_dot = mx.fast.metal_kernel(
    name="inkling_vocab_twelve_bit_dot", input_names=["x","tiles","overflow","formats","offsets"],
    output_names=["out"], source="""
        uint lane=thread_index_in_simdgroup, row=thread_position_in_grid.x/32;
        uint fmt=formats[row], mask=fmt&0x3ffffu;
        ushort base=ushort(fmt>>24);
        float dot=0;
        if (mask==0) {
            for (uint t=0;t<18;++t) {
                uint addr=((row/4*18+t)*4+row%4)*48;
                ushort h=((device const ushort*)(tiles+addr+32))[lane];
                ushort4 codes=ushort4(h,h>>4,h>>8,h>>12);
                ushort4 n=codes&7;
                ushort4 bits=((codes&8)<<12)|(((n+base)&127)<<8)|ushort4(as_type<uchar4>(tiles[addr+lane]));
                float4 w=float4(as_type<bfloat4>(bits));
                float4 a=float4(((device const bfloat4*)x)[t*32+lane]);
                for (uint j=0;j<4;++j) dot=fma(a[j],w[j],dot);
            }
        } else {
            for (uint t=0;t<18;++t) {
                uint addr=((row/4*18+t)*4+row%4)*48;
                ushort h=((device const ushort*)(tiles+addr+32))[lane];
                ushort4 codes=ushort4(h,h>>4,h>>8,h>>12);
                ushort4 n=codes&7;
                if (mask&(1u<<t)) {
                    uint index=offsets[row]+popcount(mask&((1u<<t)-1));
                    ushort hi=((device const ushort*)overflow)[index*32+lane];
                    n|=(ushort4(hi,hi>>4,hi>>8,hi>>12)&15)<<3;
                }
                ushort4 bits=((codes&8)<<12)|(((n+base)&127)<<8)|ushort4(as_type<uchar4>(tiles[addr+lane]));
                float4 w=float4(as_type<bfloat4>(bits));
                float4 a=float4(((device const bfloat4*)x)[t*32+lane]);
                for (uint j=0;j<4;++j) dot=fma(a[j],w[j],dot);
            }
        }
        for (ushort step=16;step;step>>=1) dot+=simd_shuffle_down(dot,step);
        if (!lane) out[row]=bfloat16_t(dot);
    """,
)


def encode(w):
    rows=w.shape[0]
    formats,counts=_formats(inputs=[w], grid=(rows*32,1,1), threadgroup=(128,1,1),
                            output_shapes=[(rows,),(rows,)], output_dtypes=[mx.uint32]*2)
    ends=mx.cumsum(counts)
    n_overflow=max(1,int(ends[-1].item()))
    offsets=ends-counts
    tiles,overflow=_pack(inputs=[w,formats,offsets], grid=(rows*18*32,1,1), threadgroup=(128,1,1),
                         output_shapes=[(rows,18,48),(n_overflow,16)], output_dtypes=[mx.uint32]*2)
    return tiles,overflow,formats,offsets


def project(x,weight,encoded):
    rows=weight.shape[0]
    return _dot(inputs=[x,*encoded], grid=(rows*32,1,1), threadgroup=(256,1,1),
                output_shapes=[(1,1,rows)], output_dtypes=[x.dtype])[0]
