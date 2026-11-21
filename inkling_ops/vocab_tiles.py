"""Store each BF16 coefficient in twelve bits plus exact overflow nibbles."""
import mlx.core as mx


_formats = mx.fast.metal_kernel(
    name="inkling_vocab_exponent_formats", input_names=["weight"],
    output_names=["formats", "counts"], source="""
        uint row=thread_position_in_grid.x/32, lane=thread_index_in_simdgroup;
        ushort4 exponents[COLS/128];
        ushort highest=0;
        for (uint t=0;t<COLS/128;++t) {
            ushort4 bits=((device const ushort4*)weight)[row*(COLS/4)+t*32+lane];
            ushort4 e=(bits>>7)&255;
            exponents[t]=e;
            highest=max(highest,max(max(e.x,e.y),max(e.z,e.w)));
        }
        ushort base=ushort(max(0,int(simd_max(highest))-15));
        uint mask=0;
        for (uint t=0;t<COLS/128;++t)
            if (simd_any(any(exponents[t]-base>15))) mask|=1u<<t;
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
        uint row=group/(COLS/128), tile=group%(COLS/128);
        uint addr=((row/4*(COLS/128)+tile)*4+row%4)*48;
        ushort4 bits=((device const ushort4*)weight)[group*32+lane];
        uint fmt=formats[row], mask=fmt&((1u<<(COLS/128))-1u);
        ushort4 e=(((bits>>7)&255)-ushort(fmt>>24))&255;
        tiles[addr+lane]=as_type<uint>(uchar4((bits<<1)|(bits>>15)));
        ushort4 n=e&15;
        ((device ushort*)(tiles+addr+32))[lane]=n.x|(n.y<<4)|(n.z<<8)|(n.w<<12);
        if (mask&(1u<<tile)) {
            uint index=offsets[row]+popcount(mask&((1u<<tile)-1));
            n=e>>4;
            ((device ushort*)overflow)[index*32+lane]=n.x|(n.y<<4)|(n.z<<8)|(n.w<<12);
        }
    """,
)

DOT = """
        uint fmt=formats[row], mask=fmt&((1u<<(COLS/128))-1u);
        ushort base=ushort(fmt>>24)<<8;
        float dot=0;
        for (uint t=0;t<COLS/128;++t) {
            uint addr=((row/4*(COLS/128)+t)*4+row%4)*48;
            ushort h=((device const ushort*)(tiles+addr+32))[lane];
            ushort4 n=ushort4(h,h>>4,h>>8,h>>12)&15;
            if (mask&(1u<<t)) {
                uint index=offsets[row]+popcount(mask&((1u<<t)-1));
                ushort hi=((device const ushort*)overflow)[index*32+lane];
                n|=(ushort4(hi,hi>>4,hi>>8,hi>>12)&15)<<4;
            }
            ushort4 rot=((n<<8)|ushort4(as_type<uchar4>(tiles[addr+lane])))+base;
            float4 w=float4(as_type<bfloat4>((rot>>1)|(rot<<15)));
            float4 a=float4(((device const bfloat4*)x)[t*32+lane]);
            for (uint j=0;j<4;++j) dot=fma(a[j],w[j],dot);
        }
        for (ushort step=16;step;step>>=1) dot+=simd_shuffle_down(dot,step);
"""

_dot = mx.fast.metal_kernel(
    name="inkling_vocab_twelve_bit_dot", input_names=["x","tiles","overflow","formats","offsets"],
    output_names=["out"], source="""
        uint lane=thread_index_in_simdgroup, row=thread_position_in_grid.x/32;
    """ + DOT + """
        if (!lane) out[row]=bfloat16_t(dot);
    """,
)


def encode(w):
    rows,cols=w.shape
    formats,counts=_formats(inputs=[w], template=[("COLS",cols)], grid=(rows*32,1,1), threadgroup=(128,1,1),
                            output_shapes=[(rows,),(rows,)], output_dtypes=[mx.uint32]*2)
    ends=mx.cumsum(counts)
    n_overflow=max(1,int(ends[-1].item()))
    offsets=ends-counts
    tiles,overflow=_pack(inputs=[w,formats,offsets], template=[("COLS",cols)], grid=(rows*(cols//128)*32,1,1), threadgroup=(128,1,1),
                         output_shapes=[(rows,cols//128,48),(n_overflow,16)], output_dtypes=[mx.uint32]*2)
    return tiles,overflow,formats,offsets


def project(x,weight,encoded):
    rows=weight.shape[0]
    return _dot(inputs=[x,*encoded], template=[("COLS",weight.shape[1])], grid=(rows*32,1,1), threadgroup=(256,1,1),
                output_shapes=[(1,1,rows)], output_dtypes=[x.dtype])[0]
