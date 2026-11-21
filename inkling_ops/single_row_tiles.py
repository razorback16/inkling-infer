"""Lossless BF16 tiles for single-row decode projections with inline exceptions."""
import mlx.core as mx


# Interleave adjacent rows so each projection streams its preferred stripe width.
_STRIPE = "constexpr uint stripe = COLS == 3072 ? 16u : (COLS == 2304 ? 2u : 4u);"

_pack_rows = mx.fast.metal_kernel(
    name="inkling_output_row_bytes", input_names=["w"], output_names=["payload","formats"],
    source=_STRIPE + """
        uint lane=thread_index_in_simdgroup, row=thread_position_in_grid.x/32;
        ushort highest=0;
        for (uint d=lane*4;d<COLS;d+=128) {
            ushort4 bits=*(device const ushort4*)((device const ushort*)w+row*COLS+d);
            ushort4 e=(bits>>8)&127;
            highest=max(highest,max(max(e.x,e.y),max(e.z,e.w)));
        }
        highest=simd_max(highest);
        ushort base=ushort(max(0,int(highest)-7));
        uint raw_tiles=0;
        for (uint tile=0;tile<COLS/128;++tile) {
            uint d=tile*128+lane*4;
            ushort4 bits=*(device const ushort4*)((device const ushort*)w+row*COLS+d);
            ushort4 delta=((bits>>8)&127)-base;
            uint addr=((row/stripe*(COLS/128)+tile)*stripe+row%stripe)*64;
            if (simd_any(any(delta>7))) {
                raw_tiles|=1u<<tile;
                *(device ushort4*)(payload+addr+lane*2)=bits;
            } else {
                payload[addr+lane]=as_type<uint>(uchar4(bits));
                ushort4 n=delta|((bits>>12)&8);
                ((device ushort*)(payload+addr+32))[lane]=n.x|(n.y<<4)|(n.z<<8)|(n.w<<12);
            }
        }
        if (!lane) formats[row]=(uint(base)<<24)|raw_tiles;
    """,
)
DOT=_STRIPE + """
        uint format=formats[row];
        ushort base=ushort(format>>24);
        float dot=0;
        for (uint tile=0;tile<COLS/128;++tile) {
            uint d=tile*128+lane*4, addr=((row/stripe*(COLS/128)+tile)*stripe+row%stripe)*64;
            float4 a=float4(*(device const bfloat4*)(x+d));
            ushort4 bits;
            if (format&(1u<<tile)) {
                bits=*(device const ushort4*)(payload+addr+lane*2);
            } else {
                ushort h=((device const ushort*)(payload+addr+32))[lane];
                ushort4 n=ushort4(h,h>>4,h>>8,h>>12);
                bits=((n&8)<<12)|(((n&7)+base)<<8)|ushort4(as_type<uchar4>(payload[addr+lane]));
            }
            float4 w=float4(as_type<bfloat4>(bits));
            for (uint j=0;j<4;++j) dot=fma(a[j],w[j],dot);
        }
        for (ushort step=16;step;step>>=1) dot+=simd_shuffle_down(dot,step);
"""


def encode(w):
    rows,cols=w.shape
    return _pack_rows(inputs=[w],template=[("COLS",cols)],
                      grid=(rows*32,1,1),threadgroup=(128,1,1),
                      output_shapes=[(rows,cols//128,64),(rows,)],output_dtypes=[mx.uint32,mx.uint32])


_matvec=mx.fast.metal_kernel(name="inkling_single_tile_matvec",input_names=["x","payload","formats"],
    output_names=["out"],source="""
        uint lane=thread_index_in_simdgroup,row=thread_position_in_grid.x/32;
    """ + DOT + """
        if (!lane) out[row]=bfloat16_t(dot);
    """)


def project(x,weight,encoded):
    rows,cols=weight.shape
    return _matvec(inputs=[x,*encoded],template=[("COLS",cols)],grid=(rows*32,1,1),threadgroup=(256,1,1),
                   output_shapes=[(1,1,rows)],output_dtypes=[x.dtype])[0]
