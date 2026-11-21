"""Lossless BF16 tiles for single-row decode projections with inline exceptions."""
import mlx.core as mx


_pack_rows = mx.fast.metal_kernel(
    name="inkling_output_row_bytes", input_names=["w"], output_names=["payload","formats"],
    source="""
        uint lane=thread_index_in_simdgroup, row=thread_position_in_grid.x/32;
        ushort highest=0;
        for (uint d=lane*4;d<COLS;d+=128) {
            ushort4 bits=*(device const ushort4*)((device const ushort*)w+row*COLS+d);
            ushort4 e=(bits>>7)&255;
            highest=max(highest,max(max(e.x,e.y),max(e.z,e.w)));
        }
        highest=simd_max(highest);
        ushort base=ushort(max(0,int(highest)-15));
        uint raw_tiles=0;
        for (uint tile=0;tile<COLS/128;++tile) {
            uint d=tile*128+lane*4;
            ushort4 bits=*(device const ushort4*)((device const ushort*)w+row*COLS+d);
            ushort4 delta=((bits>>7)&255)-base;
            uint addr=((row/4*(COLS/128)+tile)*4+row%4)*64;
            if (simd_any(any(delta>15))) {
                raw_tiles|=1u<<tile;
                *(device ushort4*)(payload+addr+lane*2)=bits;
            } else {
                uchar4 low=uchar4((bits<<1)|(bits>>15));
                ushort2 pair=ushort2(low.xy)|(delta.xy<<8)|(delta.zw<<12);
                payload[addr+lane]=as_type<uint>(pair);
                ((device ushort*)(payload+addr+32))[lane]=as_type<ushort>(low.zw);
            }
        }
        if (!lane) formats[row]=(uint(base)<<24)|raw_tiles;
    """,
)
DOT="""
        uint format=formats[row];
        ushort base=ushort(format>>24)<<8;
        float dot=0;
        for (uint tile=0;tile<COLS/128;++tile) {
            uint d=tile*128+lane*4, addr=((row/4*(COLS/128)+tile)*4+row%4)*64;
            float4 a=float4(*(device const bfloat4*)(x+d));
            ushort4 bits;
            if (format&(1u<<tile)) {
                bits=*(device const ushort4*)(payload+addr+lane*2);
            } else {
                ushort2 pair=as_type<ushort2>(payload[addr+lane]);
                ushort extra=((device const ushort*)(payload+addr+32))[lane];
                ushort2 other=((pair>>12)<<8)|ushort2(as_type<uchar2>(extra));
                ushort4 code=ushort4(pair&4095,other);
                ushort4 rot=code+base;
                bits=(rot>>1)|(rot<<15);
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
