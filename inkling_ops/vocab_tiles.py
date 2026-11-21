"""Eleven-bit vocabulary coefficients with exact BF16 escapes."""
import mlx.core as mx


_counts = mx.fast.metal_kernel(
    name="inkling_vocab_escape_counts", input_names=["weight"], output_names=["counts"],
    source="""
        uint group=thread_position_in_grid.x/32, lane=thread_index_in_simdgroup;
        uint count=0;
        for (uint t=0;t<2;++t) {
            ushort4 bits=((device const ushort4*)weight)[group*64+t*32+lane];
            ushort4 e=(bits>>7)&255;
            for (uint j=0;j<4;++j) count+=uint(e[j]<116 || e[j]>122);
        }
        count=simd_sum(count);
        if (!lane) counts[group]=count;
    """,
)

_pack = mx.fast.metal_kernel(
    name="inkling_vocab_pack_eleven",input_names=["weight","offsets"],
    output_names=["tiles","overflow"],source="""
        uint group=thread_position_in_grid.x/32,lane=thread_index_in_simdgroup;
        uint row=group/9,first=(group%9)*2,prior=0;
        for (uint t=0;t<2;++t) {
            ushort4 bits=((device const ushort4*)weight)[group*64+t*32+lane];
            ushort4 e=(bits>>7)&255;
            bool4 rare=(e<116)|(e>122);
            uint n=0;
            for (uint j=0;j<4;++j) n+=uint(rare[j]);
            uint index=prior+simd_prefix_exclusive_sum(n);
            prior+=simd_sum(n);
            uchar4 low=uchar4((bits<<1)|(bits>>15));
            ushort4 code=e-116;
            for (uint j=0;j<4;++j) {
                if (rare[j]) {
                    low[j]=uchar(index); code[j]=7;
                    overflow[offsets[group]+index++]=bits[j];
                }
            }
            uint addr=((row/4*18+first+t)*4+row%4)*44;
            tiles[addr+lane]=as_type<uint>(low);
            ushort4 mid=code&3;
            ((device uchar*)(tiles+addr+32))[lane]=uchar(mid.x|(mid.y<<2)|(mid.z<<4)|(mid.w<<6));
            ushort4 hi=code>>2;
            uint nibble=hi.x|(hi.y<<1)|(hi.z<<2)|(hi.w<<3);
            uint adjacent=simd_shuffle_xor(nibble,1);
            if (!(lane&1)) ((device uchar*)(tiles+addr+40))[lane/2]=uchar(nibble|(adjacent<<4));
        }
    """,
)

_dot = mx.fast.metal_kernel(
    name="inkling_vocab_eleven_bit_dot",input_names=["x","tiles","overflow","offsets"],
    output_names=["out"],source="""
        uint lane=thread_index_in_simdgroup,row=thread_position_in_grid.x/32;
        float dot=0;
        for (uint t=0;t<18;++t) {
            uint addr=((row/4*18+t)*4+row%4)*44;
            ushort4 low=ushort4(as_type<uchar4>(tiles[addr+lane]));
            ushort mid=((device const uchar*)(tiles+addr+32))[lane];
            ushort hi=((device const uchar*)(tiles+addr+40))[lane/2]>>((lane&1)*4);
            ushort4 code=(ushort4(mid,mid>>2,mid>>4,mid>>6)&3)
                         | ((ushort4(hi,hi>>1,hi>>2,hi>>3)&1)<<2);
            ushort4 rot=((code+116)<<8)|low;
            ushort4 bits=(rot>>1)|(rot<<15);
            for (uint j=0;j<4;++j)
                if (code[j]==7) bits[j]=overflow[offsets[row*9+t/2]+low[j]];
            float4 w=float4(as_type<bfloat4>(bits));
            float4 a=float4(((device const bfloat4*)x)[t*32+lane]);
            for (uint j=0;j<4;++j) dot=fma(a[j],w[j],dot);
        }
        for (ushort step=16;step;step>>=1) dot+=simd_shuffle_down(dot,step);
        if (!lane) out[row]=bfloat16_t(dot);
    """,
)


def encode(w):
    rows=w.shape[0]
    grid=(rows*9*32,1,1)
    counts,=_counts(inputs=[w],grid=grid,threadgroup=(128,1,1),
                    output_shapes=[(rows,9)],output_dtypes=[mx.uint32])
    ends=mx.cumsum(counts.reshape(-1))
    size=max(1,int(ends[-1].item()))
    offsets=ends-counts.reshape(-1)
    tiles,overflow=_pack(inputs=[w,offsets],grid=grid,threadgroup=(128,1,1),
                         output_shapes=[(rows,18,44),(size,)],output_dtypes=[mx.uint32,mx.uint16])
    return tiles,overflow,offsets


def project(x,weight,encoded):
    rows=weight.shape[0]
    return _dot(inputs=[x,*encoded], grid=(rows*32,1,1), threadgroup=(256,1,1),
                output_shapes=[(1,1,rows)], output_dtypes=[x.dtype])[0]
