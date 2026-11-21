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
        uint physical=((group/4)*18+block)*4+group%4;
        if (any_raw) {
            for (uint r=0;r<4;++r) {
                uint i=(4*group+r)*2304+block*128+lane*4;
                *(device uint2*)(payload+physical*256+r*64+lane*2)=
                    *(device const uint2*)((device const ushort*)weight+i);
            }
        } else {
            ushort4 paired[2];
            for (uint pair=0;pair<2;++pair) {
                uint r=pair*2;
                ushort a=high[r],b=high[r+1];
                ushort4 ea=ushort4(a,a>>4,a>>8,a>>12)&15;
                ushort4 eb=ushort4(b,b>>4,b>>8,b>>12)&15;
                paired[pair]=ushort4(as_type<uchar4>(low[r]))|(ea<<8)|(eb<<12);
            }
            *(device uint4*)(payload+physical*256+lane*4)=uint4(as_type<uint2>(paired[0]),as_type<uint2>(paired[1]));
            *(device uint2*)(payload+physical*256+128+lane*2)=uint2(low.y,low.w);
        }

    """,
)
_dot = mx.fast.metal_kernel(
    name="inkling_four_row_gemv", input_names=["x","payload"],output_names=["out"],
    source="""
        uint lane=thread_index_in_simdgroup, group=thread_position_in_grid.x/32;
        uint2 meta=*(device const uint2*)(payload+(group/4)*18688+(group%4)*2);
        uchar4 base=as_type<uchar4>(meta.x);
        uint raw_tiles=meta.y;
        float sums[4]={};
        for (uint block=0;block<18;++block) {
            uint tile=(group/4)*73+1+block*4+group%4;
            uint d=block*128+lane*4;
            float4 v=float4(*(device const bfloat4*)(x+d));
            if (raw_tiles & (1u<<block)) {
                for (uint r=0;r<4;++r) {
                    float4 w=float4(*(device const bfloat4*)(payload+tile*256+r*64+lane*2));
                    for (uint j=0;j<4;++j) sums[r]=fma(v[j],w[j],sums[r]);
                }
                continue;
            }
            uint4 pair_words=*(device const uint4*)(payload+tile*256+lane*4);
            uint2 low=*(device const uint2*)(payload+tile*256+128+lane*2);
            for (uint r=0;r<4;++r) {
                ushort4 both=as_type<ushort4>(r<2 ? pair_words.xy : pair_words.zw);
                ushort4 code=(r&1) ? (((both>>12)<<8)|ushort4(as_type<uchar4>(low[r/2]))) : (both&4095);
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
                 output_shapes=[(count//512,256),(rows//4,18)],output_dtypes=[mx.uint32,mx.uint32])
    formats=mx.stack((bases.view(mx.uint32),mx.sum(flags,axis=1)),axis=1)
    # One aligned header per set of sixteen rows, followed by its 72 coefficient tiles.
    headers=mx.pad(formats.reshape(-1,8),[(0,0),(0,248)])
    return (mx.concatenate((headers,payload.reshape(-1,18432)),axis=1),)


def project(x,weight,encoded):
    rows=weight.shape[0]
    return _dot(inputs=[x,*encoded],grid=(rows*8,1,1),threadgroup=(128,1,1),
                output_shapes=[(1,1,rows)],output_dtypes=[x.dtype])[0]
