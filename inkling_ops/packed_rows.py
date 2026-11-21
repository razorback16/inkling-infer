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
_dot = mx.fast.metal_kernel(
    name="inkling_four_row_gemv", input_names=["x","weight","payload","bases"],output_names=["out"],
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
    """,
)


_classify = mx.fast.metal_kernel(
    name="inkling_classify_dense_tiles",input_names=["w","bases"],output_names=["masks","counts"],
    source="""
        uint lane=thread_index_in_simdgroup, group=thread_position_in_grid.x/32, mask=0;
        uchar4 base=*(device const uchar4*)(bases+group*4);
        for (uint block=0;block<18;++block) {
            bool escape=false;
            for (uint r=0;r<4;++r) {
                uint d=(group*4+r)*2304+block*128+lane*4;
                ushort4 b=*(device const ushort4*)((device const ushort*)w+d);
                escape|=any(ushort4(((b>>7)&255)-ushort(base[r]))>15);
            }
            if (simd_any(escape)) mask|=1u<<block;
        }
        if (!lane) { masks[group]=mask; counts[group]=popcount(mask); }
    """,
)
_mixed_pack = mx.fast.metal_kernel(
    name="inkling_pack_mixed_dense_tiles",input_names=["w","bases","masks","ends"],output_names=["data","headers"],
    source="""
        uint lane=thread_index_in_simdgroup, tile=thread_position_in_grid.x/32;
        uint group=tile/18, block=tile%18, mask=masks[group];
        uint start=group*18*192+(ends[group]-popcount(mask))*64;
        uint ptr=start+block*192+popcount(mask&((1u<<block)-1))*64;
        if (!block && !lane) *(device uint2*)(headers+group*2)=uint2(start,mask);
        uchar4 base=*(device const uchar4*)(bases+group*4);
        uint4 lo;
        ushort4 hi;
        bool raw=mask&(1u<<block);
        for (uint r=0;r<4;++r) {
            uint d=(group*4+r)*2304+block*128+lane*4;
            ushort4 bits=*(device const ushort4*)((device const ushort*)w+d);
            if (raw) {
                *(device uint2*)(data+ptr+r*64+lane*2)=as_type<uint2>(bits);
            } else {
                ushort4 delta=((bits>>7)&255)-ushort(base[r]);
                lo[r]=as_type<uint>(uchar4((bits<<1)|(bits>>15)));
                hi[r]=delta.x|(delta.y<<4)|(delta.z<<8)|(delta.w<<12);
            }
        }
        if (!raw) {
            *(device uint4*)(data+ptr+lane*4)=lo;
            *(device ushort4*)(data+ptr+128+lane*2)=hi;
        }
    """,
)
_mixed_dot = mx.fast.metal_kernel(
    name="inkling_uniform_dense_tiles",input_names=["x","data","headers","bases"],output_names=["out"],
    source="""
        uint lane=thread_index_in_simdgroup,group=thread_position_in_grid.x/32;
        uint2 meta=*(device const uint2*)(headers+group*2);
        uchar4 base=*(device const uchar4*)(bases+group*4);
        uint ptr=meta.x,mask=meta.y;
        float sums[4]={};
        for (uint block=0;block<18;++block) {
            float4 v=float4(*(device const bfloat4*)(x+block*128+lane*4));
            if (mask&1) {
                for (uint r=0;r<4;++r) {
                    float4 w=float4(*(device const bfloat4*)(data+ptr+r*64+lane*2));
                    for (uint j=0;j<4;++j) sums[r]=fma(v[j],w[j],sums[r]);
                }
                ptr+=256;
            } else {
                uint4 low=*(device const uint4*)(data+ptr+lane*4);
                ushort4 high=*(device const ushort4*)(data+ptr+128+lane*2);
                for (uint r=0;r<4;++r) {
                    ushort h=high[r];
                    ushort4 ex=ushort4(h,h>>4,h>>8,h>>12)&15;
                    ushort4 rot=((ex+ushort(base[r]))<<8)|ushort4(as_type<uchar4>(low[r]));
                    float4 w=float4(as_type<bfloat4>((rot>>1)|(rot<<15)));
                    for (uint j=0;j<4;++j) sums[r]=fma(v[j],w[j],sums[r]);
                }
                ptr+=192;
            }
            mask>>=1;
        }
        for (uint r=0;r<4;++r) {
            float dot=sums[r];
            for (ushort step=16;step;step>>=1) dot+=simd_shuffle_down(dot,step);
            if (!lane) out[group*4+r]=bfloat16_t(dot);
        }
    """,
)


def encode_mixed(weight):
    rows=weight.shape[0]
    groups=rows//4
    bases=_ranges(inputs=[weight],grid=(rows*32,1,1),threadgroup=(128,1,1),
                  output_shapes=[(rows,)],output_dtypes=[mx.uint8])[0]
    masks,counts=_classify(inputs=[weight,bases],grid=(groups*32,1,1),threadgroup=(128,1,1),
                           output_shapes=[(groups,)]*2,output_dtypes=[mx.uint32]*2)
    ends=mx.cumsum(counts)
    words=groups*18*192+int(ends[-1].item())*64
    data,headers=_mixed_pack(inputs=[weight,bases,masks,ends],grid=(groups*18*32,1,1),threadgroup=(128,1,1),
                             output_shapes=[(words,),(groups,2)],output_dtypes=[mx.uint32]*2)
    return data,headers,bases


def project_mixed(x,weight,encoded):
    rows=weight.shape[0]
    return _mixed_dot(inputs=[x,*encoded],grid=(rows*8,1,1),threadgroup=(128,1,1),
                      output_shapes=[(1,1,rows)],output_dtypes=[x.dtype])[0]


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
