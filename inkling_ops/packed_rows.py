"""Lossless four-row BF16 tiles for 2304-channel decode projections."""
import mlx.core as mx

from . import vocab_tiles


_pack = mx.fast.metal_kernel(
    name="inkling_encode_query_core_and_overflow",
    input_names=["weight","records"], output_names=["payload","overflow"],
    source="""
        uint lane=thread_index_in_simdgroup, tile=thread_position_in_grid.x/32;
        uint group=tile/18, block=tile%18;
        uint4 record=((device const uint4*)records)[group];
        uchar4 bases=as_type<uchar4>(record.x);
        uint4 low;
        ushort4 mid, upper;
        for (uint r=0;r<4;++r) {
            uint i=(group*4+r)*2304+block*128+lane*4;
            ushort4 bits=*(device const ushort4*)((device const ushort*)weight+i);
            ushort4 e=(((bits>>7)&255)-ushort(bases[r]))&255;
            ushort4 n=e&15;
            low[r]=as_type<uint>(uchar4((bits<<1)|(bits>>15)));
            mid[r]=n.x|(n.y<<4)|(n.z<<8)|(n.w<<12);
            n=e>>4;
            upper[r]=n.x|(n.y<<4)|(n.z<<8)|(n.w<<12);
        }
        uint physical=((group/4)*18+block)*4+group%4;
        *(device uint4*)(payload+physical*192+lane*4)=low;
        *(device ushort4*)(payload+physical*192+128+lane*2)=mid;
        if (record.y&(1u<<block)) {
            uint index=record.z+popcount(record.y&((1u<<block)-1));
            ((device ushort4*)overflow)[index*32+lane]=upper;
        }
    """,
)
_dot = mx.fast.metal_kernel(
    name="inkling_four_row_gemv", input_names=["x","payload","overflow"],output_names=["out"],
    source="""
        uint lane=thread_index_in_simdgroup, group=thread_position_in_grid.x/32;
        auto stripe=payload+(group/4)*13888;
        uint4 meta=((device const uint4*)stripe)[group%4];
        uchar4 base=as_type<uchar4>(meta.x);
        uint raw_tiles=meta.y;
        float sums[4]={};
        for (uint block=0;block<18;++block) {
            auto data=stripe+64+(block*4+group%4)*192;
            uint d=block*128+lane*4;
            float4 v=float4(*(device const bfloat4*)(x+d));
            uint4 low=*(device const uint4*)(data+lane*4);
            ushort4 high=*(device const ushort4*)(data+128+lane*2);
            ushort4 upper=0;
            if (raw_tiles&(1u<<block)) {
                uint index=meta.z+popcount(raw_tiles&((1u<<block)-1));
                upper=((device const ushort4*)overflow)[index*32+lane];
            }
            for (uint r=0;r<4;++r) {
                ushort h=high[r];
                ushort4 ex=ushort4(h,h>>4,h>>8,h>>12)&15;
                ushort u=upper[r];
                ex|=(ushort4(u,u>>4,u>>8,u>>12)&15)<<4;
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
    rows=weight.shape[0]
    formats,_=vocab_tiles._formats(inputs=[weight],grid=(rows*32,1,1),threadgroup=(128,1,1),
                                   output_shapes=[(rows,)]*2,output_dtypes=[mx.uint32]*2)
    bases=(formats>>24).astype(mx.uint8).view(mx.uint32)
    row_masks=(formats&0x3ffff).reshape(-1,4)
    masks=row_masks[:,0]|row_masks[:,1]|row_masks[:,2]|row_masks[:,3]
    flags=(masks[:,None] >> mx.arange(18,dtype=mx.uint32)) & 1
    counts=mx.sum(flags,axis=1)
    ends=mx.cumsum(counts)
    n_overflow=max(1,int(ends[-1].item()))
    records=mx.stack((bases,masks,ends-counts,mx.zeros_like(masks)),axis=1)
    payload,overflow=_pack(inputs=[weight,records],grid=(weight.size//16,1,1),threadgroup=(128,1,1),
                           output_shapes=[(rows//16,13824),(n_overflow,64)],output_dtypes=[mx.uint32]*2)
    headers=mx.pad(records.reshape(-1,16),[(0,0),(0,48)])
    return mx.concatenate((headers,payload),axis=1),overflow


def project(x,weight,encoded):
    rows=weight.shape[0]
    return _dot(inputs=[x,*encoded],grid=(rows*8,1,1),threadgroup=(128,1,1),
                output_shapes=[(1,1,rows)],output_dtypes=[x.dtype])[0]
