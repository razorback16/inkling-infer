"""Losslessly encode frequent four-exponent patterns with twelve-bit dictionary keys."""
import mlx.core as mx
import numpy as np


def _dictionary(w):
    rows,cols=w.shape
    sample=np.asarray(w[::max(1,rows//256)].view(mx.uint16)).reshape(-1,4)
    exps=((sample>>7)&255).astype(np.int32)-108
    valid=exps[(exps.min(1)>=0)&(exps.max(1)<16)]
    packed=valid[:,0]|(valid[:,1]<<4)|(valid[:,2]<<8)|(valid[:,3]<<12)
    counts=np.bincount(packed,minlength=65536)
    words=np.argsort(-counts,kind="stable")[:4095].astype(np.uint16)
    codes=np.full(65536,4095,np.uint16)
    codes[words]=np.arange(4095,dtype=np.uint16)
    exponents=np.stack([((words>>n)&15)+108 for n in (0,4,8,12)],axis=1)
    return mx.array(codes),mx.array((exponents<<7).astype(np.uint16))


_find = """
    uint offset=row*2304+block*128+lane*4;
    ushort4 bits=*(device const ushort4*)((device const ushort*)w+offset);
    ushort4 e=((bits>>7)&255)-108;
    uint key=any(e>15) ? 4095 : codes[e.x|(e.y<<4)|(e.z<<8)|(e.w<<12)];
"""
_count=mx.fast.metal_kernel(
    name="inkling_quad_code_counts",input_names=["w","codes"],output_names=["counts"],
    source="""
        uint lane=thread_index_in_simdgroup,row=thread_position_in_grid.x/32;
        uint count=0;
        for (uint block=0;block<18;++block) {
            FIND
            count+=uint(key==4095);
        }
        count=simd_sum(count);
        if (!lane) counts[row]=count;
    """.replace("FIND",_find),
)
_pack=mx.fast.metal_kernel(
    name="inkling_quad_code_pack",input_names=["w","codes","ends"],output_names=["body","keys","exceptions"],
    source="""
        uint lane=thread_index_in_simdgroup,row=thread_position_in_grid.x/32;
        uint next=row ? ends[row-1] : 0;
        for (uint block=0;block<18;++block) {
            FIND
            uint bad=uint(key==4095);
            uint slot=next+simd_prefix_exclusive_sum(bad);
            next+=simd_sum(bad);
            uint low=as_type<uint>(uchar4((bits&127)|((bits>>8)&128)));
            if (bad) { *(device ushort4*)(exceptions+slot*4)=bits; low=slot; }
            uint tile=row/4*18+block;
            body[tile*128+lane*4+row%4]=low;
            keys[tile*128+lane*4+row%4]=ushort(key);
        }
    """.replace("FIND",_find),
)
_indices=mx.fast.metal_kernel(
    name="inkling_quad_code_indices",input_names=["keys"],output_names=["packed"],
    source="""
        uint lane=thread_index_in_simdgroup,tile=thread_position_in_grid.x/32;
        ushort4 v=*(device const ushort4*)(keys+tile*128+lane*4);
        packed[tile*48+lane]=as_type<uint>(uchar4(v));
        ushort4 h=v>>8;
        ((device ushort*)(packed+tile*48+32))[lane]=h.x|(h.y<<4)|(h.z<<8)|(h.w<<12);
    """,
)
_dot=mx.fast.metal_kernel(
    name="inkling_quad_code_gemv",input_names=["x","body","indices","lut","exceptions"],output_names=["out"],
    source="""
        uint lane=thread_index_in_simdgroup,group=thread_position_in_grid.x/32;
        float accum[4]={};
        for (uint block=0;block<18;++block) {
            uint tile=group*18+block;
            float4 v=float4(*(device const bfloat4*)(x+block*128+lane*4));
            uint4 low=*(device const uint4*)(body+tile*128+lane*4);
            ushort4 bottom=ushort4(as_type<uchar4>(indices[tile*48+lane]));
            ushort h=((device const ushort*)(indices+tile*48+32))[lane];
            ushort4 ids=bottom|((ushort4(h,h>>4,h>>8,h>>12)&15)<<8);
            for (uint r=0;r<4;++r) {
                ushort4 bits;
                if (ids[r]==4095) {
                    bits=*(device const ushort4*)(exceptions+low[r]*4);
                } else {
                    ushort4 exp=*(device const ushort4*)(lut+uint(ids[r])*4);
                    short4 m=short4(as_type<char4>(low[r]));
                    bits=exp|(as_type<ushort4>(m)&ushort(0x807f));
                }
                float4 w=float4(as_type<bfloat4>(bits));
                for (uint j=0;j<4;++j) accum[r]=fma(v[j],w[j],accum[r]);
            }
        }
        for (uint r=0;r<4;++r) {
            float dot=accum[r];
            for (ushort n=16;n;n>>=1) dot+=simd_shuffle_down(dot,n);
            if (!lane) out[group*4+r]=bfloat16_t(dot);
        }
    """,
)


def encode(w):
    n=w.shape[0]
    codes,lut=_dictionary(w)
    pattern=dict(grid=(n*32,1,1),threadgroup=(128,1,1))
    counts=_count(inputs=[w,codes],**pattern,output_shapes=[(n,)],output_dtypes=[mx.uint32])[0]
    ends=mx.cumsum(counts)
    count=max(1,int(ends[-1].item()))
    tiles=w.size//512
    body,keys,exceptions=_pack(inputs=[w,codes,ends],**pattern,output_shapes=[(tiles,128),(tiles,128),(count,4)],
                               output_dtypes=[mx.uint32,mx.uint16,mx.uint16])
    indices=_indices(inputs=[keys],grid=(tiles*32,1,1),threadgroup=(128,1,1),
                     output_shapes=[(tiles,48)],output_dtypes=[mx.uint32])[0]
    return body,indices,lut,exceptions


def project(x,w,encoded):
    n=w.shape[0]
    return _dot(inputs=[x,*encoded],grid=(n*8,1,1),threadgroup=(128,1,1),
                output_shapes=[(1,1,n)],output_dtypes=[x.dtype])[0]
