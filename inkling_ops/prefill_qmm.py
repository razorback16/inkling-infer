"""Grouped NVFP4 prompt GEMM with the checkpoint's BF16 coefficient values."""
from functools import lru_cache
from pathlib import Path
import re

import mlx.core as mx

# Offsets and tile descriptors are shared by the gate and down projections.
_offsets = mx.fast.metal_kernel(
    name="inkling_prompt_expert_offsets", input_names=["ids"], output_names=["offsets"],
    source="""
        uint expert=thread_position_in_grid.x,lo=0,hi=M;
        while (lo<hi) {
            uint mid=(lo+hi)/2;
            if (ids[mid]<expert) lo=mid+1; else hi=mid;
        }
        offsets[expert]=lo;
    """,
)

_schedule = mx.fast.metal_kernel(
    name="inkling_prompt_expert_tiles",input_names=["offsets","ends"],output_names=["tiles"],
    source="""
        uint tile=thread_position_in_grid.x;
        uint lo=0,hi=128;
        while (lo<hi) {
            uint mid=(lo+hi)/2;
            if (ends[mid]<=tile) lo=mid+1; else hi=mid;
        }
        uint row=0,count=0;
        if (lo<128) {
            uint first=lo ? ends[lo-1] : 0;
            row=offsets[lo]+32*(tile-first);
            count=min(32u,offsets[lo+1]-row);
        }
        tiles[3*tile]=row; tiles[3*tile+1]=lo; tiles[3*tile+2]=count;
    """,
)


def schedule(ids):
    m=ids.size
    offsets,=_offsets(inputs=[ids],template=[("M",m)],grid=(129,1,1),threadgroup=(32,1,1),
                      output_shapes=[(129,)],output_dtypes=[mx.uint32])
    ends=mx.cumsum((offsets[1:]-offsets[:-1]+31)//32)
    size=(m+31)//32+128
    return _schedule(inputs=[offsets,ends],grid=(size,1,1),threadgroup=(64,1,1),
                     output_shapes=[(size,3)],output_dtypes=[mx.uint32])[0]


@lru_cache(maxsize=1)
def kernel():
    # Use the installed MLX matrix accumulator to preserve its floating-point order.
    root=Path(mx.__file__).parent/"include"
    mma="mlx/backend/metal/kernels/steel/gemm/mma.h"
    if not (root/mma).is_file():
        return None
    seen=set()
    def include(name):
        if name in seen:
            return ""
        seen.add(name)
        contents=(root/name).read_text().replace("#pragma once","")
        return re.sub(r'^#include "([^"]+)"',lambda m:include(m[1]),contents,flags=re.M)
    return mx.fast.metal_kernel(
        name="inkling_prompt_lookup_gemm",input_names=["x","w","scales","table","tiles","activation","token_rows","original_slots"],
        output_names=["out"],header=include(mma),source="""
            uint tile=threadgroup_position_in_grid.y, row=tiles[3*tile], expert=tiles[3*tile+1];
            uint count=tiles[3*tile+2];
            if (!count) return;
            uint column=threadgroup_position_in_grid.x*64, tid=thread_position_in_threadgroup.x;
            uint xr=tid/4, xc=(tid%4)*8, wr=tid/2, wc=(tid%2)*16;
            uint input_row=row+xr;
            if (GATHER && xr<count) input_row=token_rows[input_row];
            uint weight_row=expert*N+column+wr;
            threadgroup bfloat16_t Xs[32*40], Ws[64*40];
            mlx::steel::BlockMMA<bfloat16_t,bfloat16_t,32,64,32,2,2,false,true,40,40>
                mma(simdgroup_index_in_threadgroup,thread_index_in_simdgroup);
            for (uint k=0;k<K;k+=32) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (ushort j=0;j<8;j+=4) {
                    bfloat4 v=0;
                    if (xr<count) v=*(device const bfloat4*)(x+input_row*K+k+xc+j);
                    *(threadgroup bfloat4*)(Xs+xr*40+xc+j)=v;
                }
                uint scale=scales[weight_row*(K/16)+(k+wc)/16];
                auto values=table+(expert*256+scale)*16;
                uint2 codes=*(device const uint2*)(w+weight_row*(K/8)+(k+wc)/8);
                for (ushort j=0;j<16;++j)
                    Ws[wr*40+wc+j]=values[(codes[j/8]>>((j%8)*4))&15];
                threadgroup_barrier(mem_flags::mem_threadgroup);
                mma.mma(Xs,Ws);
            }
            if (GATED) {
                for (short i=0;i<2;++i) {
                    uint r=mma.sm+i*16;
                    if (r>=count) continue;
                    for (short j=0;j<4;++j) {
                        float2 frag=mma.Ctile.frag_at(i,j);
                        ushort g=as_type<ushort>(bfloat16_t(frag.x));
                        ushort u=as_type<ushort>(bfloat16_t(frag.y));
                        uint gi=((uint(g)&32767)<<2)|((g>>15)<<1);
                        uint ui=((uint(u)&32767)<<2)|((u>>15)<<1);
                        uint c=column+mma.sn+j*16;
                        out[(row+r)*(N/2)+c/2]=bfloat16_t(activation[gi]*activation[ui+1]);
                    }
                }
            } else if (ROUTE_LAYOUT) {
                for (short i=0;i<2;++i) {
                    uint r=mma.sm+i*16;
                    if (r>=count) continue;
                    uint original=original_slots[row+r], token=original/8, slot=original%8;
                    uint base=((token*(N/64)+column/64)*8+slot)*64;
                    for (short j=0;j<4;++j)
                        *(device bfloat2*)(out+base+mma.sn+j*16)=bfloat2(mma.Ctile.frag_at(i,j));
                }
            } else {
                auto result=out+row*N+column;
                if (count==32) mma.store_result(result,N);
                else mma.store_result_safe(result,N,short2(64,count));
            }
        """,
    )


def applicable(proj):
    return (getattr(proj,"mode",None)=="nvfp4" and proj.bits==4 and proj.group_size==16
            and "bias" not in proj and proj.global_scale is not None and proj.global_scale.shape==(128,)
            and proj.weight.dtype==mx.uint32 and proj.scales.dtype==mx.uint8
            and proj.weight.shape[0]==128 and proj.input_dims%32==0 and proj.output_dims%64==0)


def project(proj, x, tiles, activation=None, token_rows=None, original_slots=None):
    scale=proj.global_scale
    saved=getattr(proj,"_prompt_coefficients",None)
    if saved is None or saved[0] is not scale:
        codes=mx.array([0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6],mx.float32)
        fp8=mx.from_fp8(mx.arange(256).astype(mx.uint8),mx.float32)
        table=((scale[:,None,None]/2688*fp8[None,:,None])*codes).astype(mx.bfloat16)
        mx.eval(table)
        saved=(scale,table)
        object.__setattr__(proj,"_prompt_coefficients",saved)
    n=proj.output_dims
    width=n//2 if activation is not None else n
    m=x.shape[0] if token_rows is None else token_rows.size
    return kernel()(inputs=[x,proj.weight,proj.scales,saved[1],tiles,x if activation is None else activation,
                            tiles if token_rows is None else token_rows,tiles if original_slots is None else original_slots],
                    template=[("K",proj.input_dims),("N",n),("GATED",activation is not None),("GATHER",token_rows is not None),("ROUTE_LAYOUT",original_slots is not None)],
                    grid=(n//64*128,tiles.shape[0],1),threadgroup=(128,1,1),
                    output_shapes=[(m,1,width)],output_dtypes=[x.dtype])[0]
