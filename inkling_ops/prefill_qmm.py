"""Grouped prompt GEMM consuming the original packed checkpoint weights and scales."""
from functools import lru_cache
from pathlib import Path
import re

import mlx.core as mx

ROWS_PER_TILE = 32

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
            row=offsets[lo]+BM*(tile-first);
            count=min(uint(BM),offsets[lo+1]-row);
        }
        tiles[3*tile]=row; tiles[3*tile+1]=lo; tiles[3*tile+2]=count;
    """,
)


def schedule(ids):
    bm = ROWS_PER_TILE
    m=ids.size
    offsets,=_offsets(inputs=[ids],template=[("M",m)],grid=(129,1,1),threadgroup=(32,1,1),
                      output_shapes=[(129,)],output_dtypes=[mx.uint32])
    ends=mx.cumsum((offsets[1:]-offsets[:-1]+bm-1)//bm)
    size=(m+bm-1)//bm+128
    return _schedule(inputs=[offsets,ends],template=[("BM",bm)],grid=(size,1,1),threadgroup=(64,1,1),
                     output_shapes=[(size,3)],output_dtypes=[mx.uint32])[0]


@lru_cache(maxsize=1)
def matrix_header():
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
    return include(mma)


@lru_cache(maxsize=1)
def kernel():
    header = matrix_header()
    if header is None:
        return None
    return mx.fast.metal_kernel(
        name="inkling_prompt_lookup_gemm",input_names=["x","w","scales","table","global_scale","tiles","activation","token_rows","original_slots"],
        output_names=["out"],header=header+"\nconstant float fp4[16]={0,.5,1,1.5,2,3,4,6,-0.f,-.5,-1,-1.5,-2,-3,-4,-6};\n",source="""
            uint tile=threadgroup_position_in_grid.y*SWIZZLE+threadgroup_position_in_grid.x%SWIZZLE;
            if (tile>=TILES) return;
            uint row=tiles[3*tile], expert=tiles[3*tile+1];
            uint count=tiles[3*tile+2];
            if (!count) return;
            uint column=(threadgroup_position_in_grid.x/SWIZZLE)*BN, tid=thread_position_in_threadgroup.x;
            constexpr short stride=BK;
            constexpr uint threads=WM*WN*32;
            // The workgroup decodes just this panel. All persistent inputs are
            // the checkpoint words, scales, or checkpoint-independent codebooks.
            threadgroup bfloat16_t Ws[BN*stride];
            mlx::steel::BlockMMA<bfloat16_t,bfloat16_t,BM,BN,BK,WM,WN,false,true,stride,stride>
                mma(simdgroup_index_in_threadgroup,thread_index_in_simdgroup);
            uint lane=thread_index_in_simdgroup;
            uint fn=((lane/4)&2)*2+(lane%2)*2;
            uint input_rows[BM/(WM*8)];
            for (short i=0;i<BM/(WM*8);++i) {
                uint at=row+min(uint(mma.sm)+i*(8*WM),count-1);
                input_rows[i]=GATHER ? token_rows[at] : at;
            }
            for (uint k=0;k<K;k+=BK) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint v=tid*16;v<BN*BK;v+=threads*16) {
                    uint wr=v/BK,wc=v%BK;
                    uint weight_row=expert*N+column+wr;
                    if (MX8) {
                        uint scale=scales[weight_row*(K/32)+(k+wc)/32];
                        uint4 words=*(device const uint4*)(w+weight_row*(K/4)+(k+wc)/4);
                        for (ushort j=0;j<16;++j)
                            Ws[wr*stride+wc+j]=bfloat16_t(table[(scale<<8)+((words[j/4]>>((j%4)*8))&255)]);
                    } else {
                        uint scale=scales[weight_row*(K/16)+(k+wc)/16];
                        float factor=(global_scale[expert]/2688.f)*table[scale];
                        uint2 words=*(device const uint2*)(w+weight_row*(K/8)+(k+wc)/8);
                        for (ushort j=0;j<16;++j)
                            Ws[wr*stride+wc+j]=bfloat16_t(factor*fp4[(words[j/8]>>((j%8)*4))&15]);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (short kk=0;kk<BK;kk+=8) {
                    for (short i=0;i<BM/(8*WM);++i)
                        mma.Atile.frag_at(i,0)=float2(*(device const bfloat2*)(x+input_rows[i]*K+k+kk+fn));
                    mma.Btile.template load<bfloat16_t,1,WN,1,stride>(Ws+mma.Bs_offset+kk);
                    mlx::steel::tile_matmad(mma.Ctile,mma.Atile,mma.Btile,mma.Ctile);
                }
            }
            if (GATED) {
                for (short i=0;i<BM/(WM*8);++i) {
                    uint r=mma.sm+i*(8*WM);
                    if (r>=count) continue;
                    for (short j=0;j<BN/(8*WN);++j) {
                        float2 frag=mma.Ctile.frag_at(i,j);
                        ushort g=as_type<ushort>(bfloat16_t(frag.x));
                        ushort u=as_type<ushort>(bfloat16_t(frag.y));
                        uint gi=((uint(g)&32767)<<2)|((g>>15)<<1);
                        uint ui=((uint(u)&32767)<<2)|((u>>15)<<1);
                        uint c=column+mma.sn+j*(8*WN);
                        out[(row+r)*(N/2)+c/2]=bfloat16_t(activation[gi]*activation[ui+1]);
                    }
                }
            } else if (ROUTE_LAYOUT) {
                for (short i=0;i<BM/(WM*8);++i) {
                    uint r=mma.sm+i*(8*WM);
                    if (r>=count) continue;
                    uint original=original_slots[row+r], token=original/8, slot=original%8;
                    for (short j=0;j<BN/(8*WN);++j) {
                        uint channel=column+mma.sn+j*(8*WN);
                        uint pos=((token*(N/64)+channel/64)*8+slot)*64+channel%64;
                        *(device bfloat2*)(out+pos)=bfloat2(mma.Ctile.frag_at(i,j));
                    }
                }
            } else {
                auto result=out+row*N+column;
                if (count==BM) mma.store_result(result,N);
                else mma.store_result_safe(result,N,short2(BN,count));
            }
        """,
    )


@lru_cache(maxsize=1)
def mx8_codebook():
    from .expert_gate import mx_codes
    codes=mx_codes()
    values=(codes[256:,None]*codes[None,:256]).astype(mx.bfloat16)
    mx.eval(values)
    return values


def applicable(proj):
    mode=getattr(proj,"mode",None)
    nv=(mode=="nvfp4" and proj.bits==4 and proj.group_size==16 and proj.global_scale is not None
        and proj.global_scale.shape==(128,))
    mx8=(mode=="mxfp8" and proj.bits==8 and proj.group_size==32 and proj.global_scale is None)
    return ((nv or mx8)
            and "bias" not in proj and proj.weight.dtype==mx.uint32 and proj.scales.dtype==mx.uint8
            and proj.weight.shape[0]==128 and proj.input_dims%64==0 and proj.output_dims%64==0)


def project(proj, x, tiles, activation=None, token_rows=None, original_slots=None):
    bm, bn, bk, wm, wn = ROWS_PER_TILE, 64, 64, 1, 2
    swizzle = 16
    threads=32*wm*wn
    from .expert_gate import mx_codes
    table=mx8_codebook() if proj.mode=="mxfp8" else mx_codes()
    scale=proj.global_scale if proj.global_scale is not None else table
    n=proj.output_dims
    width=n//2 if activation is not None else n
    m=x.shape[0] if token_rows is None else token_rows.size
    return kernel()(inputs=[x,proj.weight,proj.scales,table,scale,tiles,x if activation is None else activation,
                            tiles if token_rows is None else token_rows,tiles if original_slots is None else original_slots],
                    template=[("SWIZZLE",swizzle),("TILES",tiles.shape[0]),("BM",bm),("BN",bn),("BK",bk),("WM",wm),("WN",wn),("MX8",proj.mode=="mxfp8"),("K",proj.input_dims),("N",n),("GATED",activation is not None),("GATHER",token_rows is not None),("ROUTE_LAYOUT",original_slots is not None)],
                    grid=(n//bn*threads*swizzle,(tiles.shape[0]+swizzle-1)//swizzle,1),threadgroup=(threads,1,1),
                    output_shapes=[(m,1,width)],output_dtypes=[x.dtype])[0]
