"""BF16 prompt projections using the canonical row-major matrix directly."""
from functools import lru_cache, partial

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.inkling2 import layers

from . import prefill_qmm


_SOURCE = """
    constexpr uint nt=32*WM*WN;
    uint tid=thread_position_in_threadgroup.x;
    uint bx=threadgroup_position_in_grid.x/SW;
    uint by=threadgroup_position_in_grid.y*SW+threadgroup_position_in_grid.x%SW;
    uint row=BM*by,col=BN*bx;
    uint lane=thread_index_in_simdgroup;
    uint sn=((lane/4)&2)*2+(lane%2)*2;
    mlx::steel::BlockMMA<bfloat16_t,bfloat16_t,BM,BN,BK,WM,WN,false,true,BK+8,BK+8>
        mma(simdgroup_index_in_threadgroup,lane);
    threadgroup bfloat16_t weights[BN*(BK+8)];
    for (uint k=0;k<K;k+=BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint a=tid*8;a<BN*BK;a+=nt*8) {
            uint n=a/BK,d=a%BK;
            *(threadgroup uint4*)(weights+n*(BK+8)+d)=*(device const uint4*)(w+(col+n)*K+k+d);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (short kk=0;kk<BK;kk+=8) {
            for (short i=0;i<BM/(8*WM);++i)
                mma.Atile.frag_at(i,0)=float2(*(device const bfloat2*)(x+(row+mma.sm+i*8*WM)*K+k+kk+sn));
            mma.Btile.template load<bfloat16_t,1,WN,1,BK+8>(weights+mma.Bs_offset+kk);
            mlx::steel::tile_matmad(mma.Ctile,mma.Atile,mma.Btile,mma.Ctile);
        }
    }
    mma.store_result(y+row*N+col,N);
"""

@lru_cache(maxsize=1)
def kernel():
    header = prefill_qmm.matrix_header()
    if header is None:
        return None
    return mx.fast.metal_kernel(name="inkling_dense_checkpoint_gemm", input_names=["x","w"],
                                output_names=["y"], header=header, source=_SOURCE)


def linear(layer, x):
    if (type(layer) is not nn.Linear or "bias" in layer
            or x.ndim != 3 or x.shape[0] != 1 or x.shape[1] != 2048
            or x.dtype != mx.bfloat16 or layer.weight.dtype != mx.bfloat16):
        return layer(x)
    n, k = layer.weight.shape
    geometry = _GEOMETRY.get((k, n))
    if geometry is None or kernel() is None:
        return layer(x)
    bm,bn,bk,wm,wn,sw = geometry
    nt = wm*wn*32
    l = x.shape[1]
    params = list(zip(("BM","BN","BK","WM","WN","SW","K","N"), (*geometry,k,n)))
    return kernel()(inputs=[x,layer.weight],template=params,
                    grid=(n//bn*nt*sw,l//bm//sw,1),threadgroup=(nt,1,1),
                    output_shapes=[(1,l,n)],output_dtypes=[x.dtype])[0]


_GEOMETRY = {
    (2304,3072): (64,32,64,2,2,1),
    (2304,1024): (64,64,64,1,4,4),
    (3072,2304): (64,32,64,2,2,4),
}


class PromptAttention:
    """Reuse the model's attention math with three eligible linear shapes specialized."""

    _attend = layers.Attention._attend

    def __init__(self, layer):
        self.layer = layer
        self.q_proj = partial(linear, layer.q_proj)
        self.k_proj = partial(linear, layer.k_proj)
        self.v_proj = partial(linear, layer.v_proj)
        self.gate_proj = partial(linear, layer.gate_proj)
        self.o_proj = partial(linear, layer.o_proj)

    def __getattr__(self, name):
        return getattr(self.layer, name)
