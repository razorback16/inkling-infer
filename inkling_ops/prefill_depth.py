"""Reuse each prompt depth source across its score and weighted-value passes."""
import mlx.core as mx
from mlx_lm.kernels.common import row_threads, scalar
from mlx_lm.models.inkling2 import layers

_reference=layers.depth_attention
_depth=mx.fast.metal_kernel(
    name="inkling_prompt_depth_values",input_names=["blocks","partial","queries","scale"],output_names=["out"],
    source="""
        constexpr uint N=K+1;
        uint row=threadgroup_position_in_grid.x, rows=threadgroups_per_grid.x;
        uint tid=thread_position_in_threadgroup.x, lane=thread_index_in_simdgroup, sg=simdgroup_index_in_threadgroup;
        float states[D/TG][N], dots[N*Q]={};
        for (uint n=0;n<D/TG;++n) {
            uint d=n*TG+tid;
            for (uint i=0;i<K;++i) states[n][i]=blocks[(i*rows+row)*D+d];
            states[n][K]=partial[row*D+d];
            for (uint j=0;j<Q;++j) {
                float q=queries[j*D+d];
                for (uint i=0;i<N;++i) dots[i*Q+j]+=states[n][i]*q;
            }
        }
        threadgroup float parts[N*Q][TG/32], factors[N*Q];
        for (uint i=0;i<N*Q;++i) {
            float sum=simd_sum(dots[i]);
            if (!lane) parts[i][sg]=sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid<Q) {
            float w[N],m=-INFINITY,z=0;
            for (uint i=0;i<N;++i) {
                float sum=0;
                for (uint s=0;s<TG/32;++s) sum+=parts[i*Q+tid][s];
                w[i]=sum*scale;
                m=max(m,w[i]);
            }
            for (uint i=0;i<N;++i) { w[i]=metal::exp(w[i]-m); z+=w[i]; }
            for (uint i=0;i<N;++i) factors[i*Q+tid]=w[i]/z;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint n=0;n<D/TG;++n)
            for (uint j=0;j<Q;++j) {
                float sum=0;
                for (uint i=0;i<N;++i) sum+=factors[i*Q+j]*states[n][i];
                out[(row*Q+j)*D+n*TG+tid]=sum;
            }
    """,
)


def evaluate(blocks,partial,queries,*,scale):
    if (partial.ndim!=3 or partial.shape[:1]!=(1,) or partial.shape[1]<8 or partial.shape[-1]!=2304
            or blocks.shape[0]>8 or queries.shape[0]>4 or blocks.dtype!=partial.dtype
            or partial.dtype!=mx.float32 or queries.dtype!=mx.float32):
        return _reference(blocks,partial,queries,scale=scale)
    d=partial.shape[-1]
    q=queries.shape[0]
    threads=row_threads(d)
    if d%threads:
        return _reference(blocks,partial,queries,scale=scale)
    return _depth(inputs=[blocks,partial,queries,scalar(scale)],
                  template=[("K",blocks.shape[0]),("Q",q),("D",d),("TG",threads)],
                  grid=(partial.shape[1]*threads,1,1),threadgroup=(threads,1,1),
                  output_shapes=[(*partial.shape[:-1],q,d)],output_dtypes=[mx.float32])[0]
