"""Tiled single-query attention with an independent softmax accumulator per tile."""
import mlx.core as mx
from mlx_lm.kernels.common import scalar

_partial = mx.fast.metal_kernel(
    name="inkling_partial_attention", input_names=["q", "k", "v", "mask", "scale"], output_names=["out", "stats"],
    source="""
        const uint lane = thread_index_in_simdgroup;
        const uint tile = thread_position_in_grid.x / 32;
        const uint head = threadgroup_position_in_grid.y;
        const uint bank = head / GQA;
        const uint n = threads_per_grid.x / 32;
        uint d = lane * 4;
        float4 query;
        for (uint j = 0; j < 4; ++j) query[j] = float(q[head*128+d+j]) * scale;
        float peak = -1.e30f, total = 0;
        float4 acc = 0;
        for (uint pos = tile*TILE; pos < min(uint(S), (tile+1)*TILE); ++pos) {
            float bias = float(mask[head*S+pos]);
            if (bias == -INFINITY) continue;
            uint base = (bank*S+pos)*128 + d;
            float dot = 0;
            for (uint j = 0; j < 4; ++j) dot += query[j] * float(k[base+j]);
            float score = simd_sum(dot) + bias;
            float top = max(peak,score);
            float a = fast::exp(peak-top), b = fast::exp(score-top);
            for (uint j = 0; j < 4; ++j) acc[j] = acc[j]*a + float(v[base+j])*b;
            total = total*a + b;
            peak = top;
        }
        uint index = head * n + tile;
        for (uint j = 0; j < 4; ++j) out[index*128+d+j] = acc[j];
        if (!lane) {
            stats[index*2] = peak;
            stats[index*2+1] = total;
        }
    """,
)
_finish = mx.fast.metal_kernel(
    name="inkling_merge_attention", input_names=["parts", "stats", "sinks"], output_names=["out"],
    source="""
        uint d = thread_position_in_grid.x;
        uint head = thread_position_in_grid.y;
        float peak = float(sinks[head]), total = 1.f, acc = 0;
        for (uint t = 0; t < N; ++t) {
            uint i = head*N+t;
            float top = max(peak, stats[2*i]);
            float a = fast::exp(peak-top), b = fast::exp(stats[2*i]-top);
            total = total*a + stats[2*i+1]*b;
            acc = acc*a + parts[i*128+d]*b;
            peak = top;
        }
        out[head*128+d] = T(acc/total);
    """,
)


def attend(q, k, v, mask, sinks, scale):
    heads = q.shape[1]
    length = k.shape[2]
    tile = 16
    count = (length+tile-1)//tile
    parts, stats = _partial(inputs=[q,k,v,mask,scalar(scale)],
                            template=[("TILE",tile),("GQA",heads//k.shape[1]),("S",length)],
                            grid=(count*32,heads,1), threadgroup=(32,1,1),
                            output_shapes=[(heads,count,128),(heads,count,2)], output_dtypes=[mx.float32]*2)
    return _finish(inputs=[parts,stats,sinks.astype(q.dtype)], template=[("T",q.dtype),("N",count)],
                   grid=(128,heads,1), threadgroup=(128,1,1),
                   output_shapes=[(1,1,heads,128)], output_dtypes=[q.dtype])[0]
