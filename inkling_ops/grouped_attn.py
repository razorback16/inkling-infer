"""Two-pass batch-one attention with interleaved BF16 partials and fused gates."""
import mlx.core as mx
from mlx_lm.kernels.common import scalar
from mlx_lm.models.cache import BatchRotatingKVCache, RotatingKVCache

_partial = mx.fast.metal_kernel(
    name="inkling_interleaved_attention",
    input_names=["q","k","v","rel","w","slot","end","sinks","scale"],output_names=["partials","stats"],
    source="""
        uint lane = thread_index_in_simdgroup, head = threadgroup_position_in_grid.y;
        uint tile = thread_position_in_grid.x / 32;
        uint kv = head / GQA, d = lane*4;
        uint addr = (head*128+tile);
        float4 query;
        for (uint j=0; j<4; ++j) query[j] = scale*float(q[head*128+d+j]);
        float peak = tile ? -FLT_MAX : float(sinks[head]);
        float total = tile ? 0.f : 1.f;
        float4 accum = 0;
        for (uint chunk=0; chunk<S; chunk+=4096) {
            int pos = chunk + tile + lane*128;
            int dist = slot[0]-1-pos;
            if (dist < 0) dist += S;
            float bias = 0;
            if (pos >= S || dist >= end[0] || (WINDOW && dist >= WINDOW)) bias = -INFINITY;
            else if (dist < R) {
                for (uint j=0; j<WIDTH; ++j)
                    bias = fma(float(rel[head*WIDTH+j]), float(w[j*R+dist]), bias);
                bias = float(T(bias));
            }
            for (uint i=0; i<32 && chunk+tile+i*128<S; ++i) {
                float pb = simd_shuffle(bias,i);
                if (pb == -INFINITY) continue;
                uint p = ((kv*S+chunk+tile+i*128)*128)+d;
                float score = 0;
                for (uint j=0; j<4; ++j) score += query[j]*float(k[p+j]);
                score = simd_sum(score)+pb;
                float top = max(peak,score);
                float a = fast::exp(peak-top), b = fast::exp(score-top);
                total = total*a+b;
                for (uint j=0; j<4; ++j) accum[j] = accum[j]*a+b*float(v[p+j]);
                peak = top;
            }
        }
        for (uint j=0; j<4; ++j) partials[addr*128+d+j] = T(accum[j]);
        if (!lane) { stats[addr*2] = peak; stats[addr*2+1] = total; }
    """,
)
_merge = mx.fast.metal_kernel(
    name="inkling_attention_gate", input_names=["partial","stats","gate"], output_names=["out"],
    source="""
        uint lane = thread_index_in_simdgroup;
        uint head = threadgroup_position_in_grid.y;
        uint d = (thread_position_in_grid.x/32)*4;
        uint i = head*128 + lane;
        float peak = -FLT_MAX;
        for (uint n=0; n<128; n+=32) peak = max(peak,stats[(i+n)*2]);
        peak = simd_max(peak);
        float4 acc = 0;
        float z = 0;
        for (uint n=0; n<128; n+=32) {
            float a = fast::exp(stats[(i+n)*2] - peak);
            z += a*stats[(i+n)*2+1];
            for (uint j=0; j<4; ++j) acc[j] += a*float(partial[(i+n)*128+d+j]);
        }
        z = simd_sum(z);
        for (uint j=0; j<4; ++j) {
            float output = float(T(simd_sum(acc[j])/z));
            if (!lane) {
                uint col = head*128+d+j;
                float g = float(gate[col]);
                float denom = 1.f/(1.f+precise::exp(abs(g)));
                out[col] = T(output*(g<0 ? denom : 1.f-denom));
            }
        }
    """,
)


def attend(layer,q,k,v,rel,sinks,cache,gate):
    length, heads = k.shape[2], q.shape[1]
    slot = cache._idx if isinstance(cache,(RotatingKVCache,BatchRotatingKVCache)) else length
    w = layer.rel_logits_proj.proj.astype(rel.dtype)
    parts,stats = _partial(
        inputs=[q,k,v,rel,w,mx.array(slot,mx.int32).reshape(1),mx.array(cache.offset,mx.int32).reshape(1),
                sinks.astype(q.dtype),scalar(layer.scaling)],
        template=[("T",q.dtype),("S",length),("GQA",heads//k.shape[1]),("WIDTH",w.shape[0]),
                  ("R",w.shape[1]),("WINDOW",layer.sliding_window)],
        grid=(128*32,heads,1),threadgroup=(128,1,1),
        output_shapes=[(heads,128,128),(heads,128,2)],output_dtypes=[q.dtype,mx.float32])
    return _merge(inputs=[parts,stats,gate],template=[("T",q.dtype)],
                  grid=(32*32,heads,1),threadgroup=(128,1,1),
                  output_shapes=[gate.shape],output_dtypes=[q.dtype])[0]
