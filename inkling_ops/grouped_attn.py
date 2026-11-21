"""Two-pass batch-one attention with interleaved BF16 partials and fused gates."""
import mlx.core as mx
from mlx_lm.kernels.common import scalar
from mlx_lm.models.cache import BatchRotatingKVCache, RotatingKVCache

_partial = mx.fast.metal_kernel(
    name="inkling_interleaved_attention",
    input_names=["q","k","v","tail","fresh","rel","w","slot","end","sinks","scale"],output_names=["partials","stats"],
    source="""
        uint lane = thread_index_in_simdgroup, head = threadgroup_position_in_grid.y;
        int newest = slot[0] - (PRE_UPDATE ? 0 : 1);
        int valid = end[0] + (PRE_UPDATE ? 1 : 0);
        uint tile = thread_position_in_grid.x / 32;
        uint kv = head / GQA, d = lane*4;
        uint addr = (head*128+tile);
        float4 query;
        for (uint j=0; j<4; ++j) query[j] = scale*float(q[head*128+d+j]);
        float peak = tile ? -FLT_MAX : float(sinks[head]);
        float total = tile ? 0.f : 1.f;
        float4 accum = 0;
        if (S==1024 && WINDOW==1024 && valid>=1024) {
            float bias = 0;
            if (lane < 8) {
                uint dist = uint(newest-int(tile+128*lane)) & 1023u;
                if (dist<R) {
                    for (uint j=0; j<WIDTH; ++j)
                        bias = fma(float(rel[head*WIDTH+j]),float(w[SWIZZLED ? ((dist%128)*WIDTH+j)*(R/128)+dist/128 : j*R+dist]),bias);
                    bias = float(T(bias));
                }
            }
            float scores[8];
            #pragma clang loop unroll(full)
            for (uint pos=0; pos<8; ++pos) {
                uint at=((kv*1024+tile+pos*128)*128)+d;
                auto key_ptr = FRESH && uint(newest)==tile+pos*128 ? fresh+kv*128+d : k+at;
                float4 key=float4(*(device const bfloat4*)key_ptr);
                float score=0;
                for (uint j=0; j<4; ++j) score += query[j]*key[j];
                scores[pos]=simd_sum(score)+simd_shuffle(bias,pos);
            }
            #pragma clang loop unroll(full)
            for (uint pos=0; pos<8; ++pos) {
                uint at=((kv*1024+tile+pos*128)*128)+d;
                auto value_ptr = FRESH && uint(newest)==tile+pos*128 ? fresh+(NKV+kv)*128+d : v+at;
                float4 value=float4(*(device const bfloat4*)value_ptr);
                float score=scores[pos], top=max(peak,score);
                float a=fast::exp(peak-top), b=fast::exp(score-top);
                total=total*a+b;
                for (uint j=0; j<4; ++j) accum[j]=accum[j]*a+b*value[j];
                peak=top;
            }
        } else {
            for (uint chunk=0; chunk<S; chunk+=4096) {
                int pos = chunk + tile + lane*128;
                int dist = newest-pos;
                if (dist < 0) dist += S;
                float bias = 0;
                if (pos >= S || dist >= valid || (WINDOW && dist >= WINDOW)) bias = -INFINITY;
                else if (dist < R) {
                    for (uint j=0; j<WIDTH; ++j)
                        bias = fma(float(rel[head*WIDTH+j]), float(w[SWIZZLED ? ((dist%128)*WIDTH+j)*(R/128)+dist/128 : j*R+dist]), bias);
                    bias = float(T(bias));
                }
                for (uint i=0; i<32 && chunk+tile+i*128<S; ++i) {
                    float pb = simd_shuffle(bias,i);
                    if (pb == -INFINITY) continue;
                    uint position = chunk+tile+i*128;
                    bool in_tail = PAGED && position >= S-256;
                    uint pos = in_tail ? position-(S-256) : position;
                    uint length = in_tail ? 256 : (PAGED ? S-256 : S);
                    uint p = (kv*length+pos)*128+d;
                    auto keys = in_tail ? tail : k;
                    auto values = in_tail ? tail+NKV*256*128 : v;
                    float score = 0;
                    for (uint j=0; j<4; ++j) score += query[j]*float(keys[p+j]);
                    score = simd_sum(score)+pb;
                    float top = max(peak,score);
                    float a = fast::exp(peak-top), b = fast::exp(score-top);
                    total = total*a+b;
                    for (uint j=0; j<4; ++j) accum[j] = accum[j]*a+b*float(values[p+j]);
                    peak = top;
                }
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


_sliding_fused=mx.fast.metal_kernel(
    name="inkling_sliding_attention_one_dispatch",
    input_names=["q","k","v","fresh","rel","w","slot","sinks","scale","gate"],output_names=["out"],
    source="""
        uint head=threadgroup_position_in_grid.y,chunk=threadgroup_position_in_grid.x;
        uint lane=thread_index_in_simdgroup,warp=simdgroup_index_in_threadgroup;
        uint kv=head/GQA,newest=uint(slot[0]);
        float4 query=float4(*(device const bfloat4*)(q+head*128+lane*4))*scale;
        threadgroup bfloat16_t buffer[128][32];
        threadgroup float peaks[128], totals[128];
        for (uint tile=warp;tile<128;tile+=32) {
            float bias=0;
            if (lane<8) {
                uint dist=(newest-tile-128*lane)&1023;
                if (dist<R) {
                    for (uint j=0;j<WIDTH;++j)
                        bias=fma(float(rel[head*WIDTH+j]),float(w[SWIZZLED ? ((dist%128)*WIDTH+j)*(R/128)+dist/128 : j*R+dist]),bias);
                    bias=float(bfloat16_t(bias));
                }
            }
            float score[8];
            for (uint pos=0;pos<8;++pos) {
                uint at=tile+pos*128;
                auto keys=at==newest ? fresh+kv*128 : k+(kv*1024+at)*128;
                float4 key=float4(*(device const bfloat4*)(keys+lane*4));
                float dot=0;
                for (uint j=0;j<4;++j) dot+=query[j]*key[j];
                score[pos]=simd_sum(dot)+simd_shuffle(bias,pos);
            }
            float peak=tile ? -FLT_MAX : float(sinks[head]),total=tile ? 0.f : 1.f,acc=0;
            for (uint pos=0;pos<8;++pos) {
                uint at=tile+pos*128;
                auto values=at==newest ? fresh+(NKV+kv)*128 : v+(kv*1024+at)*128;
                float value=float(values[chunk*32+lane]);
                float top=max(peak,score[pos]);
                float a=fast::exp(peak-top),b=fast::exp(score[pos]-top);
                total=total*a+b; acc=acc*a+b*value; peak=top;
            }
            buffer[tile][lane]=bfloat16_t(acc);
            if (!lane) {peaks[tile]=peak;totals[tile]=total;}
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float peak=-FLT_MAX;
        for (uint i=lane;i<128;i+=32) peak=max(peak,peaks[i]);
        peak=simd_max(peak);
        float z=0,acc=0;
        for (uint i=lane;i<128;i+=32) {
            float a=fast::exp(peaks[i]-peak);
            z+=a*totals[i];
            acc+=a*float(buffer[i][warp]);
        }
        z=simd_sum(z);
        float value=float(bfloat16_t(simd_sum(acc)/z));
        if (!lane) {
            uint col=head*128+chunk*32+warp;
            float g=float(gate[col]),p=1.f/(1.f+precise::exp(abs(g)));
            out[col]=bfloat16_t(value*(g<0 ? p : 1.f-p));
        }
    """,
)


def attend(layer,q,k,v,rel,sinks,cache,gate):
    length, heads = k.shape[2], q.shape[1]
    slot = cache._idx if isinstance(cache,(RotatingKVCache,BatchRotatingKVCache)) else length
    before = getattr(cache, "write_coordinates", None)
    slot, end = before if before is not None else (slot, cache.offset)
    w = layer.rel_logits_proj.proj.astype(rel.dtype)
    width, extent = w.shape
    swizzled = extent % 128 == 0
    if swizzled:
        weight = layer.rel_logits_proj.proj
        saved = getattr(layer, "_relative_tiles", None)
        if saved is None or saved[0] is not weight or saved[1] != rel.dtype:
            tiled = mx.contiguous(w.reshape(width, extent//128, 128).transpose(2,0,1))
            saved = (weight, rel.dtype, tiled)
            object.__setattr__(layer, "_relative_tiles", saved)
        w = saved[2]
    paged = hasattr(cache, "page_tail")
    tail = cache.page_tail if paged else q
    if paged:
        k, v = mx.split(cache.page_prefix, 2, axis=1)
    # A compiled ring already carries one-element int32 graph inputs.
    if before is None:
        slot, end = mx.array(slot,mx.int32).reshape(1), mx.array(end,mx.int32).reshape(1)
    fresh = getattr(cache,"fresh",None)
    if (length==layer.sliding_window==1024 and fresh is not None and before is not None
            and q.dtype==gate.dtype==mx.bfloat16):
        return _sliding_fused(inputs=[q,k,v,fresh,rel,w,slot,sinks.astype(q.dtype),scalar(layer.scaling),gate],
                              template=[("GQA",heads//k.shape[1]),("NKV",k.shape[1]),("R",extent),
                                        ("WIDTH",width),("SWIZZLED",swizzled)],
                              grid=(4*1024,heads,1),threadgroup=(1024,1,1),
                              output_shapes=[gate.shape],output_dtypes=[gate.dtype])[0]
    parts,stats = _partial(
        inputs=[q,k,v,tail,q if fresh is None else fresh,rel,w,slot,end,
                sinks.astype(q.dtype),scalar(layer.scaling)],
        template=[("T",q.dtype),("S",length),("PRE_UPDATE",before is not None),("PAGED",paged),("FRESH",fresh is not None),("NKV",k.shape[1]),("GQA",heads//k.shape[1]),("WIDTH",width),("SWIZZLED",swizzled),
                  ("R",extent),("WINDOW",layer.sliding_window)],
        grid=(128*32,heads,1),threadgroup=(128,1,1),
        output_shapes=[(heads,128,128),(heads,128,2)],output_dtypes=[q.dtype,mx.float32])
    return _merge(inputs=[parts,stats,gate],template=[("T",q.dtype)],
                  grid=(32*32,heads,1),threadgroup=(128,1,1),
                  output_shapes=[gate.shape],output_dtypes=[q.dtype])[0]
