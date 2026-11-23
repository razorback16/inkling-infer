// BF16 weight matrix, float32 accumulator. The host controls BF16 rounding
// at the model's activation boundaries.
kernel void linear(device const float *x [[buffer(0)]], device const bfloat *w [[buffer(1)]],
                   device float *y [[buffer(2)]], constant uint2 &shape [[buffer(3)]],
                   uint2 group [[threadgroup_position_in_grid]],
                   uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    uint row = group.x * 8 + sg;
    if (row >= shape.x) return;
    float sum = 0;
    for (uint i = lane; i < shape.y; i += 32)
        sum = fma(x[group.y * shape.y + i], float(w[row * shape.y + i]), sum);
    sum = simd_sum(sum);
    if (!lane) y[group.y * shape.x + row] = sum;
}

inline float cache_value(device const uchar *buf, size_t i, bool fp8) {
    return fp8 ? e4m3(buf[i]) : float(((device const bfloat *)buf)[i]);
}

// Adapted from MLX v0.32.4 sdpa_vector.h (native/MLX-LICENSE.txt).
// Each SIMD group accumulates an online softmax over a partition of the visible
// history. The sink is a score with zero value and belongs to partition zero.
kernel void attention(device const float *q [[buffer(0)]], device const uchar *keys [[buffer(1)]],
                      device const uchar *values [[buffer(2)]], device const float *bias [[buffer(3)]],
                      device const float *sinks [[buffer(4)]], device float *out [[buffer(5)]],
                      constant uint4 &args [[buffer(6)]], uint2 group [[threadgroup_position_in_grid]],
                      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    uint head = group.x, token = group.y, row = token * 24 + head;
    uint end = args.x + token + 1, start = args.y && end > args.y ? end - args.y : 0;
    float4 query = ((device const float4 *)q)[row * 32 + lane] * rsqrt(128.f);
    float peak = sg ? -FLT_MAX : sinks[row], z = sg ? 0.f : 1.f;
    float4 accum = 0;
    for (uint pos = start + sg; pos < end; pos += 32) {
        uint distance = end - 1 - pos;
        size_t i = (size_t(pos) * 8 + head / 3) * 128 + lane * 4;
        float4 k, v;
        for (uint j = 0; j < 4; ++j) {
            k[j] = cache_value(keys, i + j, args.z);
            v[j] = cache_value(values, i + j, args.z);
        }
        float score = 0;
        for (uint j = 0; j < 4; ++j) score += query[j] * k[j];
        score = simd_sum(score);
        if (distance < args.w) score += bias[row * args.w + distance];
        float top = max(score, peak), a = fast::exp(peak - top), b = fast::exp(score - top);
        accum = accum * a + v * b;
        z = z * a + b;
        peak = top;
    }
    threadgroup float peaks[32], totals[32], partial[32][32];
    if (!lane) { peaks[sg] = peak; totals[sg] = z; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float top = simd_max(peaks[lane]);
    float weight = fast::exp(peaks[lane] - top);
    float total = simd_sum(totals[lane] * weight);
    for (uint j = 0; j < 4; ++j) {
        partial[lane][sg] = accum[j];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float sum = simd_sum(partial[sg][lane] * weight);
        if (!lane) out[row * 128 + sg * 4 + j] = sum / total;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
