// Copyright © 2026 Apple Inc. Adapted from mlx-lm 0c4c991, Inkling relative_attention.
// See MLX-LICENSE.txt. Q and bias are token-major; K/V are read from packed cache storage.
#include <metal_simdgroup_matrix>
kernel void prefill_attention(device const float *q [[buffer(0)]], device const uchar *k [[buffer(1)]],
                              device const uchar *v [[buffer(2)]], device const float *rel [[buffer(3)]],
                              device const float *sinks [[buffer(4)]], device float *out [[buffer(5)]],
                              constant uint4 &args [[buffer(6)]], uint2 group [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
    constexpr int D=128, HQ=24, HKV=8, R=1024, TG=128, BQ=32;

    constexpr float LOG2E = 1.4426950408889634f;
    constexpr int BK = 16;
    constexpr int LDK = BK + 8;
    constexpr int LDV = D + 8;
    const int L = args.x, S = args.y;
    const int WINDOW = args.z;
    const int head = group.y;
    const int kv_head = head / (HQ / HKV);
    const int q0 = group.x * BQ;
    constexpr int start = 0;
    const float qscale = rsqrt(128.f) * LOG2E;

    // The lane holds row fm and columns fn, fn + 1 of each 8x8 fragment.
    const short fm = ((lane / 4) & 4) + (lane / 2) % 4;
    const short fn = ((lane / 4) & 2) * 2 + (lane % 2) * 2;
    const int row = q0 + int(sg) * 8 + fm;
    const bool row_ok = row < L;
    const int diag = S - L + row;

    // q and k are clipped and v has unit RMS, so half holds them.
    threadgroup half Ks[D * LDK];
    threadgroup half Vs[BK * LDV];

    simdgroup_half8x8 q_frag[D / 8];
    for (int db = 0; db < D / 8; ++db) {
        half2 qv = half2(0.0h);
        if (row_ok) {
            const device float* qp = q + (row * HQ + head) * D + db * 8 + fn;
            qv = half2(float(qp[0]), float(qp[1]));
        }
        q_frag[db].thread_elements()[0] = qv.x;
        q_frag[db].thread_elements()[1] = qv.y;
    }
    simdgroup_float8x8 acc[D / 8];
    for (int db = 0; db < D / 8; ++db) {
        acc[db] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    float row_max = row_ok ? float(sinks[row * HQ + head]) * LOG2E : 0.0f;
    float row_sum = 1.0f;
    const int bias_row = row_ok ? row : 0;
    const device float* bias = rel + (size_t(bias_row) * HQ + head) * R;

    const int first_diag = S - L + q0;
    const int last_diag = S - L + min(q0 + BQ, L) - 1;
    int first_key = WINDOW > 0 ? max(0, first_diag - WINDOW + 1) : 0;
    first_key = max(first_key, start);

    for (int k0 = first_key / BK * BK; k0 <= last_diag; k0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int c = tid; c < BK * D / 8; c += TG) {
            const int r = c / (D / 8);
            const int col = (c % (D / 8)) * 8;
            const int key = k0 + r;
            size_t offset = (size_t(key) * HKV + kv_head) * D + col;
            for (int j = 0; j < 8; ++j) {
                Ks[(col + j) * LDK + r] = key < S ? half(cache_value(k, offset+j, args.w)) : 0.h;
                Vs[r * LDV + col + j] = key < S ? half(cache_value(v, offset+j, args.w)) : 0.h;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (q0 + int(sg) * 8 >= L) {
            continue;
        }

        simdgroup_float8x8 scores[BK / 8];
        for (int kb = 0; kb < BK / 8; ++kb) {
            scores[kb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int db = 0; db < D / 8; ++db) {
                simdgroup_half8x8 k_frag;
                simdgroup_load(k_frag, Ks + db * 8 * LDK + kb * 8, LDK);
                simdgroup_multiply_accumulate(scores[kb], q_frag[db], k_frag, scores[kb]);
            }
        }

        float tile_max = -INFINITY;
        for (int kb = 0; kb < BK / 8; ++kb) {
            for (int e = 0; e < 2; ++e) {
                const int key = k0 + kb * 8 + fn + e;
                const int d = diag - key;
                const bool visible = row_ok && d >= 0 && key >= start
                    && (WINDOW == 0 || d < WINDOW);
                float s = -INFINITY;
                if (visible) {
                    s = scores[kb].thread_elements()[e] * qscale;
                    if (d < R) {
                        s += float(bias[d]) * LOG2E;
                    }
                }
                scores[kb].thread_elements()[e] = s;
                tile_max = max(tile_max, s);
            }
        }
        // The 4 lanes of a row are lane ^ 1 and lane ^ 8 apart.
        tile_max = max(tile_max, simd_shuffle_xor(tile_max, 1));
        tile_max = max(tile_max, simd_shuffle_xor(tile_max, 8));
        const float new_max = max(row_max, tile_max);
        const float factor = fast::exp2(row_max - new_max);
        float tile_sum = 0.0f;
        simdgroup_half8x8 probs[BK / 8];
        for (int kb = 0; kb < BK / 8; ++kb) {
            for (int e = 0; e < 2; ++e) {
                const float p = fast::exp2(scores[kb].thread_elements()[e] - new_max);
                probs[kb].thread_elements()[e] = half(p);
                tile_sum += p;
            }
        }
        tile_sum += simd_shuffle_xor(tile_sum, 1);
        tile_sum += simd_shuffle_xor(tile_sum, 8);
        row_sum = row_sum * factor + tile_sum;
        row_max = new_max;
        for (int db = 0; db < D / 8; ++db) {
            acc[db].thread_elements()[0] *= factor;
            acc[db].thread_elements()[1] *= factor;
            for (int kb = 0; kb < BK / 8; ++kb) {
                simdgroup_half8x8 v_frag;
                simdgroup_load(v_frag, Vs + kb * 8 * LDV + db * 8, LDV);
                simdgroup_multiply_accumulate(acc[db], probs[kb], v_frag, acc[db]);
            }
        }
    }

    if (row_ok) {
        device float* op = out + (size_t(row) * HQ + head) * D;
        for (int db = 0; db < D / 8; ++db) {
            op[db * 8 + fn] = float(acc[db].thread_elements()[0] / row_sum);
            op[db * 8 + fn + 1] = float(acc[db].thread_elements()[1] / row_sum);
        }
    }
}
