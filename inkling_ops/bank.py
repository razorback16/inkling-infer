"""Evaluate all queries on a shared cross-pass message bank in one dispatch."""
import mlx.core as mx
from mlx_lm.models.inkling2 import layers

_kernel = mx.fast.metal_kernel(
    name="inkling_bank_queries", input_names=["blocks", "last", "query"], output_names=["out"],
    source="""
        uint q = threadgroup_position_in_grid.x, tid = thread_position_in_threadgroup.x;
        uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
        threadgroup float parts[K+1][9];
        for (uint s = 0; s <= K; ++s) {
            float dot = 0;
            for (uint d = tid; d < 2304; d += 288) {
                float value = s == K ? last[d] : blocks[s * 2304 + d];
                dot += value * query[q * 2304 + d];
            }
            dot = simd_sum(dot);
            if (lane == 0) parts[s][sg] = dot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float weights[K+1], peak = -INFINITY;
        for (uint s = 0; s <= K; ++s) {
            float acc = 0;
            for (uint i = 0; i < 9; ++i) acc += parts[s][i];
            weights[s] = acc * (1.f / 48.f);
            peak = max(peak, weights[s]);
        }
        float z = 0;
        for (uint s = 0; s <= K; ++s) {
            weights[s] = metal::exp(weights[s] - peak);
            z += weights[s];
        }
        for (uint d = tid; d < 2304; d += 288) {
            float value = 0;
            for (uint s = 0; s < K; ++s)
                value += (weights[s] / z) * blocks[s * 2304 + d];
            out[q * 2304 + d] = value + (weights[K] / z) * last[d];
        }
    """,
)


def messages(model, bank):
    blocks, last = bank
    queries = model.message_input.query
    if (type(model.message_input) is not layers.MessageInput or last.shape != (1, 1, 2304)
            or blocks.shape[0] < 1 or queries.dtype != mx.float32):
        return [model.message_input(bank, i) for i in range(len(model.args.loop_core))]
    n, q, d = queries.shape
    return _kernel(inputs=[blocks, last, queries], template=[("K", blocks.shape[0])],
                   grid=(n*q*288, 1, 1), threadgroup=(288, 1, 1),
                   output_shapes=[(n, 1, 1, q, d)], output_dtypes=[mx.float32])[0]


def run(model, hidden, cache):
    core = model.args.loop_core
    visits = iter(cache)
    def visit(group, state, bank=None):
        h, blocks = state
        for i, layer in enumerate(group):
            h, blocks = layer(h, blocks, cache=next(visits), conv_mask=None,
                              messages=None if bank is None else bank[i])
        return h, blocks

    prelude = visit(model.layers[:core.start], (hidden, None))
    group = model.layers[core.start:core.stop]
    state = visit(group, prelude)
    for _ in range(1, model.args.loop_config["num_passes"]):
        h, blocks = state
        bank = (blocks[prelude[1].shape[0]+1:], layers.unweighted_rms_norm(h.astype(mx.float32)))
        state = visit(group, prelude, messages(model, bank))
    return visit(model.layers[core.stop:], state)
