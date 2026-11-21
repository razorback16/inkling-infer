"""Calibrate an empirical prefill roof from BF16 GEMM and required model work.

Count one multiply-add as two FLOPs. Count useful causal QK/AV and relative
bias dots, all selected expert dots, dense projections, and recurrent message
projections. Scalar operations, dequantization and depth mixing are overhead
against this idealized matrix-compute ceiling. The vocabulary head is lazy
during upstream cache-only prefill.
"""
import json
import statistics
import time

import mlx.core as mx
from mlx_lm import load
from mlx.utils import tree_flatten

from prefill_benchmark import CHUNK
from roofline import CONTEXT, MODEL, sustained_reads


def gemm_rates():
    mx.random.seed(188)
    shapes = [(2048, 2304, 3072), (2048, 4096, 4096),
              (4096, 4096, 4096), (4096, 4096, 8192)]
    pairs = [
        tuple(mx.random.normal(shape).astype(mx.bfloat16) for shape in ((m, k), (n, k)))
        for m, k, n in shapes
    ]
    mx.eval(pairs)
    samples = [[] for _ in shapes]
    for rep in range(13):
        for j in range(len(shapes)):
            i = (j + rep) % len(shapes)
            m, k, n = shapes[i]
            a, b = pairs[i]
            start = time.perf_counter_ns()
            y = a @ b.T
            mx.eval(y)
            elapsed = time.perf_counter_ns() - start
            if rep >= 3:
                samples[i].append(2 * m * k * n / elapsed / 1000)
    return {str(shape): statistics.median(rates) for shape, rates in zip(shapes, samples)}


def work(model):
    lm = model.model.language_model
    args = lm.args
    flops = dict(dense=0, experts=0, attention=0, relative=0, messages=0)
    weight_bytes = 0
    kv_bytes = 0
    for block in lm.layer_visits:
        attn = block.self_attn
        dense = [attn.q_proj, attn.k_proj, attn.v_proj, attn.gate_proj,
                 attn.o_proj, attn.r_proj, attn.sink_proj, block.mlp.gate]
        dense += [c.weight_proj for c in (attn.k_sconv, attn.v_sconv) if hasattr(c, "weight_proj")]
        flops["dense"] += 2 * sum(p.weight.size for p in dense) * CONTEXT
        flops["experts"] += 6 * args.hidden_size * args.intermediate_size * args.num_experts_per_tok * CONTEXT
        pairs = sum(min(i, attn.sliding_window or CONTEXT) for i in range(1, CONTEXT + 1))
        flops["attention"] += 4 * pairs * attn.num_heads * attn.head_dim
        width, extent = attn.rel_logits_proj.proj.shape
        relative_pairs = sum(min(i, extent, attn.sliding_window or CONTEXT) for i in range(1, CONTEXT + 1))
        flops["relative"] += 2 * relative_pairs * attn.num_heads * width
        # Allow even all 128 experts to stream once on each visit and chunk.
        weight_bytes += sum(v.nbytes for _, v in tree_flatten(block.parameters()))
        kv_bytes += 2 * CONTEXT * attn.num_key_value_heads * attn.head_dim * 2
    for i in args.loop_core:
        layer = lm.layers[i]
        gate = layer.mlp_message_gate
        per_gate = sum(p.weight.size for p in (gate.down_proj, gate.forget_gate_proj, gate.message_gate_proj))
        flops["messages"] += 2 * CONTEXT * (args.loop_config["num_passes"] - 1) * 4 * per_gate
    chunks = (CONTEXT + CHUNK - 1) // CHUNK
    stream_bytes = chunks * weight_bytes + kv_bytes * (chunks + 1)
    return flops, stream_bytes


if __name__ == "__main__":
    compute = gemm_rates()
    bandwidth = sustained_reads()
    model, _ = load(MODEL)
    ops, traffic = work(model)
    compute_roof = max(compute.values()) * 1e12 * CONTEXT / sum(ops.values())
    stream_roof = max(bandwidth.values()) * 1e9 * CONTEXT / traffic
    ceiling = min(compute_roof, stream_roof)
    print(json.dumps(dict(device=mx.device_info()["device_name"], context=CONTEXT, chunk=CHUNK,
                          gemm_tflops=compute, stream_GB_s=bandwidth, prompt_flops=ops,
                          logical_stream_bytes=traffic, compute_roof_tps=compute_roof,
                          stream_roof_tps=stream_roof, roof_tps=ceiling,
                          target_80_tps=0.8 * ceiling), indent=2))
