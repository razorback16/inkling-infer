"""Estimate the batch-one decode bandwidth roof from native weights and attended KV."""
import json
import statistics
import time
from collections import Counter

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm import load

MODEL = "thinkingmachines/Inkling-2-Mini"
CONTEXT = 4096
GENERATED = 128


def sustained_reads():
    # The input is larger than the on-chip caches. Each SIMD lane consumes
    # consecutive vectors and emits a checksum to keep every load observable.
    read = mx.fast.metal_kernel(
        name="streaming_read",
        input_names=["data"],
        output_names=["out"],
        source="""
            uint lane = thread_position_in_grid.x;
            device const uint4 *vectors = (device const uint4 *)data;
            uint4 checksum = 0;
            for (uint i = lane; i < VECTORS; i += THREADS) {
                checksum ^= vectors[i];
            }
            out[lane] = checksum.x ^ checksum.y ^ checksum.z ^ checksum.w;
        """,
    )
    words = 2**28
    data = mx.random.uniform(shape=(words,)).view(mx.uint32)
    mx.eval(data)
    bandwidth = {}
    for threads in (65536, 262144, 1048576):
        rates = []
        for step in range(14):
            start = time.perf_counter()
            result = read(inputs=[data], template=[("VECTORS", words // 4), ("THREADS", threads)],
                          grid=(threads, 1, 1), threadgroup=(256, 1, 1),
                          output_shapes=[(threads,)], output_dtypes=[mx.uint32])
            mx.eval(result)
            if step >= 4:
                rates.append(data.nbytes / (time.perf_counter() - start) / 1e9)
        bandwidth[threads] = statistics.median(rates)
    return bandwidth


def decode_bytes(model):
    lm = model.model.language_model
    args = lm.args
    totals = Counter()
    for name, tensor in tree_flatten(model.parameters()):
        if name.startswith("model.vision_tower."):
            continue
        if name.startswith("model.language_model.layers."):
            layer = int(name.split(".")[3])
            visits = args.loop_config["num_passes"] if layer in args.loop_core else 1
            if ".experts." in name:
                assert tensor.shape[0] == args.n_routed_experts, name
                totals["selected_experts"] += visits * tensor.nbytes * args.num_experts_per_tok // tensor.shape[0]
            elif "_message_gate." in name:
                totals["dense_weights"] += (visits - 1) * tensor.nbytes
            else:
                totals["dense_weights"] += visits * tensor.nbytes
        elif name.endswith("embed_tokens.weight"):
            totals["dense_weights"] += tensor.nbytes // tensor.shape[0]
        elif name.endswith("message_input.query"):
            totals["dense_weights"] += (args.loop_config["num_passes"] - 1) * tensor.nbytes
        else:
            totals["dense_weights"] += tensor.nbytes

    # After the first token, decode sees lengths 4097 through 4223.
    mean_full_length = CONTEXT + GENERATED / 2
    for layer in lm.layer_visits:
        attn = layer.self_attn
        length = min(mean_full_length, attn.sliding_window or mean_full_length)
        totals["kv_read_write"] += int(2 * (length + 1) * attn.num_key_value_heads * attn.head_dim * 2)
    return totals


if __name__ == "__main__":
    rates = sustained_reads()
    model, _ = load(MODEL)
    traffic = decode_bytes(model)
    ceiling = max(rates.values()) * 1e9 / sum(traffic.values())
    print(json.dumps(dict(read_GB_s=rates, bytes_per_token=traffic, bandwidth_roof_tps=ceiling,
                          target_80_percent_tps=ceiling * .8), indent=2))
