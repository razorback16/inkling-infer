"""End-to-end batch-one generation on a fixed 4096-token Moby-Dick prefix."""
import argparse
import json
from pathlib import Path
import statistics
import time

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

from roofline import CONTEXT, GENERATED, MODEL

PREFIX = Path(__file__).parent / "benchmarks" / "moby-prefix.txt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", action="store_true")
    args = parser.parse_args()
    if args.kernel:
        from inkling_kernel import install
        install()
    model, tokenizer = load(MODEL)
    prompt = tokenizer.encode(PREFIX.read_text())
    assert len(prompt) == CONTEXT
    sampler = make_sampler(temp=1.0, top_p=.95, top_k=20)
    rates = []
    for seed in (41, 42, 43, 44):
        mx.random.seed(seed)
        times = []
        start = time.perf_counter()
        for result in stream_generate(model, tokenizer, prompt, sampler=sampler, max_tokens=GENERATED):
            times.append(time.perf_counter())
        if len(times) != GENERATED or result.finish_reason != "length":
            raise RuntimeError(f"Expected {GENERATED} generated tokens, got {len(times)}")
        rate = (len(times) - 1) / (times[-1] - times[0])
        if seed != 41:
            rates.append(rate)
        print(json.dumps(dict(seed=seed, ttft_s=times[0] - start, decode_tps=rate)), flush=True)
    print(statistics.median(rates))


if __name__ == "__main__":
    main()
