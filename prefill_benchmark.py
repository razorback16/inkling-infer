"""Measure uncached, batch-one prompt processing through the MLX cache-fill path."""
import argparse
import json
import statistics
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from decode_benchmark import MODEL, PREFIX
from roofline import CONTEXT

CHUNK = 2048


def fill(model, tokens, cache, chunk=CHUNK):
    for start in range(0, tokens.size, chunk):
        model(tokens[None, start:start + chunk], cache=cache)
        mx.eval([entry.state for entry in cache])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock", action="store_true")
    args = parser.parse_args()
    if not args.stock:
        from inkling_kernel import install
        install()
    model, tokenizer = load(MODEL)
    tokens = mx.array(tokenizer.encode(PREFIX.read_text()))
    assert tokens.size == CONTEXT
    mx.eval(tokens)
    rates = []
    for run in range(4):
        cache = make_prompt_cache(model)
        assert len(cache) == 48 and all(entry[0].offset == 0 for entry in cache)
        start = time.perf_counter()
        fill(model, tokens, cache)
        elapsed = time.perf_counter() - start
        assert all(entry[0].offset == CONTEXT for entry in cache)
        rate = CONTEXT / elapsed
        if run:
            rates.append(rate)
        print(json.dumps(dict(warmup=not run, seconds=elapsed, prompt_tps=rate)), flush=True)
        del cache
    print(statistics.median(rates))


if __name__ == "__main__":
    main()
