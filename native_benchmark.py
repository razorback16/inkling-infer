"""Measure warm native prefill and cached single-stream decode on the fixed prefix."""
import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
from transformers import PreTrainedTokenizerFast

from inkling_native.model import TextModel
from inkling_native.server import checkpoint


def measure(model, tokens, chunk, decode_steps, repetitions):
    chunk = min(chunk, len(tokens))
    # Compile and allocate before timing. Each sample starts with an empty KV cache.
    warm = model.cache()
    model.forward(tokens[:chunk], warm, logits=False)
    model.forward(tokens[-chunk:], warm)
    model.forward(tokens[:1], warm)
    del warm
    samples = []
    for _ in range(repetitions):
        cache = model.cache()
        start = time.perf_counter()
        for pos in range(0, len(tokens), chunk):
            logits = model.forward(tokens[pos:pos+chunk], cache, logits=pos+chunk >= len(tokens))
        mid = time.perf_counter()
        for _ in range(decode_steps):
            logits = model.forward([int(np.argmax(logits))], cache)
        end = time.perf_counter()
        samples.append((len(tokens)/(mid-start), decode_steps/(end-mid)))
    return dict(prefill=statistics.median(s[0] for s in samples),
                decode=statistics.median(s[1] for s in samples), samples=samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--chunk", type=int, default=2048)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--kv-cache-dtype", choices=("fp8", "bf16"), default="bf16")
    args = parser.parse_args()
    path = checkpoint()
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(path), local_files_only=True)
    prefix = Path(__file__).parent / "benchmarks" / "moby-prefix.txt"
    tokens = tokenizer.encode(prefix.read_text())
    if not 1 <= args.context <= len(tokens) or min(args.chunk, args.decode_tokens, args.repetitions) < 1:
        parser.error("Choose positive sizes and a context within the fixed prefix")
    model = TextModel(path, args.kv_cache_dtype)
    print(json.dumps(measure(model, tokens[:args.context], args.chunk, args.decode_tokens, args.repetitions)))


if __name__ == "__main__":
    main()
