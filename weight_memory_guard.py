"""Check that inference retains a single set of learned model tensors."""
import argparse
import gc
from collections import deque

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm import load, stream_generate
from mlx_lm.models.cache import make_prompt_cache

from inkling_kernel import install
from prefill_benchmark import CHUNK, MODEL, PREFIX


def check_module_attributes(model, parameters):
    # Compiled functions may reference the canonical tensors. Identity makes those
    # references free while catching separately attached packed or expanded arrays.
    for path, module in model.named_modules():
        pending = deque(vars(module).items())
        seen = set()
        while pending:
            name, value = pending.pop()
            if id(value) in seen:
                continue
            seen.add(id(value))
            if isinstance(value, mx.array):
                if id(value) not in parameters:
                    raise AssertionError(f"Extra tensor on {path}.{name} ({value.nbytes} bytes)")
            elif isinstance(value, (tuple, list)):
                pending.extend((name, item) for item in value)
            elif isinstance(value, dict) and not hasattr(value, "parameters"):
                pending.extend((name, item) for item in value.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-tokens", type=int, default=4096)
    args = parser.parse_args()
    if args.context_tokens < 8:
        parser.error("the prompt must contain at least eight tokens")
    install()
    model, tokenizer = load(MODEL)
    original = {name: id(a) for name, a in tree_flatten(model.parameters())}
    prefix = tokenizer.encode(PREFIX.read_text())
    repeats = (args.context_tokens + len(prefix) - 1) // len(prefix)
    tokens = mx.array((prefix * repeats)[:args.context_tokens])
    mx.eval(tokens)
    cache = make_prompt_cache(model)
    baseline = mx.get_active_memory()
    for response in stream_generate(model, tokenizer, tokens, prompt_cache=cache,
                                    prefill_step_size=CHUNK, max_tokens=8):
        pass
    mx.synchronize()
    assert response.prompt_tokens == args.context_tokens
    assert all(kv.offset >= args.context_tokens for kv, _ in cache)
    print(f"Live and peak MLX memory: {mx.get_active_memory()/2**30:.3f} and "
          f"{mx.get_peak_memory()/2**30:.3f} GiB")
    del response, cache
    gc.collect()
    mx.synchronize()
    mx.clear_cache()

    assert original == {k: id(a) for k, a in tree_flatten(model.parameters())}
    check_module_attributes(model, set(original.values()))
    extra = mx.get_active_memory() - baseline
    print(f"Canonical parameters verified. Retained MLX memory change: {extra:,} bytes.")
    # Generic activation/code lookup tables are independent of the checkpoint.
    # Allow room for framework constants but catch hidden compiled weight replicas.
    assert extra < 8 * 1024**2, f"Unexpected persistent allocation of {extra:,} bytes"


if __name__ == "__main__":
    main()
