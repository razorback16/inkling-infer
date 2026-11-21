"""Compare teacher-forced cached decode logits against the original checkpoint."""
import gc
import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from decode_benchmark import MODEL, PREFIX


def snapshots():
    model, tokenizer = load(MODEL)
    ids = tokenizer.encode(PREFIX.read_text())
    code = tokenizer.encode("def fibonacci(n):\n    return")
    saved = []
    for prompt in (code, ids[:1030], ids):
        cache = make_prompt_cache(model)
        # Fill the real model cache exactly as the generator does on prefill.
        for start in range(0, len(prompt), 512):
            model(mx.array([prompt[start:start + 512]]), cache=cache)
            mx.eval([c.state for c in cache])
        for token in code[:6]:
            logits = model(mx.array([[token]]), cache=cache)[0, -1].astype(mx.float32)
            saved.append(np.array(logits))
    return saved


if __name__ == "__main__":
    reference = snapshots()
    gc.collect()
    mx.clear_cache()
    from inkling_kernel import install
    install()
    candidate = snapshots()
    for step, (orig, actual) in enumerate(zip(reference, candidate, strict=True)):
        relative_rms = np.linalg.norm(actual - orig) / np.linalg.norm(orig)
        best, runner_up = np.argsort(orig)[-2:][::-1]
        assert np.isfinite(relative_rms) and relative_rms < .02, (step, relative_rms)
        if orig[best] - orig[runner_up] > .5:
            assert actual.argmax() == best, (step, best, actual.argmax())
    print(f"Passed {len(reference)} full-vocabulary logit comparisons")
