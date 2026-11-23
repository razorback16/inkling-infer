"""Generate reference logits with upstream PyTorch, then check the native serving path.

The reference and check commands can run in separate virtual environments. The
archive records the actual token IDs, so both phases exercise identical inputs.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from inkling_native.server import checkpoint

CONTEXTS = (10, 1030, 4096)
TOLERANCE = 0.02


def holdout_cases(tokenizer):
    density = "Explain in plain English why ice floats on water. Then give a two-line Python example of density as mass divided by volume. Be precise about the units."
    journal = ("The regression test appends records, rotates the journal, and reads all entries in order. "
               "At 08:42 UTC a new worker starts. It asks whether job number 418 was acknowledged, whether the queue has room, and who will retry it. "
               "測試資料。 Résumé des événements.\n")
    extra = tokenizer.encode(" The density of water is close to")[:4]
    for name, prompt in (("density", tokenizer.encode(density)), ("journal", tokenizer.encode(journal * 40)[:1537])):
        yield name, prompt + extra, [len(prompt) - 1, 1] + [1] * len(extra)
    notebook = ("Summarize this field notebook. At low tide we counted eleven red crabs and six blue mussels. "
                "À marée haute, trois bateaux sont entrés dans le port. The entries were written at 13:07 and 14:19. "
                "Which statements give a count, and which give a time? Explain the distinction in two short paragraphs. ")
    chunks = [2, 7, 8, 16, 64, 3, 1, 1, 1, 1, 1, 1]
    yield "notebook", tokenizer.encode(notebook * 3)[:sum(chunks)], chunks


def greedy_holdout_cases(tokenizer):
    prompts = {
        "search": 'def binary_search(a, target):\n    """Return the index of target or -1."""\n',
        "french": "Écris un courriel très court pour déplacer notre réunion de lundi à mercredi.",
        "digits": "Complete each pair following the same rule. 1 → 1, 2 → 4, 3 → 9, 4 →",
        "json": '{"name":"Mercury","moons":0}\n{"name":"Venus","moons":0}\n{"name":',
        "poem": "On a foggy morning the bicycle bell rang twice. Write the next two lines.",
        "sql": "SELECT department, COUNT(*) AS employees\nFROM staff\n",
        "zh": "用两句话解释为什么海水是咸的。",
        "math": "Let x and y be positive integers such that x + y = 17 and xy = 60. Find x and y.",
    }
    for name, prompt in prompts.items():
        ids = tokenizer.encode(prompt)
        yield name, ids, [len(ids)] + [1] * 15


def record_long_context(output, model, metadata):
    import torch

    prompt = np.load(Path(__file__).parent / "benchmarks/long-context-prompt.npy", allow_pickle=False)
    if prompt.shape != (8192,) or prompt.dtype != np.uint32:
        raise ValueError("Invalid long-context token fixture")
    ids = prompt.tolist()
    chunks = [512] * 12 + [1, 511] + [512] * 3 + [1] * 4
    cache, start = None, 0
    positions, logits = [], []
    with torch.inference_mode():
        for size in chunks:
            torch.set_num_threads(1 if size == 1 else 12)
            if start == len(ids):
                ids.append(int(logits[-1].argmax()))
            result = model(input_ids=torch.tensor([ids[start:start + size]]), cache=cache,
                           use_cache=True, logits_to_keep=1)
            cache = result.cache
            start += size
            if start >= 6144:
                positions.append(start)
                logits.append(result.logits[0, -1].float().numpy().copy())
            print(f"Processed {start} tokens", flush=True)
    np.savez(output, ids=ids, chunks=chunks, positions=positions, logits=logits, metadata=metadata)


def reference(output, chunk, holdouts=False, greedy_holdouts=False, long_context=False):
    import torch
    import transformers
    from transformers import AutoModelForMultimodalLM, PreTrainedTokenizerFast

    path = checkpoint()
    torch.set_num_threads(12)
    model = AutoModelForMultimodalLM.from_pretrained(
        path, local_files_only=True, dtype=torch.bfloat16, device_map="cpu",
        attn_implementation="eager", experts_implementation="eager").eval()
    metadata = json.dumps(dict(checkpoint=str(path.resolve()), revision=path.name, torch=torch.__version__,
                               transformers=transformers.__version__, cache="bf16", chunk=chunk))
    if long_context:
        return record_long_context(output, model, metadata)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(path, local_files_only=True)
    prose = tokenizer.encode((Path(__file__).parent / "benchmarks/moby-prefix.txt").read_text())
    code = tokenizer.encode("def fibonacci(n):\n    return")
    records = dict(metadata=metadata, continuation=code[:6])
    with torch.inference_mode():
        for length, prompt in zip(CONTEXTS, (code, prose[:1030], prose[:4096]), strict=True):
            if len(prompt) != length:
                raise ValueError(f"Unexpected token count for {length}-token case")
            records[f"prompt_{length}"] = prompt
            cache = None
            steps = []
            for start in range(0, length, chunk):
                batch = torch.tensor([prompt[start:start+chunk]])
                result = model(input_ids=batch, cache=cache, use_cache=True, logits_to_keep=1)
                cache = result.cache
            steps.append(result.logits[0, -1].float().numpy().copy())
            for token in code[:6]:
                result = model(input_ids=torch.tensor([[token]]), cache=cache, use_cache=True, logits_to_keep=1)
                cache = result.cache
                steps.append(result.logits[0, -1].float().numpy().copy())
            records[f"logits_{length}"] = np.stack(steps)
            print(f"Recorded {length} prompt tokens and {len(steps)-1} decode steps", flush=True)
        if holdouts or greedy_holdouts:
            cases = list(holdout_cases(tokenizer)) if holdouts else []
            greedy = list(greedy_holdout_cases(tokenizer)) if greedy_holdouts else []
            cases.extend(greedy)
            greedy_names = {name for name, _, _ in greedy}
            records["holdout_names"] = [name for name, _, _ in cases]
            for name, ids, chunks in cases:
                if name in greedy_names:
                    # Small CPU depthwise convolutions run faster on one thread.
                    torch.set_num_threads(1)
                records[f"{name}_chunks"] = chunks
                steps, start, cache = [], 0, None
                for size in chunks:
                    if start == len(ids):
                        ids.append(int(steps[-1].argmax()))
                    batch = torch.tensor([ids[start:start + size]])
                    result = model(input_ids=batch, cache=cache, use_cache=True, logits_to_keep=1)
                    steps.append(result.logits[0, -1].float().numpy().copy())
                    start += size
                    cache = result.cache
                records[f"{name}_ids"] = ids
                records[f"{name}_logits"] = np.stack(steps)
                print(f"Recorded {name} at {len(chunks)} positions", flush=True)
    np.savez(output, **records)


def compare_position(logits, expected, vocab, label):
    if logits.shape != expected.shape or logits.shape != (vocab,) or not np.all(np.isfinite(expected)):
        raise ValueError("Invalid full-vocabulary reference")
    expected = expected.astype(np.float64)
    delta = logits.astype(np.float64) - expected
    error = np.linalg.norm(delta) / np.linalg.norm(expected)
    print(f"{label} relative L2 {error:.4%} max abs {np.max(np.abs(delta)):.6f} "
          f"top-1 {logits.argmax() == expected.argmax()}")
    return np.isfinite(error) and error <= TOLERANCE


def check_long_context(model, records):
    ids, chunks, positions, expected = (records[k] for k in ("ids", "chunks", "positions", "logits"))
    ends = np.cumsum(chunks)
    if (np.any(chunks < 1) or ends[-1] != len(ids) or len(ids) != 8196
            or not np.array_equal(positions, ends[ends >= 6144]) or len(positions) != len(expected)):
        raise ValueError("Invalid long-context reference")
    by_position = dict(zip(positions, expected, strict=True))
    cache, start = model.cache(), 0
    failures = 0
    for end in ends:
        logits = model.forward(ids[start:end], cache, logits=end in by_position)
        if end in by_position:
            failures += int(not compare_position(logits, by_position[end], model.vocab, f"long context {end}"))
        start = end
    if failures:
        raise SystemExit(f"{failures} long-context positions exceeded the 2% full-logit tolerance")


def check_decode_forks(model, prompt):
    """Compare interleaved greedy and forced tokens with independent prefix replays."""
    cache = model.cache()
    logits = model.forward(prompt, cache)
    first = int(logits.argmax())
    logits = model.forward([first], cache)
    branches = (cache, [visit.fork() for visit in cache])
    scores = [logits, logits]
    histories = [[], []]
    for branch in (0, 1, 1, 0, 1, 0):
        token = int(scores[branch].argmax())
        if branch == 1 and not histories[branch]:
            token = (token + 1) % model.vocab
        scores[branch] = model.forward([token], branches[branch])
        histories[branch].append((token, scores[branch]))

    for history in histories:
        cache = model.cache()
        model.forward(prompt, cache)
        model.forward([first], cache)
        for token, expected in history:
            model.plans[1, True].discard_prediction()
            actual = model.forward([token], cache)
            np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


def check(source, dtype):
    from inkling_native.model import TextModel
    path = checkpoint()
    model = TextModel(path, dtype)
    failures = []
    with np.load(source, allow_pickle=False) as records:
        metadata = json.loads(str(records["metadata"]))
        if metadata["revision"] != path.name:
            raise ValueError("Native and PyTorch checkpoint revisions differ")
        if "positions" in records:
            return check_long_context(model, records)
        chunk = metadata["chunk"]
        for length in CONTEXTS:
            cache = model.cache()
            prompt = records[f"prompt_{length}"]
            for start in range(0, length, chunk):
                logits = model.forward(prompt[start:start+chunk], cache, logits=start+chunk >= length)
            for step, expected in enumerate(records[f"logits_{length}"]):
                if step:
                    logits = model.forward([records["continuation"][step-1]], cache)
                if not compare_position(logits, expected, model.vocab, f"context {length:4} step {step}"):
                    failures.append((length, step))
        check_decode_forks(model, records["prompt_10"])
        for name in records.get("holdout_names", ()):
            ids, chunks = records[f"{name}_ids"], records[f"{name}_chunks"]
            if np.any(chunks < 1) or chunks.sum() != len(ids):
                raise ValueError(f"Invalid chunk sizes in {name}")
            cache, start = model.cache(), 0
            for size, expected in zip(chunks, records[f"{name}_logits"], strict=True):
                logits = model.forward(ids[start:start + size], cache)
                start += size
                if not compare_position(logits, expected, model.vocab, f"{name} position {start} chunk {size}"):
                    failures.append((name, start))
    if failures:
        raise SystemExit(f"{len(failures)} positions exceeded the 2% full-logit tolerance")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    ref = commands.add_parser("reference")
    ref.add_argument("archive", type=Path)
    ref.add_argument("--chunk", type=int, default=512)
    ref.add_argument("--holdouts", action="store_true", help="Record 24 additional positions on independent prompts")
    ref.add_argument("--greedy-holdouts", action="store_true", help="Record 128 positions using PyTorch's greedy continuations on eight prompts")
    long_ref = commands.add_parser("reference-long", help="Record global KV page boundaries and four cached decode steps")
    long_ref.add_argument("archive", type=Path)
    native = commands.add_parser("check")
    native.add_argument("archive", type=Path)
    native.add_argument("--kv-cache-dtype", choices=("bf16", "fp8"), default="bf16")
    args = parser.parse_args()
    if args.command == "reference":
        if args.chunk < 1:
            parser.error("chunk must be positive")
        reference(args.archive, args.chunk, args.holdouts, args.greedy_holdouts)
    elif args.command == "reference-long":
        reference(args.archive, 512, long_context=True)
    else:
        check(args.archive, args.kv_cache_dtype)
