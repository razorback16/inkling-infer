# inkling-infer

Run Inkling-2-Mini on Apple silicon with the project's native Metal inference library. Claude Code and Codex connect through a local LiteLLM proxy. The server defaults to a BF16 key/value cache and a 262,144-token window shared by the prompt and output.

## Setup

Use an Apple silicon Mac with BF16-capable Metal, macOS 14 or later, Xcode Command Line Tools, Python 3.12, `jq`, and `openssl`. Install the serving packages and create a private proxy key.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
(umask 077 && set -C && printf 'sk-%s\n' "$(openssl rand -hex 32)" > .proxy-key)
```

## Run

Start the server with `./serve`. It builds the native library with Clang and compiles its shaders on the Metal device. It uses the pinned Inkling-2-Mini safetensors checkpoint from the Hugging Face cache or downloads it on first use. To use an existing checkpoint directory, set `INKLING_MODEL_PATH`. The proxy listens on `127.0.0.1:4010` and the native backend on `127.0.0.1:8089`.

From another terminal, run `./inkling codex` or `./inkling claude` with the corresponding client installed. To use the launcher from another project, invoke it by its full path or symlink it into a directory on your `PATH`. For a smaller FP8 E4M3 cache, use `./serve --kv-cache-dtype fp8`.

## Runtime

The [native library](native/) owns Metal buffers, compute pipelines and dispatch. It implements dense projections, NVFP4 and MXFP8 expert projections, causal prefill, and cached attention. The [model runtime](inkling_native/) loads checkpoint tensors, runs the recurrent layers, and manages generation and prefix reuse. Metal handles normalization and expert routing. At load time, Metal packs the vocabulary projection into six-bit weights with a scale and zero point per 32 weights. During inference, each GPU lane combines eight consecutive weights with their activations and accumulates in FP32. NumPy stages tokens and reads back the logits. One request uses the inference queue at a time.

During greedy generation, Metal starts the next token's transformer layers while the caller receives the current logits. The runtime checks the token and its KV cache before reusing that work. Requests that sample tokens or adjust scores with penalties or logit bias run each chosen token directly.

The serving install requires neither MLX nor MLX-LM. The earlier MLX implementation and its benchmarks are available for development comparisons. The [native benchmark results](benchmarks/native-metal.json) include synchronized BF16 comparisons against MLX with the custom kernels, a separate streaming benchmark, and full-vocabulary error against PyTorch CPU on the same checkpoint.

To measure the native implementation, run `.venv/bin/python native_benchmark.py` after the library has been built. It times a warm 4,096-token prefill and 128 cached decode steps across three repetitions.

Run `.venv/bin/python native_guard.py` to compare both Metal attention paths with a CPU reference and check cache rollover and branching.

To check full-model accuracy, generate the reference once in a separate PyTorch CPU environment with `python native_logit_guard.py reference .build/torch-logits.npz`. Then run `.venv/bin/python native_logit_guard.py check .build/torch-logits.npz` in the serving environment. The check compares all 200,058 logits at each of 21 positions and requires relative L2 error of 2% or less at every position. Add `--holdouts` to the reference command to include 24 more positions across three independent prompts, including variable prefill chunks and the transition to cached decode. Add `--greedy-holdouts` to check another 128 positions across eight PyTorch-generated continuations. For global cache growth, run `python native_logit_guard.py reference-long .build/torch-long-pages.npz` in the CPU environment and check that archive with the same native command. It compares ten positions from 6,144 through 8,196 tokens, including four greedy steps after an 8,192-token prompt.
