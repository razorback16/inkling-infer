# inkling-infer

Run Inkling-2-Mini on Apple silicon with MLX and connect Claude Code or Codex through a local LiteLLM proxy. The server defaults to custom Metal kernels, an FP8 E4M3 key/value cache, and a 262,144-token window shared by the prompt and output.

## Setup

Use Python 3.12 on an Apple silicon Mac with `jq` and `openssl` on your path. From a fresh checkout, install the Inkling-compatible MLX-LM build and create a private key for the local proxy.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install 'mlx==0.32.4' 'transformers==5.23.0' \
  'mlx-lm @ https://github.com/jundot/mlx-lm/archive/0c4c99192230f1943dd468d297f0b8a64ae7b7c2.zip' \
  'litellm[proxy]==1.110.2'
(umask 077 && set -C && printf 'sk-%s\n' "$(openssl rand -hex 32)" > .proxy-key)
```

## Run

Start the server with `./serve`. MLX-LM loads the weights for `thinkingmachines/Inkling-2-Mini` from the Hugging Face cache or downloads them on first use. The proxy listens on `127.0.0.1:4010` and the MLX backend on `127.0.0.1:8089`.

From another terminal, run `./inkling codex` or `./inkling claude` with the corresponding client installed. To use the launcher from another project, invoke it by its full path or symlink it into a directory on your `PATH`.

Use `./serve --kv-cache-dtype bf16` for a BF16 cache. Recorded benchmark settings and measurements are in [benchmarks](benchmarks/).
