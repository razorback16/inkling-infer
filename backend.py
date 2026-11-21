"""Configure upstream MLX-LM's stream parser for Inkling's message grammar."""
import json
import os
from pathlib import Path
import sys

from mlx_lm import server
from mlx_lm.tokenizer_utils import TokenizerWrapper

original_init = TokenizerWrapper.__init__
original_tokenize = server.ResponseGenerator._tokenize
original_headers = server.APIHandler._set_completion_headers


class ContextLengthExceeded(ValueError):
    pass


def install_context_limit(window):
    def tokenize(generator, tokenizer, request, args):
        result = original_tokenize(generator, tokenizer, request, args)
        prompt_tokens = len(result[0])
        if prompt_tokens + args.max_tokens > window:
            raise ContextLengthExceeded(
                f"Maximum context length is {window} tokens. The prompt uses "
                f"{prompt_tokens} tokens and the request reserves {args.max_tokens} output tokens."
            )
        return result

    def set_headers(handler, status_code=200):
        # MLX-LM maps every pre-generation exception to 404, including invalid input.
        if status_code == 404 and isinstance(sys.exception(), ContextLengthExceeded):
            status_code = 400
        original_headers(handler, status_code)

    server.ResponseGenerator._tokenize = tokenize
    server.APIHandler._set_completion_headers = set_headers


def parse_tool_call(text, tools):
    payload = json.loads(text)
    return {"name": payload["name"], "arguments": payload["args"]}


def initialize(self, tokenizer, *args, **kwargs):
    if "<|content_invoke_tool_json|>" in tokenizer.get_vocab():
        kwargs.update(
            tool_parser=parse_tool_call,
            tool_call_start="<|content_invoke_tool_json|>",
            tool_call_end="<|end_message|>",
            response_format={
                "initial_state": "header",
                "transitions": {
                    "header": [
                        ("<|content_thinking|>", "reasoning"),
                        ("<|content_text|>", "normal"),
                        ("<|content_invoke_tool_json|>", "tool"),
                    ],
                    "reasoning": [("<|end_message|>", "header")],
                    "normal": [("<|end_message|>", "header")],
                },
            },
        )
    original_init(self, tokenizer, *args, **kwargs)


if __name__ == "__main__":
    if "INKLING_CONTEXT_WINDOW" in os.environ:
        window = int(os.environ["INKLING_CONTEXT_WINDOW"])
    else:
        catalog = json.loads(Path(__file__).with_name("inkling-model.json").read_text())
        window = catalog["models"][0]["context_window"]
    install_context_limit(window)
    TokenizerWrapper.__init__ = initialize
    cache_dtype = os.getenv("INKLING_KV_CACHE_DTYPE", "fp8")
    if cache_dtype == "fp8":
        from inkling_ops.fp8_cache import install
        install()
    elif cache_dtype != "bf16":
        raise SystemExit("INKLING_KV_CACHE_DTYPE must be bf16 or fp8")
    kernel = os.getenv("INKLING_CUSTOM_KERNEL", "1")
    if kernel == "1":
        from inkling_kernel import install
        install()
    elif kernel != "0":
        raise SystemExit("INKLING_CUSTOM_KERNEL must be 0 or 1")
    server.main()
