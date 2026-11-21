"""Configure upstream MLX-LM's stream parser for Inkling's message grammar."""
import json
import os

from mlx_lm import server
from mlx_lm.tokenizer_utils import TokenizerWrapper

original_init = TokenizerWrapper.__init__


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
    TokenizerWrapper.__init__ = initialize
    if os.getenv("INKLING_CUSTOM_KERNEL") == "1":
        from inkling_kernel import install
        install()
    server.main()
