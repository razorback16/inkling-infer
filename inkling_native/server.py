"""OpenAI-compatible HTTP serving for the native Inkling text runtime."""
import argparse
import json
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from huggingface_hub import snapshot_download
from tokenizers.decoders import DecodeStream
from transformers import PreTrainedTokenizerFast

from .generation import ChatParser, Generator, Sampler, StopText
from .model import TextModel
from .vision import IMAGE_TOKEN, read_image

MODEL = "thinkingmachines/Inkling-2-Mini"
REVISION = "ed40e7da625f594fc9c4990f5f66be24873c22d7"


def checkpoint():
    if path := os.environ.get("INKLING_MODEL_PATH"):
        return Path(path)
    args = dict(repo_id=MODEL, revision=REVISION, allow_patterns=[
        "model-*.safetensors", "model.safetensors.index.json", "tokenizer*.json", "config.json",
        "chat_template.jinja", "special_tokens_map.json"])
    try:
        path = Path(snapshot_download(**args, local_files_only=True))
        if all((path / file).is_file() for file in ("model-00001-of-00001.safetensors", "tokenizer.json",
                                                     "chat_template.jinja", "model.safetensors.index.json")):
            return path
    except FileNotFoundError:
        pass
    return Path(snapshot_download(**args))


class Service:
    def __init__(self, path, cache_dtype, window, chunk=2048, template_options=None):
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(str(path), local_files_only=True)
        self.generator = Generator(TextModel(path, cache_dtype), chunk)
        self.template_options = {"reasoning_effort": "low", **(template_options or {})}
        self.window = window
        self.lock = threading.Lock()

    def prompt(self, body, chat):
        if chat:
            messages = body["messages"]
            if not isinstance(messages, list) or not messages:
                raise ValueError("messages must be a nonempty list")
            images = []
            for message in messages:
                parts = message.get("content")
                if isinstance(parts, list):
                    contains_image = False
                    for part in parts:
                        if part.get("type") in ("input_image", "image_url"):
                            src = part["image_url"]
                            images.append(read_image(src["url"] if isinstance(src, dict) else src))
                            contains_image = True
                        elif part.get("type") not in ("text", "input_text"):
                            raise ValueError("Unsupported content part")
                    if not contains_image:
                        message["content"] = "".join(part["text"] for part in parts)
                for call in message.get("tool_calls") or []:
                    fn = call["function"]
                    if isinstance(fn.get("arguments"), str):
                        fn["arguments"] = json.loads(fn["arguments"])
            options = self.template_options.copy()
            if "reasoning_effort" in body:
                options["reasoning_effort"] = body["reasoning_effort"]
            options.update(body.get("chat_template_kwargs") or {})
            tokens = self.tokenizer.apply_chat_template(messages, tools=body.get("tools"), tokenize=True,
                                                       add_generation_prompt=True, return_dict=False, **options)
            if not images:
                return tokens, []
            markers = [i for i, t in enumerate(tokens) if t == IMAGE_TOKEN]
            if len(markers) != len(images):
                raise ValueError("Image placeholders do not match the image inputs")
            expanded, spans, previous = [], [], 0
            for index, (digest, pixels, count) in zip(markers, images):
                expanded.extend(tokens[previous:index])
                spans.append((len(expanded), digest, pixels, count))
                expanded.extend([IMAGE_TOKEN] * count)
                previous = index + 1
            return expanded + tokens[previous:], spans
        prompt = body["prompt"]
        if isinstance(prompt, str):
            return self.tokenizer.encode(prompt), []
        if isinstance(prompt, list) and all(type(i) is int and 0 <= i < self.generator.model.vocab for i in prompt):
            return prompt, []
        raise ValueError("prompt must be text or token IDs")


class Handler(BaseHTTPRequestHandler):
    def json(self, status, body):
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path in ("/health", "/health/readiness"):
            self.json(200, {"status": "ok"})
        elif self.path == "/v1/models":
            self.json(200, {"object": "list", "data": [{"id": MODEL, "object": "model", "owned_by": "local"}]})
        else:
            self.send_error(404)

    def do_POST(self):
        chat = self.path in ("/v1/chat/completions", "/chat/completions")
        if not chat and self.path != "/v1/completions":
            self.send_error(404)
            return
        svc = self.server.service
        started = False
        try:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if not isinstance(body, dict):
                raise ValueError("Expected a JSON object")
            prompt, images = svc.prompt(body, chat)
            limit = body.get("max_completion_tokens", body.get("max_tokens", 512))
            if not prompt or type(limit) is not int or limit < 1 or len(prompt) + limit > svc.window:
                raise ValueError(f"Request uses {len(prompt)} prompt tokens and reserves {limit} output tokens "
                                 f"with a context window of {svc.window}")
            stops = body.get("stop") or []
            stops = [stops] if isinstance(stops, str) else stops
            if not isinstance(stops, list) or any(not isinstance(s, str) or not s for s in stops):
                raise ValueError("stop must contain nonempty strings")
            stopper = StopText(stops)
            sampler = Sampler(body, prompt, svc.tokenizer, svc.generator.model.vocab)
            token_logs = []
            stream = body.get("stream", False)
            model = body.get("model", MODEL)
            if model not in (MODEL, "inkling"):
                self.json(404, {"error": {"message": f"Unknown model {model}"}})
                return
            if body.get("n", 1) != 1:
                raise ValueError("n must be 1")
            base = dict(id=("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex,
                        created=int(time.time()), model=model)
            base["object"] = ("chat.completion.chunk" if stream else "chat.completion") if chat else "text_completion"

            def event(choices, **extra):
                data = json.dumps(dict(base, choices=choices, **extra))
                self.wfile.write(("data: " + data + "\n\n").encode())
                self.wfile.flush()

            def choice(delta, finish=None):
                payload = {"delta": delta} if chat else {"text": delta.get("content", "")}
                return dict(index=0, finish_reason=finish, **payload)

            message = {"role": "assistant", "content": ""}
            tokenizer = svc.tokenizer.backend_tokenizer
            parser = ChatParser(tokenizer) if chat else None
            decoder = DecodeStream(skip_special_tokens=False)
            with svc.lock:
                greedy = sampler.temperature == 0 and not (sampler.penalties or sampler.bias)
                cache, logits, cached = svc.generator.prepare(prompt, images, predict_next=greedy and limit > 1)
                if stream:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    started = True
                    if chat:
                        event([choice({"role": "assistant"})])
                reason = "length"
                for count in range(1, limit + 1):
                    token, log = sampler.draw(logits)
                    if log and not stream:
                        token_logs.append(log)
                    if token == svc.tokenizer.eos_token_id:
                        reason = "stop"
                        break
                    delta = parser.feed(token) if parser else {"content": decoder.step(tokenizer, token) or ""}
                    if "content" in delta:
                        delta["content"] = stopper.feed(delta["content"])
                    if delta or log:
                        for field, value in delta.items():
                            if field == "tool_calls":
                                message.setdefault(field, []).extend({k: v for k, v in c.items() if k != "index"}
                                                                     for c in value)
                            else:
                                message[field] = message.get(field, "") + value
                        if stream:
                            chunk = choice(delta)
                            if log:
                                chunk["logprobs"] = {"content": [log]}
                            event([chunk])
                    if stopper.stopped:
                        reason = "stop"
                        break
                    if count < limit:
                        cache[0].predict_next = greedy and count + 1 < limit
                        logits = svc.generator.model.forward([token], cache)
            if stopper.tail:
                message["content"] += stopper.tail
                if stream:
                    event([choice({"content": stopper.tail})])
            if parser and parser.calls and reason == "stop":
                reason = "tool_calls"
            usage = dict(prompt_tokens=len(prompt), completion_tokens=count, total_tokens=len(prompt)+count,
                         prompt_tokens_details={"cached_tokens": cached})
            if stream:
                event([choice({}, reason)])
                if (body.get("stream_options") or {}).get("include_usage"):
                    event([], usage=usage)
                self.wfile.write(b"data: [DONE]\n\n")
            else:
                result = {"message": message} if chat else {"text": message["content"]}
                if token_logs:
                    result["logprobs"] = {"content": token_logs}
                self.json(200, dict(base, choices=[dict(index=0, finish_reason=reason, **result)], usage=usage))
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:
            if started:
                event([], error={"message": str(exc)})
                self.wfile.write(b"data: [DONE]\n\n")
            else:
                from jinja2 import TemplateError
                status = 400 if isinstance(exc, (TypeError, KeyError, ValueError, TemplateError)) else 500
                self.json(status, {"error": {"message": str(exc)}})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    parser.add_argument("--chat-template-args", type=json.loads, default={})
    parser.add_argument("--port", type=int, default=8089)
    parser.add_argument("--ready-file", type=Path)
    args = parser.parse_args()
    if args.prefill_step_size < 1:
        parser.error("--prefill-step-size must be positive")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    path = checkpoint() if args.model == MODEL else Path(args.model)
    server.service = Service(path, os.getenv("INKLING_KV_CACHE_DTYPE", "bf16"),
                             int(os.getenv("INKLING_CONTEXT_WINDOW", "262144")), args.prefill_step_size,
                             args.chat_template_args)
    if args.ready_file:
        args.ready_file.write_text("ready\n")
    server.serve_forever()


if __name__ == "__main__":
    main()
