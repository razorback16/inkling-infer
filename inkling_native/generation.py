"""Prompt reuse, sampling and Inkling chat framing for the HTTP backend."""
import json
import uuid

import numpy as np
from tokenizers.decoders import DecodeStream


class Generator:
    def __init__(self, model, chunk=2048):
        self.model, self.chunk = model, chunk
        self.prefix, self.saved = [], None

    def prepare(self, prompt, images=(), *, predict_next=True):
        from .vision import encode_image
        # Include image identity in prefix matching. Equal patch counts can
        # represent different images, with identical tokenizer IDs.
        cache_key = list(prompt)
        for start, digest, pixels, count in images:
            cache_key[start:start+count] = [(digest, i) for i in range(count)]
        prefix = cache_key[:-1]
        if self.saved and len(prefix) >= len(self.prefix) and prefix[:len(self.prefix)] == self.prefix:
            cache = [c.fork() for c in self.saved]
            cached = len(self.prefix)
        else:
            cache, cached = self.model.cache(), 0
        cache[0].predict_next = predict_next
        rows = {}
        for start, digest, pixels, count in images:
            if start + count > cached:
                rows.update(enumerate(encode_image(self.model, pixels), start))
        for start in range(cached, len(prefix), self.chunk):
            end = min(len(prefix), start + self.chunk)
            selected = [(i-start, rows[i]) for i in range(start, end) if i in rows]
            self.model.forward(prompt[start:end], cache, logits=False, image_rows=selected)
        self.prefix, self.saved = prefix, [c.fork() for c in cache]
        final_image = [(0, rows[len(prefix)])] if len(prefix) in rows else None
        logits = self.model.forward(prompt[-1:], cache, image_rows=final_image)
        return cache, logits, cached


class Sampler:
    def __init__(self, body, prompt, tokenizer, vocab):
        self.temperature = body.get("temperature", 0.)
        self.top_p = body.get("top_p", 1.)
        self.min_p = body.get("min_p", 0.)
        self.top_k = body.get("top_k", 0)
        self.xtc = body.get("xtc_probability", 0.)
        self.threshold = body.get("xtc_threshold", .1)
        if (not np.isfinite(self.temperature) or self.temperature < 0 or not 0 <= self.top_p <= 1
                or not 0 <= self.min_p <= 1 or not 0 <= self.xtc <= 1 or not 0 <= self.threshold <= .5
                or type(self.top_k) is not int or not 0 <= self.top_k < vocab):
            raise ValueError("Invalid sampling parameters")
        self.history = list(prompt)
        self.penalties = []
        for name in ("repetition", "presence", "frequency"):
            amount = body.get(name + "_penalty", 0.)
            length = body.get(name + "_context_size", 20)
            if not np.isfinite(amount) or (name == "repetition" and amount < 0) or type(length) is not int or length < 0:
                raise ValueError(f"Invalid {name} penalty")
            if amount:
                self.penalties.append((name, amount, length))
        bias = body.get("logit_bias") or {}
        self.bias = [(int(i), float(value)) for i, value in bias.items()]
        if any(i < 0 or i >= vocab or not np.isfinite(v) for i, v in self.bias):
            raise ValueError("Invalid logit bias")
        self.rng = np.random.default_rng(body.get("seed"))
        self.exempt = tokenizer.encode("\n") + [tokenizer.eos_token_id]
        self.tokenizer = tokenizer
        self.top_logs = body.get("top_logprobs", -1)
        if type(self.top_logs) is not int or not -1 <= self.top_logs <= 11:
            raise ValueError("Invalid top_logprobs")
        self.report = body.get("logprobs", False) or self.top_logs > 0

    def draw(self, logits):
        scores = logits.astype(np.float64)
        for i, bias in self.bias:
            scores[i] += bias
        for name, amount, length in self.penalties:
            ids, count = np.unique(self.history[-length:], return_counts=True)
            if name == "repetition":
                scores[ids] = np.where(scores[ids] < 0, scores[ids] * amount, scores[ids] / amount)
            else:
                scores[ids] -= amount * (count if name == "frequency" else 1)
        if self.temperature == 0:
            token = int(scores.argmax())
        else:
            weights = np.exp(scores - scores.max())
            keep = np.ones(len(scores), dtype=bool)
            if 0 < self.top_p < 1:
                ordered = np.sort(weights)[::-1]
                cutoff = np.searchsorted(np.cumsum(ordered), self.top_p * weights.sum())
                keep &= weights >= ordered[min(cutoff, len(ordered)-1)]
            if self.min_p:
                keep &= weights >= self.min_p * weights.max()
            if self.xtc and self.rng.random() < self.xtc:
                eligible = weights > self.threshold * weights[keep].sum()
                if (keep & eligible).any():
                    remove = weights > weights[keep & eligible].min()
                    remove[self.exempt] = False
                    keep &= ~remove
            ids = np.flatnonzero(keep)
            if 0 < self.top_k < len(ids):
                ids = ids[np.argpartition(-scores[ids], self.top_k-1)[:self.top_k]]
            probs = np.exp((scores[ids] - scores[ids].max()) / self.temperature)
            token = int(self.rng.choice(ids, p=probs/probs.sum()))
        self.history.append(token)
        if not self.report:
            return token, None
        log_z = scores.max() + np.log(np.exp(scores - scores.max()).sum())
        def entry(i):
            text = self.tokenizer.decode([int(i)])
            return dict(id=int(i), token=text, bytes=list(text.encode()), logprob=float(scores[i] - log_z))
        info = entry(token)
        if self.top_logs > 0:
            top = np.argpartition(-scores, self.top_logs-1)[:self.top_logs]
            info["top_logprobs"] = [entry(i) for i in top[np.argsort(-scores[top])]]
        return token, info


class ChatParser:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.decoder = DecodeStream(skip_special_tokens=False)
        self.states = {tokenizer.token_to_id(f"<|{marker}|>"): state for marker, state in (
            ("message_model", "header"), ("end_message", "header"), ("content_text", "content"),
            ("content_thinking", "reasoning_content"), ("content_invoke_tool_json", "tool"))}
        self.state = "header"
        self.tool = ""
        self.calls = 0

    def feed(self, token):
        if token in self.states:
            delta = {}
            if self.state == "tool" and self.tool:
                call = json.loads(self.tool)
                delta["tool_calls"] = [dict(index=self.calls, id="call_" + uuid.uuid4().hex, type="function",
                                            function=dict(name=call["name"], arguments=json.dumps(call["args"])))]
                self.calls += 1
            self.state, self.tool = self.states[token], ""
            self.decoder = DecodeStream(skip_special_tokens=False)
            return delta
        piece = self.decoder.step(self.tokenizer, token)
        if self.state == "tool" and piece:
            self.tool += piece
        if self.state in ("content", "reasoning_content") and piece:
            return {self.state: piece}
        return {}


class StopText:
    """Hold only a suffix that could still complete a requested stop string."""
    def __init__(self, stops):
        self.stops, self.tail, self.stopped = stops, "", False

    def feed(self, piece):
        text = self.tail + piece
        hits = [i for stop in self.stops if (i := text.find(stop)) >= 0]
        if hits:
            self.tail, self.stopped = "", True
            return text[:min(hits)]
        hold = max((size for stop in self.stops for size in range(1, min(len(stop), len(text)+1))
                    if text.endswith(stop[:size])), default=0)
        self.tail = text[-hold:] if hold else ""
        return text[:-hold] if hold else text
