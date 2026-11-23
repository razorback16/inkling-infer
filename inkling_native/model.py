# Original Inkling model implementation copyright © 2026 Apple Inc.
"""Inkling-2-Mini text inference. Layer equations follow the MIT-licensed
mlx-lm Inkling implementation at 0c4c991. See native/MLX-LICENSE.txt.
"""
import copy
import json
import math
import struct
from pathlib import Path

import numpy as np

from .metal import Buffer, Metal, bf16, from_bits, to_bits


def rms(x):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6)


def clip_norm(x, scale, threshold):
    x = x * scale
    norm = np.sqrt((x * x).sum(axis=-1, keepdims=True))
    return bf16(x * np.minimum(threshold / (norm + 1e-6), 1.0))


def softmax(x, axis=-1):
    weights = np.exp(x - x.max(axis=axis, keepdims=True))
    return weights / weights.sum(axis=axis, keepdims=True)


def depth(states, queries):
    # [tokens, sources, width] and [queries, width].
    bank = np.stack(states, axis=1)
    scores = (bank @ queries.reshape(-1, 2304).T) / 48
    return softmax(scores, axis=1).swapaxes(1, 2) @ bank


def sigmoid(x):
    tail = np.exp(-np.abs(x))
    return np.where(x >= 0, 1 / (1 + tail), tail / (1 + tail))


def fp8_table():
    # BF16 bit patterns to saturating E4M3FN, round to nearest, ties to even.
    codes = np.arange(127)
    levels = np.where(codes < 8, codes / 512, (1 + (codes % 8) / 8) * 2.0 ** (codes // 8 - 7))
    raw = np.arange(65536, dtype=np.uint16)
    values = np.abs(from_bits(raw & 0x7fff)).clip(0, 448)
    values = np.nan_to_num(values, nan=448).astype(np.float64)
    hi = np.searchsorted(levels, values).clip(0, 126)
    lo = np.maximum(hi - 1, 0)
    dl, dh = values - levels[lo], levels[hi] - values
    code = np.where((dl < dh) | ((dl == dh) & (lo % 2 == 0)), lo, hi)
    return (code | ((raw >> 8) & 128)).astype(np.uint8)


FP8 = fp8_table()


class VisitCache:
    def __init__(self, gpu, window, dtype):
        self.gpu, self.window, self.dtype = gpu, window, dtype
        self.length = self.position = 0
        self.keys = self.values = None
        self.written = [0]
        self.history = [None] * 4
        self.graph_history = self.graph_spare = None
        self.shared_history = False
        self.predict_next = True

    def fork(self):
        # Siblings share the written extent of their backing KV buffers.
        branch = copy.copy(self)
        branch.history = self.history.copy()
        branch.graph_spare = None
        self.shared_history = branch.shared_history = True
        return branch

    def reserve(self, count, copies=None):
        keep = min(self.length, self.window - 1) if self.window else self.length
        capacity = self.keys.shape[0] if self.keys else 0
        # Claim the suffix before encoding a write. A sibling may have already
        # appended past our prefix, in which case we need independent storage.
        if self.length < self.written[0] or self.length + count > capacity:
            if self.length + count <= capacity:
                new_size = capacity
            elif self.window:
                # Leave room for the next decode page after a multi-row append.
                # A window rollover may otherwise copy the whole window a token later.
                new_size = max(self.window + 128, keep + count + 128)
            else:
                # Reserve a page beyond a prompt-sized allocation so that reading
                # its first generated token can reuse the existing KV buffers.
                needed = keep + count
                new_size = max(128, needed + (128 if count > 1 else 0),
                               capacity + min(capacity, 2048))
            new_k = Buffer(self.gpu, (new_size, 8, 128), self.dtype)
            new_v = Buffer(self.gpu, new_k.shape, self.dtype)
            if keep and copies is not None:
                row_bytes = 8 * 128 * self.keys.dtype.itemsize
                offset = (self.length - keep) * row_bytes
                copies.extend([(self.keys, new_k, offset, keep * row_bytes),
                               (self.values, new_v, offset, keep * row_bytes)])
            elif keep:
                new_k.array[:keep] = self.keys.array[self.length - keep:self.length]
                new_v.array[:keep] = self.values.array[self.length - keep:self.length]
            self.keys, self.values, self.length = new_k, new_v, keep
            self.written = [keep]
        self.written[0] = self.length + count
        return self.length

    def append(self, keys, values):
        count = len(keys)
        past = self.reserve(count)
        sl = slice(past, past + count)
        k, v = to_bits(keys), to_bits(values)
        self.keys.array[sl] = FP8[k] if self.dtype == np.uint8 else k
        self.values.array[sl] = FP8[v] if self.dtype == np.uint8 else v
        self.length += count
        self.position += count
        return past


class Weights:
    def __init__(self, path, gpu):
        index = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
        self.tensors, self.buffers, self.small, self.banks = {}, {}, {}, {}
        types = {"F32": "<f4", "BF16": "<u2", "F8_E4M3": "u1", "U8": "u1", "F8_E8M0": "u1"}
        for filename in set(index.values()):
            file = path / filename
            with file.open("rb") as handle:
                size, = struct.unpack("<Q", handle.read(8))
                header = json.loads(handle.read(size))
            mapping = np.memmap(file, dtype=np.uint8, mode="r")
            for key, info in header.items():
                if not key.startswith(("model.llm.", "model.visual.")) or index.get(key) != filename:
                    continue
                a, b = info["data_offsets"]
                dtype = np.dtype(types[info["dtype"]])
                if a < 0 or b < a or size + 8 + b > mapping.size or b-a != math.prod(info["shape"]) * dtype.itemsize:
                    raise ValueError(f"Invalid checkpoint tensor {key}")
                self.tensors[key.removeprefix("model.").removeprefix("llm.")] = np.ndarray(
                    info["shape"], dtype, mapping, offset=size + 8 + a)
        self.gpu = gpu

    def array(self, name):
        if name not in self.small:
            a = self.tensors[name]
            self.small[name] = from_bits(a) if a.dtype == np.uint16 else a
        return self.small[name]

    def buffer(self, name):
        if name not in self.buffers:
            self.buffers[name] = self.gpu.upload(self.tensors[name])
        return self.buffers[name]

    def output_projection(self):
        key=("unembed.weight","affine6_output")
        if key not in self.buffers:
            blocks=self.tensors["unembed.weight"].reshape(-1,256)
            data=Buffer(self.gpu,(len(blocks),192),np.uint8)
            scales=Buffer(self.gpu,(len(blocks),8),np.uint16)
            chunk_size=65536
            staging=Buffer(self.gpu,(chunk_size,256),np.uint16)
            for start in range(0,len(blocks),chunk_size):
                count=min(len(blocks)-start,chunk_size)
                staging.array[:count]=blocks[start:start+count]
                self.gpu.run("pack_affine6_vocabulary",[staging,data,scales],[start,count],
                             ((count+7)//8,1,1),256)
            self.buffers[key]=(data,scales)
        return self.buffers[key]

    def depth_query_streams(self, name):
        # Separate the final product of each stream. The CPU adds it after
        # the ordered multiply-accumulates in each 512-channel panel.
        key = (name, "depth_query_streams")
        if key not in self.buffers:
            query = self.array(name).reshape(-1, 2304)[0]
            streams = np.zeros((20, 128), np.float32)
            tails = np.empty(20, np.float32)
            for panel in range(5):
                chunk = query[panel*512:(panel+1)*512].reshape(-1, 4).T
                rows = slice(4*panel, 4*panel+4)
                streams[rows, :chunk.shape[1]-1] = chunk[:, :-1]
                tails[rows] = chunk[:, -1]
            self.buffers[key] = self.gpu.upload(np.concatenate((streams.ravel(), tails)))
        return self.buffers[key]

    def prompt_controller_weights(self, name):
        key = (name, "prompt_controller")
        if key not in self.buffers:
            weights = self.tensors[name].reshape(8, 8, 18, 8, 32)
            tiles = weights.transpose(0, 4, 2, 3, 1)
            lanes = np.arange(32)
            r = ((lanes // 4) & 4) + (lanes // 2) % 4
            c = ((lanes // 4) & 2) * 2 + (lanes % 2) * 2
            packed = tiles[:, :, :, r[:, None], c[:, None] + [0, 1]]
            self.buffers[key] = self.gpu.upload(np.ascontiguousarray(packed))
        return self.buffers[key]

    def prompt_controller_gates(self, prefix):
        key = (prefix, "paired_gates")
        if key not in self.buffers:
            lane = np.arange(32)
            k = ((lane // 4) & 4) + (lane // 2) % 4
            n = ((lane // 4) & 2) * 2 + (lane % 2) * 2
            fragments = []
            for kind in ("forget_weight", "gain_weight"):
                raw = self.tensors[prefix + kind].reshape(288, 8, 8, 8)
                tiles = raw.transpose(0, 2, 3, 1)
                fragments.append(tiles[:, :, k[:, None], n[:, None] + [0, 1]])
            self.buffers[key] = self.gpu.upload(np.ascontiguousarray(np.concatenate(fragments, axis=-1)))
        return self.buffers[key]

    def paired_blend_weights(self,prefix):
        key=(prefix,"blend_rms")
        if key not in self.buffers:
            tile,sg,step,lane=np.ogrid[:40,:8,:4,:32]
            rank=sg*8+lane//4
            c=(tile//8)*512+(rank//4)*32+(tile%8)*4+rank%4
            k=step*16+(lane%4)*4
            fragments=[]
            for name in ("forget_weight","gain_weight"):
                w=self.tensors[prefix+name]
                fragments.append(w[np.minimum(c,2303)[...,None],k[...,None]+np.arange(4)])
            self.buffers[key]=self.gpu.upload(np.ascontiguousarray(np.stack(fragments,axis=-2)))
        return self.buffers[key]

    def controller_stream_weights(self, name):
        key = (name, "controller_streams")
        if key not in self.buffers:
            weights = self.tensors[name].reshape(64, 18, 8, 4, 8)
            lanes = np.arange(32)
            r = ((lanes // 4) & 4) + (lanes // 2) % 4
            c = ((lanes // 4) & 2) * 2 + (lanes % 2) * 2
            panels = weights.transpose(0, 3, 1, 4, 2)
            packed = panels[:, :, :, r[:, None], c[:, None] + [0, 1]]
            self.buffers[key] = self.gpu.upload(np.ascontiguousarray(packed))
        return self.buffers[key]

    def repair_projection(self, name):
        key=(name,"repair_vectors")
        if key not in self.buffers:
            raw=self.tensors[name]
            rows,width=raw.shape
            vectors=raw.reshape(rows,width//128,4,32).transpose(0,1,3,2)
            self.buffers[key]=self.gpu.upload(np.ascontiguousarray(vectors))
        return self.buffers[key]

    def prefill_projection(self, name):
        key=(name,"prefill_fragments")
        if key not in self.buffers:
            raw=self.tensors[name]
            rows,width=raw.shape
            lane=np.arange(32)
            kr=((lane//4)&4)+(lane//2)%4
            nc=((lane//4)&2)*2+(lane%2)*2
            packed=Buffer(self.gpu,((rows+31)//32,width//8,4,32,2),raw.dtype)
            for begin in range(0,rows,32):
                block=np.zeros((32,width),raw.dtype)
                live=min(32,rows-begin)
                block[:live]=raw[begin:begin+live]
                tiles=block.reshape(4,8,width//8,8).transpose(2,0,3,1)
                packed.array[begin//32]=tiles[:,:,kr[:,None],nc[:,None]+[0,1]]
            self.buffers[key]=packed
        return self.buffers[key]

    def decode_router(self, name):
        key=(name,"decode_router")
        if key not in self.buffers:
            raw=self.tensors[name]
            matrix=raw.reshape(128,9,4,2,32).transpose(0,1,4,2,3)
            self.buffers[key]=self.gpu.upload(np.ascontiguousarray(matrix))
        return self.buffers[key]

    def paired_high_bytes(self, name, *, four_rows=False):
        """Pack adjacent BF16 high bytes into one lossless index per pair."""
        key = (name, "paired_high_bytes", four_rows)
        if key in self.buffers:
            return self.buffers[key]
        raw = self.tensors[name]
        rows, width = raw.shape
        group_rows = 32
        table_count = (rows + group_rows - 1) // group_rows
        payload_size = rows * width * 3 // 2
        out = Buffer(self.gpu, (payload_size + 1024 * table_count,), np.uint8)
        panel_rows = 4 if four_rows else 1
        panels = out.array[:payload_size].reshape(rows//panel_rows, width//128, 192*panel_rows)
        tables = out.array[payload_size:].view(np.uint32).reshape(-1, 256)
        lookup = np.empty(65536, np.uint8)
        for begin in range(0, rows, group_rows):
            block = raw[begin:begin+group_rows].reshape(-1, width//128, 4, 32).transpose(0,1,3,2)
            high = np.ascontiguousarray(block >> 8, dtype=np.uint8).view(np.uint16)
            pairs = np.flatnonzero(np.bincount(high.ravel(), minlength=65536))
            if len(pairs) > 256:
                self.buffers[key] = None
                return None
            lookup[pairs] = np.arange(len(pairs), dtype=np.uint8)
            words = pairs.astype(np.uint32)
            tables[begin//group_rows, :len(pairs)] = ((words & 255) << 8) | ((words >> 8) << 24)
            # In a four-row dot each lane consumes its four rows together.
            # The output residual uses one output row per SIMD group.
            panel = panels[begin//panel_rows:(begin+len(block))//panel_rows]
            low = block.astype(np.uint8).reshape(-1,panel_rows,width//128,32,4).transpose(0,2,3,1,4)
            indices = lookup[high].reshape(-1,panel_rows,width//128,32,2).transpose(0,2,3,1,4)
            panel[:, :, :128*panel_rows] = low.reshape(len(panel),width//128,128*panel_rows)
            panel[:, :, 128*panel_rows:] = indices.reshape(len(panel),width//128,64*panel_rows)
        self.buffers[key] = out
        return out

    def decode_projection(self, name):
        key = (name, "lossless")
        if key in self.buffers:
            return self.buffers[key]
        raw = self.tensors[name].reshape(-1, 128)
        count = len(raw)
        # Each common block stores the sign, mantissa, and four exponent bits.
        # Blocks with a wider exponent range remain verbatim in the exception bank.
        aligned = (count + 1) & ~1
        payload = np.zeros((aligned // 2, 32, 3), np.uint32)
        headers = np.empty(count, np.uint32)
        exceptions = []
        cursor = 0
        for start in range(0, count, 16384):
            rows = raw[start:start+16384].reshape(-1, 4, 32).transpose(0, 2, 1)
            exponent = (rows >> 7) & 255
            base = exponent.min(axis=(1, 2))
            delta = exponent - base[:, None, None]
            wide = delta.max(axis=(1, 2)) > 15
            h = base.astype(np.uint32)
            n = int(wide.sum())
            if n:
                h[wide] = 0x80000000 + np.arange(cursor, cursor+n, dtype=np.uint32)
                cursor += n
                exceptions.append(rows[wide].copy())
            headers[start:start+len(rows)] = h
            mant = np.ascontiguousarray(((rows >> 8) & 128) | (rows & 127), dtype=np.uint8).view(np.uint32)[:,:,0]
            block = payload[start//2:(start+len(rows)+1)//2]
            block[:,:,0] = mant[::2]
            block[:len(rows)//2,:,1] = mant[1::2]
            packed = delta[:,:,0] | (delta[:,:,1]<<4) | (delta[:,:,2]<<8) | (delta[:,:,3]<<12)
            block[:,:,2] = packed[::2]
            block[:len(rows)//2,:,2] |= packed[1::2].astype(np.uint32) << 16
        rows,width = self.tensors[name].shape
        if width == 2304 and rows % 4 == 0:
            # Two output rows share each pair of input blocks. The SIMD loads
            # four mantissa words and two exponent words per lane.
            src=payload.reshape(rows//2,2,9,32,3).transpose(0,2,3,1,4)
            tiles=np.empty((rows//2,9,192),np.uint32)
            tiles[:,:,:128]=src[:,:,:,:,:2].reshape(rows//2,9,128)
            tiles[:,:,128:]=src[:,:,:,:,2].reshape(rows//2,9,64)
            payload=tiles
        buf = Buffer(self.gpu, (192*aligned + 4*count + 256*cursor,), np.uint8)
        buf.array[:192*aligned] = payload.reshape(-1).view(np.uint8)
        buf.array[192*aligned:192*aligned+4*count] = headers.view(np.uint8)
        offset = 192*aligned + 4*count
        for chunk in exceptions:
            view = chunk.reshape(-1).view(np.uint8)
            buf.array[offset:offset+view.size] = view
            offset += view.size
        self.buffers[key] = buf
        return self.buffers[key]

    def expert(self, prefix):
        if prefix in self.banks:
            return self.banks[prefix]
        width = 2304 if prefix.endswith("w13_weight") else 1152
        nvfp4 = prefix + "_amax" in self.tensors
        packed_tensor = self.tensors[prefix]
        scale_tensor = self.tensors[prefix + "_scale"]
        if (packed_tensor.shape != (128, 2304, width // (2 if nvfp4 else 1))
                or scale_tensor.shape != (128, 2304, width // (16 if nvfp4 else 32))
                or packed_tensor.dtype != np.uint8 or scale_tensor.dtype != np.uint8):
            raise ValueError(f"Invalid expert bank {prefix}")
        source = self.gpu.upload(packed_tensor)
        source_scales = self.gpu.upload(scale_tensor)
        packed = Buffer(self.gpu, source.shape, source.dtype)
        scales = Buffer(self.gpu, source_scales.shape, source_scales.dtype)
        blocks = packed_tensor.size // (64 if nvfp4 else 128)
        threads = blocks * (16 if nvfp4 else 32)
        self.gpu.run("pack_decode_bank", [source, source_scales, packed, scales], [blocks, width//128],
                     ((threads+255)//256,1,1), 256, 4 if nvfp4 else 8)
        norm = None
        if prefix + "_amax" in self.tensors:
            name = prefix + "_global_scale"
            if name not in self.buffers:
                # The upstream BF16 loader casts source scale tensors to the
                # parameter dtype before dequantizing the FP4 codes in FP32.
                factor = bf16(self.array(prefix + "_amax")) / bf16(self.array(prefix + "_outer_scale_denominator"))
                codes = np.arange(128, dtype=np.int32)
                values = np.where(codes < 8, codes / 512., (1 + codes % 8 / 8) * 2.**(codes//8-7))
                levels = np.array([0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6], np.float32)
                table = values.astype(np.float32)[None,:,None] * levels * factor[:,None,None]
                self.buffers[name] = self.gpu.upload(to_bits(table))
            norm = self.buffers[name]
        self.banks[prefix] = packed, scales, norm
        return self.banks[prefix]


class TextModel:
    def __init__(self, path, cache_dtype="bf16"):
        path = Path(path)
        config = json.loads((path / "config.json").read_text())["text_config"]
        if (config["hidden_size"], config["num_hidden_layers"], config["num_experts_per_tok"]) != (2304, 24, 8):
            raise ValueError("Expected an Inkling-2-Mini checkpoint")
        self.gpu = Metal()
        self.w = Weights(path, self.gpu)
        self.local = set(config["local_layer_ids"])
        self.visits = list(range(6)) + list(range(6, 18)) * 3 + list(range(18, 24))
        self.cache_dtype = {"fp8": np.uint8, "bf16": np.uint16}[cache_dtype]
        self.vocab = config["unpadded_vocab_size"]
        q = self.w.array("middle_cycle_cross_pass_queries").reshape(12, 2, -1)
        kv = self.w.array("middle_cycle_cross_pass_kv_queries").reshape(12, 2, -1)
        self.messages = np.concatenate((q[:, :1], kv, q[:, 1:]), axis=1)
        from .graph import make_tables
        self.tables = make_tables(self, FP8)
        self.plans = {}

    def cache(self):
        return [VisitCache(self.gpu, 1024 if i in self.local else 0, self.cache_dtype) for i in self.visits]

    def linear(self, x, name, rounded=True):
        y = self.gpu.linear(x, self.w.buffer(name))
        return bf16(y) if rounded else y

    def forward(self, tokens, cache, *, logits=True, image_rows=None):
        if not len(tokens):
            raise ValueError("Expected at least one input token")
        # A 24-head BF16 score panel needs 48 bytes per query/key pair.
        # Bound temporary memory as cached context grows. Powers of two
        # let successive chunks reuse the same dispatch schedule.
        positions = cache[0].position + len(tokens)
        max_rows = max(1, (8 * 1024 * 1024) // ((positions + 31) // 32 * 32))
        if len(tokens) > max_rows:
            step = 1 << (max_rows.bit_length() - 1)
            result = None
            for start in range(0, len(tokens), step):
                end = min(start + step, len(tokens))
                selected = [(row - start, value) for row, value in image_rows or ()
                            if start <= row < end]
                result = self.forward(tokens[start:end], cache,
                                      logits=logits and end == len(tokens),
                                      image_rows=selected)
            return result
        from .graph import ForwardPlan
        key = (len(tokens), bool(logits))
        if key != (1, True) and (1, True) in self.plans:
            self.plans[1, True].discard_prediction()
        if key not in self.plans:
            # Keep the hot decode plan when the prompt ends in a partial chunk.
            prompt_plans = [old for old in self.plans if old[0] > 1]
            if len(tokens) > 1 and len(prompt_plans) >= 2:
                del self.plans[prompt_plans[0]]
            self.plans[key] = ForwardPlan(self, *key)
        return self.plans[key].run(tokens, cache, image_rows)
