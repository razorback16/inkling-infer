"""Owned Metal buffers and the kernel calls used by the Inkling text model."""
import ctypes as C
import weakref
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PTR = C.c_void_p
UINT = C.c_uint32


class CacheCopy(C.Structure):
    _fields_ = [("source", PTR), ("destination", PTR),
                ("source_offset", C.c_size_t), ("length", C.c_size_t)]


class Buffer:
    def __init__(self, gpu, shape, dtype):
        self.dtype = np.dtype(dtype)
        self.shape = tuple(shape)
        size = int(np.prod(shape)) * self.dtype.itemsize
        self.handle = gpu.lib.ink_buffer(gpu.handle, size)
        if not self.handle:
            raise MemoryError(gpu.error())
        backing = (C.c_ubyte * size).from_address(gpu.lib.ink_contents(self.handle))
        # Tie the allocation to all NumPy views, including detached slices.
        backing._release = weakref.finalize(backing, gpu.lib.ink_release, self.handle)
        self.array = np.frombuffer(backing, self.dtype).reshape(shape)


class Metal:
    def __init__(self):
        self.lib = C.CDLL(str(ROOT / ".build/native/libinkling.dylib"))
        signatures = {
            "ink_open": (PTR, []), "ink_error": (C.c_char_p, []),
            "ink_release": (None, [PTR]), "ink_buffer": (PTR, [PTR, C.c_size_t]),
            "ink_supports_gate_publication": (C.c_int, [PTR]),
            "ink_contents": (PTR, [PTR]),
            "ink_copy_cache": (C.c_int, [PTR, C.POINTER(CacheCopy), C.c_size_t]),
            "ink_run": (C.c_int, [PTR, C.c_char_p, C.POINTER(PTR), UINT, C.POINTER(UINT),
                                  UINT, C.POINTER(UINT), C.POINTER(UINT), C.c_int]),
        }
        for name, (result, args) in signatures.items():
            function = getattr(self.lib, name)
            function.restype, function.argtypes = result, args
        self.handle = self.lib.ink_open()
        if not self.handle:
            raise RuntimeError(self.error())
        self.supports_gate_publication = bool(self.lib.ink_supports_gate_publication(self.handle))
        self._release = weakref.finalize(self, self.lib.ink_release, self.handle)

    def error(self):
        return self.lib.ink_error().decode()

    def upload(self, data):
        buf = Buffer(self, data.shape, data.dtype)
        buf.array[...] = data
        return buf

    def copy_cache(self, copies):
        ops = (CacheCopy * len(copies))(*(CacheCopy(src.handle, dst.handle, offset, size)
                                         for src, dst, offset, size in copies))
        if self.lib.ink_copy_cache(self.handle, ops, len(ops)):
            raise RuntimeError(self.error())

    def run(self, name, buffers, args, groups, threads, bits=-1):
        handles = (PTR * len(buffers))(*(b.handle if b else None for b in buffers))
        result = self.lib.ink_run(self.handle, name.encode(), handles, len(buffers),
                                  (UINT * len(args))(*args), len(args),
                                  (UINT * 3)(*groups), (UINT * 3)(threads, 1, 1), bits)
        if result:
            raise RuntimeError(self.error())

    def linear(self, x, w):
        rows, width = w.shape
        if x.shape[-1] != width or w.dtype != np.uint16:
            raise ValueError(f"Incompatible projection {x.shape}, {w.shape}, {w.dtype}")
        x = self.upload(np.asarray(x, np.float32).reshape(-1, width))
        out = Buffer(self, (x.shape[0], rows), np.float32)
        self.run("linear", [x, w, out], [rows, width], ((rows + 7) // 8, x.shape[0], 1), 256)
        return out.array

    def expert_tables(self):
        if not hasattr(self, "_expert_tables"):
            packed = Path(__file__).with_name("swiglu.f32.zlib").read_bytes()
            activation = np.frombuffer(zlib.decompress(packed), dtype="<f4").reshape(65536, 2)
            code = np.arange(256)
            magnitude = code % 128
            values = np.where(magnitude < 8, magnitude / 512.,
                              (1 + magnitude % 8 / 8) * 2. ** (magnitude // 8 - 7))
            values[magnitude == 127] = np.nan
            values[code >= 128] *= -1
            with np.errstate(all="ignore"):
                codes = to_bits(np.exp2(code[:, None] - 127) * values)
            self._expert_tables = self.upload(activation), self.upload(codes)
        return self._expert_tables

    def experts(self, x, ids, routing, gate, down):
        length = len(x)
        if x.shape != (length, 2304) or ids.shape != (length, 8) or routing.shape != ids.shape:
            raise ValueError("Invalid expert input or routing shape")
        active = Buffer(self, (length, 8, 1152), np.uint16)
        out = Buffer(self, (length, 2304), np.uint16)
        indices = self.upload(ids.astype(np.uint32))
        routes = self.upload(routing.astype(np.float32))
        if np.any(ids < 0) or np.any(ids >= 128):
            raise ValueError("Expert indices must be in the checkpoint's 128-expert bank")
        activation, codes = self.expert_tables()
        self.run("selected_gate", [self.upload(to_bits(x)), indices, *gate, codes, activation, active],
                 [], (144, 8, length), 128, 4 if gate[2] else 8)
        self.run("selected_down", [active, indices, *down, codes, routes, out],
                 [], (576, length, 1), 256, 4 if down[2] else 8)
        return from_bits(out.array)

    def attend(self, q, keys, values, bias, sinks, past, window):
        if (q.shape != (len(q), 24, 128) or bias.shape != (len(q), 24, 1024)
                or sinks.shape != (len(q), 24) or keys.shape != values.shape
                or keys.shape[1:] != (8, 128) or past < 0 or past + len(q) > keys.shape[0]):
            raise ValueError("Invalid attention dimensions")
        out = Buffer(self, q.shape, np.float32)
        buffers = [self.upload(q), keys, values, self.upload(bias), self.upload(sinks), out]
        if len(q) == 1:
            self.run("attention", buffers, [past, window, keys.dtype == np.uint8, bias.shape[-1]],
                     (24, 1, 1), 1024)
        else:
            self.run("prefill_attention", buffers, [len(q), past + len(q), window, keys.dtype == np.uint8],
                     ((len(q) + 31) // 32, 24, 1), 128)
        return out.array


def to_bits(x):
    bits = np.asarray(x, np.float32).view(np.uint32)
    return ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def from_bits(x):
    return (x.astype(np.uint32) << 16).view(np.float32)


def bf16(x):
    return from_bits(to_bits(x))
