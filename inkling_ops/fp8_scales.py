"""Exact Metal E4M3 scale lookup for the routed FP4 weights."""
import struct


def _fp32_bits(byte):
    magnitude = byte & 127
    half = (magnitude << 7) | (((magnitude + 1) >> 7) << 14) | ((byte & 128) << 8)
    value = struct.unpack(">e", struct.pack(">H", half))[0] * 256.0
    return struct.unpack(">I", struct.pack(">f", value))[0]


SCALE_HEADER = "constant uint inkling_scale_bits[256]={" + ",".join(
    f"0x{_fp32_bits(byte):08x}u" for byte in range(256)
) + "}; inline float decode_scale(uchar byte) {return as_type<float>(inkling_scale_bits[byte]);}"
