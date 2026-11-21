"""Retain native BF16 sources for projections that the Inkling loader widens."""
import mlx.core as mx
from mlx_lm.models.inkling2 import checkpoint
from mlx_lm.models.inkling2.model import Model

_reference = Model.sanitize


def sanitize(model, weights):
    arrays = _reference(model, weights)
    modules = dict(model.named_modules())
    for name, raw in weights.items():
        if raw.dtype != mx.bfloat16 or not name.endswith((".mlp.gate.weight", ".forget_weight", ".gain_weight",
                ".forget_gate_proj.weight", ".message_gate_proj.weight")):
            continue
        path = checkpoint._rename(name) if name.startswith("model.llm.") else name
        module = modules.get(path.removesuffix(".weight"))
        expanded = arrays.get(path)
        if module is not None and expanded is not None and expanded.dtype == mx.float32:
            object.__setattr__(module, "_native_gate", (expanded, raw))
    return arrays


def native(proj):
    ref = getattr(proj, "_native_gate", None)
    if ref is not None and ref[0] is proj.weight:
        return ref[1]
    return proj.weight


def install():
    Model.sanitize = sanitize
