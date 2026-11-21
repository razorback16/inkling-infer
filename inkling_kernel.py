"""Single-token weighted mixture of eight Inkling expert outputs."""
import mlx.core as mx
from mlx_lm.models.inkling2 import layers

_reference = layers.SparseMoeBlock.__call__
_mix = mx.fast.metal_kernel(
    name="inkling_mix8",
    input_names=["states", "weights"],
    output_names=["out"],
    source="""
        uint c = thread_position_in_grid.x;
        volatile float value = 0;
        for (uint e = 0; e < 8; ++e) {
            volatile float term = float(states[e * 2304 + c]) * float(weights[e]);
            value = value + term;
        }
        out[c] = static_cast<T>(value);
    """,
)


def mix(states, weights, dtype):
    return _mix(inputs=[states, weights], template=[("T", dtype)],
                grid=(2304, 1, 1), threadgroup=(128, 1, 1),
                output_shapes=[(1, 1, 2304)], output_dtypes=[dtype])[0]


def single_token_moe(self, x):
    if x.shape != (1, 1, 2304) or self.gate.top_k != 8 or self.sharding_group is not None:
        return _reference(self, x)
    experts, weights = self.gate(x)
    return mix(self.experts(x, experts), weights, x.dtype)


def install():
    from inkling_ops import attention_mask, compiled_ring, dynamic_conv, message_gate, projections, short_conv, triplet
    short_conv.install()
    message_gate.install()
    attention_mask.install()
    projections.install()
    triplet.install()
    compiled_ring.install()
    dynamic_conv.install()
    layers.SparseMoeBlock.__call__ = single_token_moe
