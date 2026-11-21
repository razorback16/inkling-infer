"""Fuse the position-dependent four-tap K/V filters in full-attention decode."""
import mlx.core as mx
from mlx_lm.models.inkling2 import layers

_reference = layers.DynamicShortConvolution.__call__
_kernel = mx.fast.metal_kernel(
    name="inkling_dynamic_conv4",
    input_names=["x", "history", "fixed", "dynamic"],
    output_names=["y", "window"],
    source="""
        uint c = thread_position_in_grid.x;
        uint head = c / (C / H);
        float current = float(x[c]);
        window[3*C+c] = current;
        volatile float sum = 0;
        for (uint lag = 0; lag < 4; ++lag) {
            float value = current;
            if (lag > 0) {
                uint pos = (3-lag)*C+c;
                value = history[pos];
                window[pos] = value;
            }
            float w = float(fixed[lag*C+c]) + float(dynamic[head*4+lag]);
            volatile float product = value * w;
            sum = sum + product;
        }
        y[c] = T((sum + current) / 3.0f);
    """,
)


def convolve(layer, hidden_states, dynamic_input, conv_state, conv_mask):
    width = hidden_states.shape[-1]
    if (hidden_states.shape[:2] != (1, 1) or conv_mask is not None or layer.kernel_size != 4
            or conv_state is None or conv_state.shape != (1, 3, width)):
        return _reference(layer, hidden_states, dynamic_input, conv_state, conv_mask)
    return filtered(layer, hidden_states, conv_state, layer.weight_proj(dynamic_input))


def filtered(layer, hidden_states, conv_state, dynamic):
    width = hidden_states.shape[-1]
    return _kernel(
        inputs=[hidden_states, conv_state, layer.static_weight, dynamic],
        template=[("T", hidden_states.dtype), ("C", width), ("H", layer.num_heads)],
        grid=(width, 1, 1), threadgroup=(128, 1, 1),
        output_shapes=[hidden_states.shape, (1, 4, width)], output_dtypes=[hidden_states.dtype, mx.float32],
    )


def install():
    layers.DynamicShortConvolution.__call__ = convolve
