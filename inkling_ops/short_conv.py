"""Four-tap depthwise convolution and residual for a single decoded token."""
import mlx.core as mx
from mlx_lm.models.inkling2 import layers

_reference_conv = layers.ShortConvolution.__call__
_reference_state = layers._conv_state
_decode_conv = mx.fast.metal_kernel(
    name="inkling_short_conv4",
    input_names=["x", "history", "weights"],
    output_names=["y", "window"],
    source="""
        uint c = thread_position_in_grid.x;
        float value = float(x[c]);
        float total = 0;
        for (uint k = 0; k < 3; ++k) {
            float v = history[k * C + c];
            window[k * C + c] = v;
            total = fma(float(W(v)), float(weights[c * 4 + k]), total);
        }
        window[3 * C + c] = value;
        total = fma(float(W(value)), float(weights[c * 4 + 3]), total);
        y[c] = T((float(W(total)) + value) * 0.4472135954999579f);
    """,
)


def convolve(layer, x, conv_state, conv_mask):
    if (x.shape[:2] != (1, 1) or conv_mask is not None or layer.kernel_size != 4
            or conv_state is None or conv_state.shape != (1, 3, x.shape[-1])):
        return _reference_conv(layer, x, conv_state, conv_mask)
    weights = layer.conv1d.weight
    width = x.shape[-1]
    return _decode_conv(
        inputs=[x, conv_state, weights],
        template=[("C", width), ("T", x.dtype), ("W", weights.dtype)],
        grid=(width, 1, 1), threadgroup=(128, 1, 1),
        output_shapes=[x.shape, (1, 4, width)], output_dtypes=[x.dtype, mx.float32],
    )


def next_state(window, lengths, n):
    if lengths is None and window.shape[:2] == (1, n + 1):
        return window[:, 1:]
    return _reference_state(window, lengths, n)


def install():
    layers.ShortConvolution.__call__ = convolve
    layers._conv_state = next_state
