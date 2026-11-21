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


_prefill_conv = mx.fast.metal_kernel(
    name="inkling_prompt_short_convolution", input_names=["x", "history", "weights"],
    output_names=["y", "window"], source="""
        uint c=thread_position_in_grid.x, first=thread_position_in_grid.y*4;
        float4 w=float4(((device const vec<W,4>*)weights)[c]);
        float a=0, b=0, prev=0;
        if (first) {
            a=float(x[(first-3)*C+c]);
            b=float(x[(first-2)*C+c]);
            prev=float(x[(first-1)*C+c]);
        } else {
            if (HAS_HISTORY) { a=history[c]; b=history[C+c]; prev=history[2*C+c]; }
            if (!TAIL) {window[c]=a; window[C+c]=b; window[2*C+c]=prev;}
        }
        for (uint i=first;i<min(first+4u,uint(L));++i) {
            float v=float(x[i*C+c]);
            float total=fma(float(W(a)),w.x,0.f);
            total=fma(float(W(b)),w.y,total);
            total=fma(float(W(prev)),w.z,total);
            total=fma(float(W(v)),w.w,total);
            y[i*C+c]=T((float(W(total))+v)*0.4472135954999579f);
            if (!TAIL || i+3>=L) window[(TAIL ? i+3-L : i+3)*C+c]=v;
            a=b; b=prev; prev=v;
        }
    """,
)


def convolve(layer, x, conv_state, conv_mask):
    if (x.ndim==3 and x.shape[0]==1 and x.shape[1]>1 and x.dtype==mx.bfloat16
            and conv_mask is None and layer.kernel_size==4 and "bias" not in layer.conv1d
            and layer.conv1d.weight.shape==(x.shape[-1],4,1)
            and (conv_state is None or (conv_state.shape==(1,3,x.shape[-1]) and conv_state.dtype==mx.float32))):
        _, length, width=x.shape
        weight=layer.conv1d.weight
        from .prefill_layer import _tail_only
        tail = _tail_only.get() and length>=3
        return _prefill_conv(
            inputs=[x, x if conv_state is None else conv_state, weight],
            template=[("TAIL",tail),("T",x.dtype),("W",weight.dtype),("C",width),("L",length),("HAS_HISTORY",conv_state is not None)],
            grid=(width,(length+3)//4,1),threadgroup=(min(128,width),1,1),
            output_shapes=[x.shape,(1,3 if tail else length+3,width)],output_dtypes=[x.dtype,mx.float32],
        )
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
