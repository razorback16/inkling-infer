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


_prompt = mx.fast.metal_kernel(
    name="inkling_prompt_dynamic_filter", input_names=["x","history","fixed","dynamic"],
    output_names=["y","window"], source="""
        #pragma clang fp contract(off)
        uint c=thread_position_in_grid.x, t=thread_position_in_grid.y;
        uint head=c/(C/H);
        float original=float(x[t*C+c]), total=0;
        for (uint lag=0;lag<4;++lag) {
            float v=lag<=t ? float(x[(t-lag)*C+c]) :
                HAS_HISTORY ? float(history[(3+t-lag)*C+c]) : 0.f;
            float w=float(fixed[lag*C+c])+float(dynamic[(t*H+head)*4+lag]);
            volatile float product=v*w;
            total+=product;
        }
        y[t*C+c]=T((total+original)/3.f);
        if (!TAIL || t+3>=L) window[(TAIL ? t+3-L : t+3)*C+c]=original;
        if (t==0 && !TAIL) {
            for (uint i=0;i<3;++i) window[i*C+c]=HAS_HISTORY ? float(history[i*C+c]) : 0.f;
        }
    """,
)


def convolve(layer, hidden_states, dynamic_input, conv_state, conv_mask):
    if (hidden_states.ndim==3 and hidden_states.shape[0]==1 and hidden_states.shape[1]>1
            and conv_mask is None and layer.kernel_size==4 and hidden_states.dtype==mx.bfloat16
            and hidden_states.shape[-1]%layer.num_heads==0
            and (conv_state is None or conv_state.shape==(1,3,hidden_states.shape[-1]))):
        _,length,width=hidden_states.shape
        from .prefill_layer import _tail_only
        tail = _tail_only.get() and length>=3
        return _prompt(
            inputs=[hidden_states,hidden_states if conv_state is None else conv_state,
                    layer.static_weight,layer.weight_proj(dynamic_input)],
            template=[("TAIL",tail),("L",length),("T",hidden_states.dtype),("C",width),("H",layer.num_heads),("HAS_HISTORY",conv_state is not None)],
            grid=(width,length,1),threadgroup=(min(128,width),1,1),
            output_shapes=[hidden_states.shape,(1,3 if tail else length+3,width)],output_dtypes=[hidden_states.dtype,mx.float32],
        )
    width = hidden_states.shape[-1]
    if (hidden_states.shape[:2] != (1, 1) or conv_mask is not None or layer.kernel_size != 4
            or conv_state is None or conv_state.shape != (1, 3, width)):
        return _reference(layer, hidden_states, dynamic_input, conv_state, conv_mask)
    dynamic = layer.weight_proj(dynamic_input)
    return _kernel(
        inputs=[hidden_states, conv_state, layer.static_weight, dynamic],
        template=[("T", hidden_states.dtype), ("C", width), ("H", layer.num_heads)],
        grid=(width, 1, 1), threadgroup=(128, 1, 1),
        output_shapes=[hidden_states.shape, (1, 4, width)], output_dtypes=[hidden_states.dtype, mx.float32],
    )


def install():
    layers.DynamicShortConvolution.__call__ = convolve
