"""Gather the eight sorted expert outputs directly into the prompt mixture."""
import mlx.core as mx
from mlx_lm.layers.moe import _gather_sort
from mlx_lm.models.inkling2 import layers


_mix_sorted = mx.fast.metal_kernel(
    name="inkling_prefill_mix_sorted", input_names=["states", "inverse", "weights"],
    output_names=["out"], source="""
        #pragma clang fp contract(off)
        uint token=thread_position_in_grid.y, col=thread_position_in_grid.x*4;
        float4 sum=0;
        for (uint slot=0;slot<8;++slot) {
            uint pos=token*8+slot;
            uint row=inverse[pos];
            float4 state=float4(*(device const bfloat4*)(states+row*2304+col));
            sum+=state*float(weights[pos]);
        }
        *(device bfloat4*)(out+token*2304+col)=bfloat4(sum);
    """,
)


def mix(layer, x):
    indices, weights = layer.gate(x)
    hidden, sorted_ids, inverse = _gather_sort(mx.expand_dims(x, (-2, -3)), indices)
    experts = layer.experts
    gated = experts.gate_up_proj(hidden, sorted_ids, sorted_indices=True)
    active = experts.act_fn(gated[..., 0::2], gated[..., 1::2])
    results = experts.down_proj(active, sorted_ids, sorted_indices=True)
    return _mix_sorted(
        inputs=[results, inverse, weights], grid=(576, x.shape[1], 1),
        threadgroup=(64, 1, 1), output_shapes=[x.shape], output_dtypes=[x.dtype],
    )[0]


def applicable(layer, x):
    return (x.ndim == 3 and x.shape[0] == 1 and x.shape[1] >= 8 and x.shape[2] == 2304
            and x.dtype == mx.bfloat16 and layer.sharding_group is None
            and layer.gate.top_k == 8 and type(layer.experts) is layers.Experts
            and all(getattr(p, "mode", None) in ("nvfp4", "mxfp8")
                    for p in (layer.experts.gate_up_proj, layer.experts.down_proj)))
