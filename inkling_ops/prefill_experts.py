"""Gather the eight sorted expert outputs directly into the prompt mixture."""
import mlx.core as mx
from mlx_lm.models.inkling2 import layers


_permutation = mx.fast.metal_kernel(
    name="inkling_prompt_expert_permutation",input_names=["ids","order"],
    output_names=["rows","sorted_ids","inverse"],source="""
        uint sorted=thread_position_in_grid.x, original=order[sorted];
        rows[sorted]=original/8;
        sorted_ids[sorted]=ids[original];
        inverse[original]=sorted;
    """,
)


_mix_sorted = mx.fast.metal_kernel(
    name="inkling_prefill_mix_sorted", input_names=["states", "inverse", "weights"],
    output_names=["out"], source="""
        #pragma clang fp contract(off)
        uint token=thread_position_in_grid.y, col=thread_position_in_grid.x*4;
        float4 sum=0;
        for (uint slot=0;slot<8;++slot) {
            uint pos=token*8+slot;
            uint addr=ROUTE_LAYOUT ? ((token*36+col/64)*8+slot)*64+col%64 : inverse[pos]*2304+col;
            float4 state=float4(*(device const bfloat4*)(states+addr));
            sum+=state*float(weights[pos]);
        }
        *(device bfloat4*)(out+token*2304+col)=bfloat4(sum);
    """,
)


_activate = mx.fast.metal_kernel(
    name="inkling_prompt_gate_activation", input_names=["pairs","table"], output_names=["out"],
    source="""
        uint first=thread_position_in_grid.x*4;
        for (uint j=0;j<4;++j) {
            ushort2 bits=((device const ushort2*)pairs)[first+j];
            uint g=((uint(bits.x)&32767)<<2)|((bits.x>>15)<<1);
            uint u=((uint(bits.y)&32767)<<2)|((bits.y>>15)<<1);
            out[first+j]=bfloat16_t(table[g]*table[u+1]);
        }
    """,
)


def mix(layer, x):
    indices, weights = layer.gate(x)
    ids=indices.reshape(-1)
    order=mx.argsort(ids)
    rows, sorted_ids, inverse=_permutation(
        inputs=[ids,order],grid=(ids.size,1,1),threadgroup=(min(ids.size,256),1,1),
        output_shapes=[ids.shape]*3,output_dtypes=[mx.uint32,ids.dtype,mx.uint32],
    )
    hidden=x.reshape(-1,1,x.shape[-1])[rows]
    experts = layer.experts
    from . import prefill_qmm
    custom=(hidden.shape[0]>=4096 and prefill_qmm.applicable(experts.gate_up_proj)
            and prefill_qmm.applicable(experts.down_proj) and prefill_qmm.kernel() is not None)
    tiles=prefill_qmm.schedule(sorted_ids) if custom else None
    tabled=(type(experts.act_fn) is layers.SoftcappedSwiGLU and x.dtype==mx.bfloat16
            and experts.act_fn.gate_softcap==4 and experts.act_fn.up_softcap==25)
    if custom and tabled:
        from .expert_gate import activation_tables
        active=prefill_qmm.project(experts.gate_up_proj,x.reshape(-1,1,x.shape[-1]),tiles,activation_tables(),rows)
    else:
        gated=(prefill_qmm.project(experts.gate_up_proj,hidden,tiles) if custom
               else experts.gate_up_proj(hidden,sorted_ids,sorted_indices=True))
        if tabled and gated.shape[-1]%8==0:
            from .expert_gate import activation_tables
            shape=(*gated.shape[:-1],gated.shape[-1]//2)
            active=_activate(inputs=[gated,activation_tables()],grid=(gated.size//8,1,1),
                             threadgroup=(128,1,1),output_shapes=[shape],output_dtypes=[gated.dtype])[0]
        else:
            active=experts.act_fn(gated[...,0::2],gated[...,1::2])
    results=(prefill_qmm.project(experts.down_proj,active,tiles,original_slots=order) if custom
             else experts.down_proj(active,sorted_ids,sorted_indices=True))
    return _mix_sorted(
        inputs=[results, inverse, weights], template=[("ROUTE_LAYOUT",custom)], grid=(576, x.shape[1], 1),
        threadgroup=(64, 1, 1), output_shapes=[x.shape], output_dtypes=[x.dtype],
    )[0]


def applicable(layer, x):
    return (x.ndim == 3 and x.shape[0] == 1 and x.shape[1] >= 8 and x.shape[2] == 2304
            and x.dtype == mx.bfloat16 and layer.sharding_group is None
            and layer.gate.top_k == 8 and type(layer.experts) is layers.Experts
            and all(getattr(p, "mode", None) in ("nvfp4", "mxfp8")
                    for p in (layer.experts.gate_up_proj, layer.experts.down_proj)))
