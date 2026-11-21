"""Apply the two rank-64 message projections and mix their source states."""
import mlx.core as mx

_kernel = mx.fast.metal_kernel(
    name="inkling_gain_forget", input_names=["x", "packed", "state", "msg"], output_names=["out"],
    source="""
        uint lane = thread_index_in_simdgroup;
        uint row = thread_position_in_grid.x / 4;
        uint col = (lane % 4) * 4;
        uint bank = thread_position_in_grid.y;
        uint base = (bank * 4608 + row) * 64;
        float f = 0, g = 0;
        for (uint tile = 0; tile < 64; tile += 16) {
            for (uint i = 0; i < 4; ++i) {
                uint k = tile + col + i;
                float v = float(x[bank * 64 + k]);
                uint addr=((((bank*288+row/8)*4+tile/16)*2)*8+row%8)*16+col+i;
                f = fma(v, float(packed[TILED ? addr : base+k]), f);
                g = fma(v, float(packed[TILED ? addr+128 : base+2304*64+k]), g);
            }
        }
        f += simd_shuffle_down(f, 2);
        g += simd_shuffle_down(g, 2);
        f += simd_shuffle_down(f, 1);
        g += simd_shuffle_down(g, 1);
        if ((lane % 4) == 0) {
            uint c = bank * 2304 + row;
            volatile float retained = (1.f - precise::tanh(f)) * float(state[SHARED ? row : c]);
            volatile float added = precise::tanh(g) * float(msg[c]);
            out[c] = retained + added;
        }
    """,
)


def prepare(model):
    from .exact_sources import native
    arrays=[]
    for index in model.args.loop_core:
        layer=model.layers[index]
        triplet=(layer.q_message_gate,layer.k_message_gate,layer.v_message_gate)
        for owner,gates in ((layer,triplet),(layer.mlp_message_gate,(layer.mlp_message_gate,))):
            if any(g is None for g in gates):
                continue
            modules=tuple(p for g in gates for p in (g.forget_gate_proj,g.message_gate_proj))
            weights=tuple(p.weight for p in modules)
            if any(w.shape!=(2304,64) for w in weights):
                continue
            old=getattr(owner,"_warp_gain",None)
            if old is not None and all(x is y for x,y in zip(weights,old[0])):
                continue
            raw=mx.stack([native(p) for p in modules]).reshape(len(gates),2,288,8,4,16)
            tiled=mx.contiguous(raw.transpose(0,2,4,1,3,5))
            object.__setattr__(owner,"_warp_gain",(weights,tiled))
            arrays.append(tiled)
    if arrays:
        mx.eval(arrays)


def lookup(owner,weights):
    old=getattr(owner,"_warp_gain",None)
    if old is not None and all(x is y for x,y in zip(weights,old[0])):
        return old[1]
    return None


def apply(x, packed, state, messages, tiles=None):
    banks = messages.size // 2304
    return _kernel(inputs=[x, packed if tiles is None else tiles, state, messages], template=[("SHARED",state.size != messages.size),("TILED",tiles is not None)],
                   grid=(2304 * 4, banks, 1), threadgroup=(128, 1, 1),
                   output_shapes=[messages.shape], output_dtypes=[mx.float32])[0]
