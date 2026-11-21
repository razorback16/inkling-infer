"""Score the fixed residual blocks for MLP and attention inputs in their six-layer segment."""
from contextvars import ContextVar
import mlx.core as mx

_active = ContextVar("inkling_segment_scores",default=None)
_kernel = mx.fast.metal_kernel(
    name="inkling_segment_depth_scores",input_names=["blocks","queries"],output_names=["out"],
    source="""
        uint tid=thread_position_in_threadgroup.x,lane=thread_index_in_simdgroup;
        uint sg=simdgroup_index_in_threadgroup,q=threadgroup_position_in_grid.x;
        uint k=threadgroup_position_in_grid.y;
        float dot=0;
        for (uint d=tid;d<2304;d+=288) dot+=blocks[k*2304+d]*queries[q*2304+d];
        dot=simd_sum(dot);
        threadgroup float parts[9];
        if (!lane) parts[sg]=dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!tid) {
            float total=0;
            for (uint i=0;i<9;++i) total+=parts[i];
            out[k*threadgroups_per_grid.x+q]=total;
        }
    """,
)


def prepare(model):
    result={}
    width=model.args.n_layers_per_attn_res_block
    arrays=[]
    for start in range(0,len(model.layers),width):
        layers=model.layers[start:start+width]
        modules = [layer.mlp_input for layer in layers]
        modules.extend(layer.attention_input for layer in layers
                       if not layer.is_block_boundary and layer.attention_input is not None)
        queries = [module.query for module in modules]
        if any(q.size % 2304 or q.dtype != mx.float32 for q in queries):
            continue
        table = mx.concatenate([q.reshape(-1, 2304) for q in queries])
        arrays.append(table)
        column = 0
        for module, query in zip(modules, queries):
            result[id(module)] = (table, column)
            column += query.size // 2304
    if arrays:
        mx.eval(arrays)
    return result


def run(grouped,func,*args):
    token=_active.set((grouped,{}))
    try:
        return func(*args)
    finally:
        _active.reset(token)


def lookup(module,blocks):
    ctx=_active.get()
    if ctx is None or id(module) not in ctx[0] or not 2<=blocks.shape[0]<32:
        return blocks,[("CACHED",False),("STRIDE",1),("COLUMN",0)]
    table,col=ctx[0][id(module)]
    key=(id(blocks),id(table)); n=table.shape[0];k=blocks.shape[0]
    if key not in ctx[1]:
        scores=_kernel(inputs=[blocks,table],grid=(n*288,k,1),threadgroup=(288,1,1),
                       output_shapes=[(k,n)],output_dtypes=[mx.float32])[0]
        ctx[1][key]=(blocks,scores)
    return ctx[1][key][1],[("CACHED",True),("STRIDE",n),("COLUMN",col)]
