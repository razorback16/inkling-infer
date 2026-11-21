"""Score the fixed residual blocks for MLP and attention inputs in their six-layer segment."""
from contextvars import ContextVar
from functools import lru_cache
import mlx.core as mx

_active = ContextVar("inkling_segment_scores",default=None)
@lru_cache(maxsize=16)
def _kernel(rows):
    names = [f"query{i}" for i in range(len(rows))]
    offsets = [sum(rows[:i]) for i in range(len(rows))]
    selections = [
        f"if (q >= {start}) {{ query = {name}; local_row = q - {start}; }}"
        for name, start in zip(names[1:], offsets[1:])
    ]
    source="""
        uint tid=thread_position_in_threadgroup.x,lane=thread_index_in_simdgroup;
        uint sg=simdgroup_index_in_threadgroup,q=threadgroup_position_in_grid.x;
        uint k=threadgroup_position_in_grid.y;
        auto query=query0;
        uint local_row=q;
        SELECT_QUERY
        float dot=0;
        for (uint d=tid;d<2304;d+=288) dot+=blocks[k*2304+d]*query[local_row*2304+d];
        dot=simd_sum(dot);
        threadgroup float parts[9];
        if (!lane) parts[sg]=dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!tid) {
            float total=0;
            for (uint i=0;i<9;++i) total+=parts[i];
            out[k*threadgroups_per_grid.x+q]=total;
        }
    """.replace("SELECT_QUERY", "\n".join(selections))
    return mx.fast.metal_kernel(name="inkling_segment_depth_scores", input_names=["blocks",*names],
                                output_names=["out"], source=source)


def prepare(model):
    result={}
    width=model.args.n_layers_per_attn_res_block
    for start in range(0,len(model.layers),width):
        layers=model.layers[start:start+width]
        modules = [layer.mlp_input for layer in layers]
        modules.extend(layer.attention_input for layer in layers
                       if not layer.is_block_boundary and layer.attention_input is not None)
        queries = [module.query for module in modules]
        if any(q.size % 2304 or q.dtype != mx.float32 for q in queries):
            continue
        table = tuple(queries)
        column = 0
        for module, query in zip(modules, queries):
            result[id(module)] = (table, column)
            column += query.size // 2304
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
    key=(id(blocks),id(table)); rows=tuple(q.size//2304 for q in table); n=sum(rows); k=blocks.shape[0]
    if key not in ctx[1]:
        scores=_kernel(rows)(inputs=[blocks,*table],grid=(n*288,k,1),threadgroup=(288,1,1),
                       output_shapes=[(k,n)],output_dtypes=[mx.float32])[0]
        ctx[1][key]=(blocks,scores)
    return ctx[1][key][1],[("CACHED",True),("STRIDE",n),("COLUMN",col)]
