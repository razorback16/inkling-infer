"""Cache learned depth-query dot products once per closed residual block."""
from contextlib import contextmanager
from contextvars import ContextVar

import mlx.core as mx

_active = ContextVar("inkling_closed_scores",default=None)
_dot = mx.fast.metal_kernel(
    name="inkling_closed_block_queries", input_names=["x","queries"], output_names=["out"],
    source="""
        uint q=threadgroup_position_in_grid.x, tid=thread_position_in_threadgroup.x;
        uint lane=thread_index_in_simdgroup, sg=simdgroup_index_in_threadgroup;
        float dot=0;
        for (uint d=tid;d<2304;d+=288) dot+=x[d]*queries[q*2304+d];
        dot=simd_sum(dot);
        threadgroup float partial[9];
        if (!lane) partial[sg]=dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!tid) {
            float total=0;
            for (uint i=0;i<9;++i) total+=partial[i];
            out[q]=total;
        }
    """,
)


@contextmanager
def trace(model):
    modules = [m for layer in model.layers for m in (layer.attention_input, layer.mlp_input)
               if m is not None]
    if any(m.query.shape[-1] != 2304 or m.query.dtype != mx.float32 for m in modules):
        yield
        return
    starts = {}
    rows = []
    count = 0
    for m in modules:
        starts[id(m)] = count
        q = m.query.reshape(-1,2304)
        rows.append(q)
        count += q.shape[0]
    token = _active.set((starts,mx.concatenate(rows),{}))
    try:
        yield
    finally:
        _active.reset(token)


def append(old, blocks, normalized):
    active = _active.get()
    if active is None:
        return
    _, queries, saved = active
    if old is not None and id(old) not in saved:
        return
    count = queries.shape[0]
    scores = _dot(inputs=[normalized,queries], grid=(count*288,1,1),threadgroup=(288,1,1),
                  output_shapes=[(1,count)],output_dtypes=[mx.float32])[0]
    if old is not None:
        scores = mx.concatenate((saved[id(old)][1],scores),axis=0)
    saved[id(blocks)] = (blocks,scores)


def lookup(module, blocks):
    active = _active.get()
    if active is not None:
        starts, queries, saved = active
        if id(blocks) in saved and id(module) in starts and blocks.shape[0] < 32:
            return saved[id(blocks)][1], [("CLOSED",True),("Q0",starts[id(module)]),("STRIDE",queries.shape[0])]
    return blocks, [("CLOSED",False),("Q0",0),("STRIDE",1)]
