"""Gather pending per-visit KV pairs for an aligned sliding-ring update."""
from functools import lru_cache

import mlx.core as mx


@lru_cache(maxsize=20)
def gather_kernel(count):
    names = [f"pair_{i}" for i in range(count)]
    cases = "\n".join(f"case {i}: source = {name}; break;" for i,name in enumerate(names))
    return mx.fast.metal_kernel(
        name=f"inkling_gather_{count}_ring_pairs",
        input_names=names, output_names=["out"],
        source="""
            uint channel = thread_position_in_grid.x;
            uint visit = thread_position_in_grid.y;
            device const bfloat16_t *source = pair_0;
            switch (visit) { CASES }
            out[visit*2048+channel] = source[channel];
        """.replace("CASES",cases),
    )


def gather(pairs):
    if len(pairs)<5 or any(p.dtype!=mx.bfloat16 or p.shape!=(1,16,1,128) for p in pairs):
        return mx.concatenate(pairs,axis=0)
    parts=[]
    for start in range(0,len(pairs),20):
        group=pairs[start:start+20]
        count=len(group)
        part=gather_kernel(count)(inputs=group,grid=(2048,count,1),threadgroup=(128,1,1),
                                  output_shapes=[(count,16,1,128)],output_dtypes=[mx.bfloat16])[0]
        parts.append(part)
    return parts[0] if len(parts)==1 else mx.concatenate(parts,axis=0)
