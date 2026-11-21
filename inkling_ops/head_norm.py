"""Normalize four sets of attention heads in a single decode launch."""
import mlx.core as mx
from mlx_lm.kernels.common import scalar

_sink = """
        uint norm_heads = 2*H + 2*KV;
        uint group = threadgroup_position_in_grid.x;
        uint sg = simdgroup_index_in_threadgroup, tid = thread_position_in_threadgroup.x;
        uint lane0 = thread_index_in_simdgroup;
        if (group >= (norm_heads+3)/4) {
            uint row = group - (norm_heads+3)/4;
            float dot = 0;
            for (uint first = tid*4; first < 2304; first += 1024) {
                for (uint j = 0; j < 4; ++j)
                    dot = fma(float(sink_input[first+j]),float(sink_weight[row*2304+first+j]),dot);
            }
            for (ushort step=16; step; step>>=1) dot += simd_shuffle_down(dot,step);
            threadgroup float sums[8];
            if (lane0 == 0) sums[sg] = dot;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (!tid) {
                float total = sums[0];
                for (uint i=1; i<8; ++i) total += sums[i];
                sinks[row] = T(total);
            }
            return;
        }
        if (sg>=4 || group*4+sg>=norm_heads) return;
        const uint head = group*4+sg;
"""
_source = """
        const uint head = threadgroup_position_in_grid.x;
        const uint lane = thread_index_in_simdgroup;
        const bool query = head < H;
        const bool key = head >= H && head < H + KV;
        const bool val = head >= H + KV && head < H + 2 * KV;
        const bool rel = head >= H + 2 * KV;
        const uint width = rel ? R : D;
        const uint local_head = query ? head : key ? head-H : val ? head-H-KV : head-H-2*KV;
        const uint base = local_head * width;
        float states[D / 32];
        float sum = 0;
        for (uint d = lane * ((width+31)/32); d < min(width,(lane+1)*((width+31)/32)); ++d) {
            uint i = base + d;
            float x = query ? float(q[i]) * float(qs[i]) :
                      key ? float(k[i]) : val ? float(v[i]) : float(r[i]) * float(rs[i]);
            states[d % ((width+31)/32)] = x;
            volatile float square = x*x;
            sum += square;
        }
        if (rel && R <= 64) {
            sum = 0;
            for (uint j = 0; j < R; ++j) {
                float y = float(r[base+j]) * float(rs[base+j]);
                volatile float square = y*y;
                sum += square;
            }
        } else {
            sum = simd_sum(sum);
        }
        float scale = val ? metal::precise::rsqrt(sum / float(D) + eps) :
            min(1.0f, (rel ? rel_threshold : threshold) / (metal::precise::sqrt(sum) + 1e-6f));
        for (uint d = lane * ((width+31)/32); d < min(width,(lane+1)*((width+31)/32)); ++d) {
            T x = T(states[d % ((width+31)/32)] * scale);
            if (query) qo[base+d] = x;
            else if (key) kvo[base+d] = x;
            else if (val) kvo[KV*D+base+d] = x;
            else ro[base+d] = x;
        }
    """


def _make(with_sink):
    names = ["q", "k", "v", "r", "qs", "rs", "threshold", "rel_threshold", "eps"]
    outputs = ["qo", "kvo", "ro"]
    source = _source
    if with_sink:
        names += ["sink_input", "sink_weight"]
        outputs.append("sinks")
        source = source.replace("const uint head = threadgroup_position_in_grid.x;", _sink)
    return mx.fast.metal_kernel(
        name="inkling_head_norm_sink" if with_sink else "inkling_head_norm",
        input_names=names, output_names=outputs, source=source,
    )


_kernel, _with_sink = _make(False), _make(True)


def normalize(layer, query, key, value, relative, return_pair=False, sink_input=None):
    heads, kv, dim = layer.num_heads, layer.num_key_value_heads, layer.head_dim
    rel = relative.shape[-1] // heads
    inputs = [query, key, value, relative, layer.q_l2_clip.scale, layer.rel_l2_clip.scale,
              scalar(layer.q_l2_clip.threshold),scalar(layer.rel_l2_clip.threshold),scalar(layer.eps)]
    shapes = [(1,1,heads,dim),(1,1,2*kv,dim),(1,1,heads,rel)]
    groups, threads, kernel = 2*heads+2*kv, 32, _kernel
    if sink_input is not None:
        inputs += [sink_input,layer.sink_proj.weight]
        shapes.append((1,1,heads))
        groups, threads, kernel = (groups+3)//4+heads, 256, _with_sink
    outputs = kernel(inputs=inputs,template=[("T",query.dtype),("H",heads),("KV",kv),("D",dim),("R",rel)],
                     grid=(groups*threads,1,1),threadgroup=(threads,1,1),
                     output_shapes=shapes,output_dtypes=[query.dtype]*len(shapes))
    q, pair, r = outputs[:3]
    k, v = mx.split(pair,2,axis=2)
    if sink_input is not None:
        return q,k,v,r,pair,outputs[3]
    return (q,k,v,r,pair) if return_pair else (q,k,v,r)
