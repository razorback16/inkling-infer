"""Normalize four sets of attention heads in a single decode launch."""
import mlx.core as mx
from mlx_lm.kernels.common import scalar

_source = """
        #pragma clang fp contract(off)
        const uint head = thread_position_in_grid.x / 32;
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
            float square = x*x;
            sum += square;
        }
        if (rel && R <= 64) {
            sum = 0;
            for (uint j = 0; j < R; ++j) {
                float y = float(r[base+j]) * float(rs[base+j]);
                float square = y*y;
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
_inputs = ["q", "k", "v", "r", "qs", "rs", "threshold", "rel_threshold", "eps"]
_outputs = ["qo", "kvo", "ro"]
_kernel = mx.fast.metal_kernel(name="inkling_head_norm", input_names=_inputs,
                               output_names=_outputs, source=_source)
_filter = """
            if (key || val) {
                auto history = key ? kh : vh;
                auto conv = key ? kw : vw;
                auto next = key ? k_window : v_window;
                float total = 0;
                for (uint t = 0; t < 3; ++t) {
                    float old = history[t*KV*D+i];
                    next[t*KV*D+i] = old;
                    total = fma(float(W(old)), float(conv[i*4+t]), total);
                }
                next[3*KV*D+i] = x;
                total = fma(float(W(x)), float(conv[i*4+3]), total);
                x = float(T((float(W(total))+x)*0.4472135954999579f));
            }
"""
_filtered = mx.fast.metal_kernel(
    name="inkling_head_norm_filtered_pair", input_names=_inputs + ["kh","vh","kw","vw"],
    output_names=_outputs + ["k_window","v_window"],
    source=_source.replace("            states[d", _filter + "            states[d", 1),
)



_dynamic_filter = """
            if (key || val) {
                auto hist=key ? kh : vh;
                auto fixed=key ? kw : vw;
                auto coefs=key ? kd : vd;
                auto window=key ? k_window : v_window;
                float sum=0;
                for (uint lag=0;lag<4;++lag) {
                    float v=x;
                    if (lag) {
                        uint at=(3-lag)*KV*D+i;
                        v=hist[at];
                        window[at]=v;
                    }
                    float weight=float(fixed[lag*KV*D+i])+float(coefs[local_head*4+lag]);
                    volatile float product=v*weight;
                    sum+=product;
                }
                window[3*KV*D+i]=x;
                x=float(T((sum+x)/3.f));
            }
"""
_dynamic = mx.fast.metal_kernel(
    name="inkling_head_norm_dynamic_pair",input_names=_inputs+["kh","vh","kw","vw","kd","vd"],
    output_names=_outputs+["k_window","v_window"],
    source=_source.replace("            states[d",_dynamic_filter+"            states[d",1),
)


_positioned_source=_source.replace("            states[d",_dynamic_filter+"            states[d",1)
_positioned_source=_positioned_source.replace("if (query) qo[base+d] = x;",
    "if (query) qo[base+d] = T(float(x)*table[offset[0]]);").replace(
    "else ro[base+d] = x;","else ro[base+d] = T(float(x)*table[offset[0]]);")
_positioned=mx.fast.metal_kernel(
    name="inkling_positioned_dynamic_heads",input_names=_inputs+["kh","vh","kw","vw","kd","vd","table","offset"],
    output_names=_outputs+["k_window","v_window"],source=_positioned_source,
)


def normalize(layer, query, key, value, relative, return_pair=False, filters=None, position=None):
    heads, kv, dim = layer.num_heads, layer.num_key_value_heads, layer.head_dim
    rel = relative.shape[-1] // heads
    inputs = [query, key, value, relative, layer.q_l2_clip.scale, layer.rel_l2_clip.scale,
              scalar(layer.q_l2_clip.threshold), scalar(layer.rel_l2_clip.threshold), scalar(layer.eps)]
    shapes = [(1, 1, heads, dim), (1, 1, 2*kv, dim), (1, 1, heads, rel)]
    types = [query.dtype] * 3
    template = [("T", query.dtype), ("H", heads), ("KV", kv), ("D", dim), ("R", rel)]
    kernel = _kernel
    if filters is not None:
        inputs.extend(filters)
        shapes.extend([(1,4,kv*dim)] * 2)
        types.extend([mx.float32] * 2)
        if position is not None:
            inputs.extend(position)
            kernel=_positioned
        elif len(filters)==6:
            kernel=_dynamic
        else:
            template.append(("W", filters[2].dtype))
            kernel=_filtered
    q, pair, r, *windows = kernel(
        inputs=inputs, template=template, grid=(32*(2*heads+2*kv),1,1), threadgroup=(128,1,1),
        output_shapes=shapes, output_dtypes=types)
    if filters is not None:
        return q, pair, r, windows
    k, v = mx.split(pair,2,axis=2)
    return (q, k, v, r, pair) if return_pair else (q, k, v, r)
