"""Normalize the residual and evaluate the learned depth queries in one launch."""
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.kernels.common import scalar

_filter = """
            uint d = tid*4+j;
            float conv = 0;
            for (uint t = 0; t < 3; ++t) {
                float old = history[t*2304+d];
                conv = fma(float(W(old)), float(w[d*4+t]), conv);
                window[t*2304+d] = old;
            }
            window[3*2304+d] = v;
            conv = fma(float(W(v)), float(w[d*4+3]), conv);
            float filtered = float(T((float(W(conv))+v) * 0.4472135954999579f));
            v = BOUNDARY ? filtered : float(hidden[d]) + filtered;
            residual[d] = v;
"""

_source = """
        uint tid = thread_position_in_threadgroup.x, sg = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup, query = threadgroup_position_in_grid.x;
        threadgroup float norm[2304], sums[18], scores[K+1][9], coeff[K+1];
        float4 values;
        float sum = 0;
        for (uint j = 0; j < 4; ++j) {
            float v = float(x[tid*4+j]);
            FILTER
            values[j] = v;
            sum += v*v;
        }
        sum = simd_sum(sum);
        if (lane == 0) sums[sg] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float scale = precise::rsqrt(simd_sum(lane < 18 ? sums[lane] : 0.f)/2304.f + 1.e-6f);
        for (uint j = 0; j < 4; ++j) {
            uint d = tid*4+j;
            norm[d] = values[j]*scale;
            if (!query) normalized[d] = norm[d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint group = tid/288;
        for (uint i=group; i<=K; i+=2) {
            float dot=0;
            for (uint d=tid%288; d<2304; d+=288)
                dot+=(i==K ? norm[d] : blocks[i*2304+d])*queries[query*2304+d];
            dot=simd_sum(dot);
            if (!lane) scores[i][sg%9]=dot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (K<32 && !sg) {
            // Each source keeps its serial nine-warp sum; sources proceed independently.
            float logit=-INFINITY;
            if (lane<=K) {
                float dot=0;
                for (uint j=0; j<9; ++j) dot+=scores[lane][j];
                logit=dot*(1.f/48.f);
            }
            float weight=metal::exp(logit-simd_max(logit));
            float total=0;
            for (uint i=0; i<=K; ++i) total+=simd_shuffle(weight,i);
            if (lane<=K) coeff[lane]=weight/total;
        }
        else if (K>=32 && !tid) {
            float peak = -INFINITY, z = 0;
            for (uint i = 0; i <= K; ++i) {
                float dot = 0;
                for (uint j = 0; j < 9; ++j) dot += scores[i][j];
                coeff[i] = dot * (1.f/48.f);
                peak = max(peak, coeff[i]);
            }
            for (uint i = 0; i <= K; ++i) {
                coeff[i] = metal::exp(coeff[i]-peak);
                z += coeff[i];
            }
            for (uint i = 0; i <= K; ++i) coeff[i] /= z;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j = 0; j < 4; ++j) {
            uint d = tid*4+j;
            float v = 0;
            for (uint i = 0; i < K; ++i) v += blocks[i*2304+d]*coeff[i];
            mixed[query*2304+d] = v + norm[d]*coeff[K];
        }
    """


def make_kernel(filtered):
    inputs = ["x", "blocks", "queries"]
    outputs = ["mixed", "normalized"]
    if filtered:
        inputs.extend(["history", "w", "hidden"])
        outputs.extend(["residual", "window"])
    return mx.fast.metal_kernel(
        name="inkling_filtered_depth" if filtered else "inkling_residual_depth",
        input_names=inputs, output_names=outputs,
        source=_source.replace("FILTER", _filter if filtered else ""),
    )


_kernel = make_kernel(False)
_filtered = make_kernel(True)


def evaluate(module, blocks, x):
    q = module.query.size // 2304
    return _kernel(inputs=[x,blocks,module.query], template=[("K",blocks.shape[0])],
                   grid=(q*576,1,1),threadgroup=(576,1,1),
                   output_shapes=[(1,1,q,2304),x.shape],output_dtypes=[mx.float32,mx.float32])


def normalized_source(filtered=False):
    source = _source.replace("FILTER", _filter if filtered else "").replace(
        "mixed[query*2304+d] = v + norm[d]*coeff[K];", "values[j] = v + norm[d]*coeff[K];")
    return source + """
        float norm2=0;
        for (uint j=0; j<4; ++j) norm2+=values[j]*values[j];
        norm2=simd_sum(norm2);
        if (!lane) sums[sg]=norm2;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float inv=precise::rsqrt(simd_sum(lane<18 ? sums[lane] : 0.f)/2304.f+eps);
        for (uint j=0; j<4; ++j) {
            uint col=query*2304+tid*4+j;
            mixed[col]=bfloat16_t((values[j]*inv)*float(weights[col]));
        }
    """


_normalized_inputs = mx.fast.metal_kernel(
    name="inkling_depth_attention_input_norm",
    input_names=["x","blocks","queries","weights","eps"], output_names=["mixed","normalized"],
    source=normalized_source(),
)
_mlp_inputs = mx.fast.metal_kernel(
    name="inkling_depth_mlp_input_norm",
    input_names=["x","blocks","queries","history","w","hidden","weights","eps"],
    output_names=["mixed","normalized","residual","window"],
    source=normalized_source(filtered=True),
)


def attention_inputs(layer, blocks, x):
    if hasattr(layer, "input_layernorm"):
        norms = (layer.input_layernorm,)
    else:
        norms = (layer.q_input_layernorm, layer.k_input_layernorm, layer.v_input_layernorm)
    if any(type(norm) is not nn.RMSNorm or norm.eps != norms[0].eps for norm in norms):
        return None
    weights = tuple(norm.weight for norm in norms)
    if len(weights) == 1:
        joined = weights[0]
    else:
        saved = getattr(layer, "_attention_norm_weights", None)
        if saved is None or any(a is not b for a,b in zip(saved[0],weights)):
            saved = (weights, mx.concatenate(weights))
            object.__setattr__(layer, "_attention_norm_weights", saved)
        joined = saved[1]
    q = len(norms)
    states, normalized = _normalized_inputs(
        inputs=[x,blocks,layer.attention_input.query,joined,scalar(norms[0].eps)],
        template=[("K",blocks.shape[0])], grid=(q*576,1,1), threadgroup=(576,1,1),
        output_shapes=[(1,q,2304),x.shape],output_dtypes=[mx.bfloat16,mx.float32])
    inputs = (states,)*3 if q==1 else tuple(mx.split(states,3,axis=1))
    return inputs, normalized


def post_attention(layer, blocks, attention, history, hidden, normalize=False):
    weights = layer.post_attention_convolution.conv1d.weight
    inputs = [attention,blocks,layer.mlp_input.query,history,weights,hidden]
    kernel = _filtered
    if normalize:
        norm = layer.post_attention_layernorm
        inputs += [norm.weight,scalar(norm.eps)]
        kernel = _mlp_inputs
    mixed, _, residual, window = kernel(
        inputs=inputs,
        template=[("K", blocks.shape[0]), ("T", attention.dtype), ("W", weights.dtype),
                  ("BOUNDARY", layer.is_block_boundary)],
        grid=(576,1,1), threadgroup=(576,1,1),
        output_shapes=[(1,1,1,2304), hidden.shape, hidden.shape, (1,4,2304)],
        output_dtypes=[attention.dtype if normalize else mx.float32]+[mx.float32]*3)
    return mixed, residual, window
