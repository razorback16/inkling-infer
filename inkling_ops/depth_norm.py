"""Normalize the residual and evaluate the learned depth queries in one launch."""
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.kernels.common import scalar
from . import block_lookahead

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
        float4 blend = 0;
        for (uint i = 0; i < K; ++i) {
            uint d = i*2304 + tid*4;
            float4 block = float4(blocks[d], blocks[d+1], blocks[d+2], blocks[d+3]);
            blend += block * coeff[i];
        }
        for (uint j = 0; j < 4; ++j) {
            uint d = tid*4+j;
            float v = blend[j];
            mixed[query*2304+d] = v + norm[d]*coeff[K];
        }
    """


def cached_source(source):
    source=source.replace("for (uint i=group; i<=K; i+=2)",
                          "for (uint i=(CACHED ? K+group : group); i<=K; i+=2)")
    source=source.replace("for (uint j=0; j<9; ++j) dot+=scores[lane][j];",
                          "if (CACHED && lane<K) dot=closed[lane*STRIDE+COLUMN+query]; "
                          "else for (uint j=0; j<9; ++j) dot+=scores[lane][j];")
    return source


def make_kernel(filtered):
    inputs = ["x", "blocks", "queries"]
    outputs = ["mixed", "normalized"]
    if filtered:
        inputs.extend(["history", "w", "hidden"])
        outputs.extend(["residual", "window"])
    source = _source.replace("FILTER", _filter if filtered else "")
    if not filtered:
        inputs.append("closed")
        source = cached_source(source)
    return mx.fast.metal_kernel(
        name="inkling_filtered_depth" if filtered else "inkling_residual_depth",
        input_names=inputs, output_names=outputs,
        source=source,
    )


_kernel = make_kernel(False)
_filtered = make_kernel(True)


def evaluate(module, blocks, x):
    q = module.query.size // 2304
    closed, config = block_lookahead.lookup(module, blocks)
    return _kernel(inputs=[x,blocks,module.query,closed], template=[("K",blocks.shape[0]), *config],
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
    input_names=["x","blocks","queries","weights","eps","closed"], output_names=["mixed","normalized"],
    source=cached_source(normalized_source()),
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
    closed, config = block_lookahead.lookup(layer.attention_input, blocks)
    states, normalized = _normalized_inputs(
        inputs=[x,blocks,layer.attention_input.query,joined,scalar(norms[0].eps),closed],
        template=[("K",blocks.shape[0]), *config], grid=(q*576,1,1), threadgroup=(576,1,1),
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

def make_projected(normalize,routes=False,bottleneck=False):
    source = normalized_source() if normalize else _source.replace("FILTER","")
    source = source.replace("sum += v*v;", "").replace("sum = simd_sum(sum);","sum = simd_sum(squares[tid]);",1)
    source = source.replace("if (!query) normalized[d] = norm[d];","")
    source = cached_source(source)
    inputs = ["x","blocks","queries","squares","closed"]
    if normalize:
        inputs.extend(["weights","eps"])
    if routes:
        inputs.append("gate")
        source=source.replace("query = threadgroup_position_in_grid.x", "query = 0").replace(
            "mixed[col]=bfloat16_t((values[j]*inv)*float(weights[col]));",
            "bfloat16_t value=bfloat16_t((values[j]*inv)*float(weights[col])); "
            "norm[col]=float(value); if (!threadgroup_position_in_grid.x) mixed[col]=value;")
    if bottleneck:
        inputs.extend(["gate","message"])
        source=source.replace("query = threadgroup_position_in_grid.x", "query = 0").replace(
            "mixed[query*2304+d] = v + norm[d]*coeff[K];",
            "values[j]=v+norm[d]*coeff[K]; "
            "if (!threadgroup_position_in_grid.x) mixed[d]=values[j];")
        source += """
            float norm2=0;
            for (uint j=0;j<4;++j) norm2+=values[j]*values[j];
            norm2=simd_sum(norm2);
            if (!lane) sums[sg]=norm2;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float inv=precise::rsqrt(simd_sum(lane<18 ? sums[lane] : 0.f)/2304.f+1.e-6f);
            for (uint j=0;j<4;++j) norm[tid*4+j]=float(bfloat16_t(values[j]*inv));
        """
    if routes or bottleneck:
        tail = """
            threadgroup_barrier(mem_flags::mem_threadgroup);
            threadgroup float partial[16][1];
            uint first=threadgroup_position_in_grid.x*2+sg/8;
            if (tid<512) {
                float dots[1]={};
                for (uint col=(tid%256)*4;col<2304;col+=1024)
                    for (uint j=0;j<4;++j)
                        for (uint r=0;r<1;++r)
                            dots[r]=fma(norm[col+j],float(gate[(first+r)*2304+col+j]),dots[r]);
                for (uint r=0;r<1;++r) {
                    float sum=dots[r];
                    for (ushort step=16;step;step>>=1) sum+=simd_shuffle_down(sum,step);
                    if (!lane) partial[sg][r]=sum;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid<2) {
                uint bank=tid*8, col=0;
                float sum=partial[bank][col];
                for (uint i=1;i<8;++i) sum+=partial[bank+i][col];
                logits[threadgroup_position_in_grid.x*2+tid]=sum;
            }
        """
        if bottleneck:
            tail=tail.replace("*2+tid]=sum;", "*2+tid]=bfloat16_t(sum);")
            tail=tail.replace("col<2304", "col<4608").replace(
                "norm[col+j],float(gate[(first+r)*2304+col+j])",
                "(col<2304 ? norm[col+j] : float(bfloat16_t(message[col+j-2304]))),"
                "float(gate[(first+r)*4608+col+j])")
        source+=tail
        return mx.fast.metal_kernel(name="inkling_depth_message_reduce" if bottleneck else "inkling_depth_router_scores",input_names=inputs,
                                    output_names=["mixed","logits"],source=source)
    return mx.fast.metal_kernel(name="inkling_projected_depth_norm" if normalize else "inkling_projected_depth",
                                input_names=inputs,output_names=["mixed"],source=source)


_projected = (make_projected(False),make_projected(True))
_projected_routes = make_projected(True,True)
_projected_message = make_projected(False,bottleneck=True)


def from_projection(layer, blocks, residual, squares, normalize, gate=None, message=None):
    closed,config=block_lookahead.lookup(layer.mlp_input,blocks)
    inputs = [residual,blocks,layer.mlp_input.query,squares,closed]
    if normalize:
        norm=layer.post_attention_layernorm
        inputs.extend([norm.weight,scalar(norm.eps)])
    if message is not None:
        return _projected_message(inputs=[*inputs,gate,message],template=[("K",blocks.shape[0]),*config],
                                  grid=(32*576,1,1),threadgroup=(576,1,1),
                                  output_shapes=[(1,1,1,2304),(1,1,64)],output_dtypes=[mx.float32,mx.bfloat16])
    if gate is not None:
        return _projected_routes(inputs=[*inputs,gate],template=[("K",blocks.shape[0]),*config],
                                 grid=(64*576,1,1),threadgroup=(576,1,1),
                                 output_shapes=[(1,1,1,2304),(1,1,128)],output_dtypes=[mx.bfloat16,mx.float32])
    return _projected[normalize](inputs=inputs,template=[("K",blocks.shape[0]),*config],
                                 grid=(576,1,1),threadgroup=(576,1,1),
                                 output_shapes=[(1,1,1,2304)],
                                 output_dtypes=[mx.bfloat16 if normalize else mx.float32])[0]
