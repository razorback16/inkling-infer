"""Mix projected experts, roll the convolution history, and add the layer residual."""
import mlx.core as mx

_filter = """
        float current = float(T(mixed));
        float sum = 0;
        for (uint j=0; j<3; ++j) {
            float old = hist[j*2304+c];
            sum = fma(float(W(old)),float(w[c*4+j]),sum);
            if (j) next_history[(j-1)*2304+c] = old;
        }
        next_history[2*2304+c] = current;
        sum = fma(float(W(current)),float(w[c*4+3]),sum);
        out[c] = T(residual[c] + float(T((float(W(sum))+current)*0.4472135954999579f)));
"""

_kernel = mx.fast.metal_kernel(
    name="inkling_projected_postmix", input_names=["states","route","hist","w","residual"],
    output_names=["out","next_history"],
    source="""
        uint c = thread_position_in_grid.x;
        volatile float mixed = 0;
        for (uint e=0; e<8; ++e) {
            volatile float term = float(states[e*2304+c])*route[e];
            mixed = mixed + term;
        }
    """ + _filter,
)


def finish(states, routing, history, weights, residual):
    return _kernel(inputs=[states,routing,history,weights,residual],
                   template=[("T",states.dtype),("W",weights.dtype)],
                   grid=(2304,1,1),threadgroup=(128,1,1),
                   output_shapes=[residual.shape,history.shape],output_dtypes=[states.dtype,mx.float32])

_mx_down = mx.fast.metal_kernel(
    name="inkling_mxfp8_down_filter",
    input_names=["x","indices","route","packed","scales","codes","hist","w","residual"],
    output_names=["out","next_history"],
    source="""
        #pragma clang fp contract(off)
        uint lane=thread_index_in_simdgroup, slot=simdgroup_index_in_threadgroup;
        uint first=threadgroup_position_in_grid.x*4;
        uint expert=indices[slot];
        float dots[4]={};
        for (uint word=lane; word<288; word+=32) {
            float4 activation=float4(*(device const bfloat4*)(x+slot*1152+word*4));
            for (uint r=0; r<4; ++r) {
                uint row=expert*2304+first+r;
                uchar4 bytes=as_type<uchar4>(packed[row*288+word]);
                float sum=0;
                for (uint k=0;k<4;++k) sum+=activation[k]*codes[bytes[k]];
                float scale=as_type<float>(uint(scales[row*36+word/8]) << 23);
                dots[r]+=sum*scale;
            }
        }
        threadgroup float value[8][4];
        for (uint r=0;r<4;++r) {
            float dot=simd_sum(dots[r]);
            if (!lane) value[slot][r]=float(T(dot));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!slot && lane<4) {
            uint c=first+lane;
            float mixed=0;
            for (uint e=0;e<8;++e) mixed+=value[e][lane]*route[e];
    """ + _filter + "}",
)


def project_and_finish(down, active, selected, routing, history, conv, residual):
    if (getattr(down,"mode",None)!="mxfp8" or down.group_size!=32 or down.bits!=8
            or down.weight.shape[1:]!=(2304,288) or down.scales.shape[1:]!=(2304,36)
            or down.weight.dtype!=mx.uint32 or down.scales.dtype!=mx.uint8
            or getattr(down,"global_scale",None) is not None or "bias" in down
            or active.shape!=(1,1,8,1,1152) or active.dtype!=mx.bfloat16):
        states=down(active,selected).squeeze(-2)
        return finish(states,routing,history,conv,residual)
    from .expert_gate import mx_codes
    return _mx_down(inputs=[active,selected,routing,down.weight,down.scales,mx_codes(),history,conv,residual],
                    template=[("T",active.dtype),("W",conv.dtype)],
                    grid=(576*256,1,1),threadgroup=(256,1,1),
                    output_shapes=[residual.shape,history.shape],output_dtypes=[active.dtype,mx.float32])
