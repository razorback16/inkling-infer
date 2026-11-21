"""Mix projected experts, roll the convolution history, and add the layer residual."""
import mlx.core as mx

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
    """,
)


def finish(states, routing, history, weights, residual):
    return _kernel(inputs=[states,routing,history,weights,residual],
                   template=[("T",states.dtype),("W",weights.dtype)],
                   grid=(2304,1,1),threadgroup=(128,1,1),
                   output_shapes=[residual.shape,history.shape],output_dtypes=[states.dtype,mx.float32])
