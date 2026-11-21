"""Apply attention output projection, temporal filter, and residual in one pass."""
import mlx.core as mx
import mlx.nn as nn
from . import packed_rows

_kernel = mx.fast.metal_kernel(
    name="inkling_project_attention_residual",
    input_names=["x","weight","history","conv","hidden","payload","bases"],
    output_names=["residual","window","squares"],
    source="""
        uint lane=thread_index_in_simdgroup;
        uint group=thread_position_in_grid.x/32, first=group*4;
        float4 dots=0;
        if (COMPACT) {
    """ + packed_rows.accumulate + """
            for (uint r=0;r<4;++r) dots[r]=sums[r];
        } else {
        for (uint d=lane*4; d<3072; d+=128) {
            float4 a=float4(*(device const bfloat4*)(x+d));
            for (uint r=0;r<4;++r) {
                float4 b=float4(*(device const bfloat4*)(weight+(first+r)*3072+d));
                for (uint k=0;k<4;++k) dots[r]=fma(a[k],b[k],dots[r]);
            }
        }
        }
        float norm2=0;
        for (uint r=0;r<4;++r) {
            float dot=dots[r];
            for (ushort n=16;n;n>>=1) dot+=simd_shuffle_down(dot,n);
            if (!lane) {
                uint d=first+r;
                float val=float(bfloat16_t(dot)), filtered=0;
                for (uint t=0;t<3;++t) {
                    float old=history[t*2304+d]; window[t*2304+d]=old;
                    filtered=fma(float(W(old)),float(conv[d*4+t]),filtered);
                }
                filtered=fma(float(W(val)),float(conv[d*4+3]),filtered);
                window[3*2304+d]=val;
                float out=float(bfloat16_t((float(W(filtered))+val)*0.4472135954999579f));
                if (!BOUNDARY) out+=float(hidden[d]);
                residual[d]=out; norm2+=out*out;
            }
        }
        if (!lane) squares[group]=norm2;
    """,
)


def prepare(model):
    arrays = []
    for block in model.layers:
        projection = block.self_attn.o_proj
        w = projection.weight
        if type(projection) is not nn.Linear or w.shape!=(2304,3072) or w.dtype!=mx.bfloat16:
            continue
        saved = getattr(projection,"_output_tiles",None)
        if saved is None or saved[0] is not w:
            encoded = packed_rows.encode(w)
            object.__setattr__(projection,"_output_tiles",(w,encoded))
            arrays.extend(encoded)
    if arrays:
        mx.eval(arrays)


def apply(projection, x, post):
    if (post is None or type(projection) is not nn.Linear or "bias" in projection
            or projection.weight.shape != (2304,3072) or projection.weight.dtype != mx.bfloat16
            or x.dtype != mx.bfloat16 or x.shape != (1,1,3072)):
        return None
    layer,history,hidden=post
    conv=layer.post_attention_convolution.conv1d.weight
    saved = getattr(projection,"_output_tiles",None)
    compact = saved is not None and saved[0] is projection.weight
    tiles = saved[1] if compact else (x.view(mx.uint32),)*2
    return tuple(_kernel(inputs=[x,projection.weight,history,conv,hidden,*tiles],
                         template=[("W",conv.dtype),("BOUNDARY",layer.is_block_boundary),("COLS",3072),("COMPACT",compact)],
                         grid=(576*32,1,1),threadgroup=(128,1,1),
                         output_shapes=[hidden.shape,(1,4,2304),(576,)],
                         output_dtypes=[mx.float32]*3))
