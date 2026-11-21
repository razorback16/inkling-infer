"""Apply attention output projection, temporal filter, and residual in one pass."""
import mlx.core as mx
import mlx.nn as nn
from . import packed_rows

_kernel = mx.fast.metal_kernel(
    name="inkling_project_attention_residual",
    input_names=["x","weight","history","conv","hidden"],
    output_names=["residual","window","squares"],
    source="""
        uint lane=thread_index_in_simdgroup;
        uint group=thread_position_in_grid.x/32, first=group*4;
        float4 dots=0;
        for (uint d=lane*4; d<3072; d+=128) {
            float4 a=float4(*(device const bfloat4*)(x+d));
            for (uint r=0;r<4;++r) {
                float4 b=float4(*(device const bfloat4*)(weight+(first+r)*3072+d));
                for (uint k=0;k<4;++k) dots[r]=fma(a[k],b[k],dots[r]);
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


_compact = mx.fast.metal_kernel(
    name="inkling_two_row_projection_residual",
    input_names=["x","weight","history","conv","hidden","payload","bases"],
    output_names=["residual","window","squares"],
    source="""
        uint lane=thread_index_in_simdgroup, simd=thread_position_in_grid.x/32;
        constexpr uint ROW_COUNT=2;
        uint group=simd/2, first=group*4, first_row=(simd%2)*2;
    """ + packed_rows.accumulate + """
        threadgroup float partial[4][2];
        for (uint r=first_row;r<first_row+ROW_COUNT;++r) {
            float dot=sums[r];
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
                residual[d]=out; partial[simdgroup_index_in_threadgroup][r-first_row]=out;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint sg=simdgroup_index_in_threadgroup;
        if (!lane && !(sg & 1)) {
            float norm2=0;
            for (uint j=0;j<4;++j) {
                float out=partial[sg+j/2][j%2]; norm2+=out*out;
            }
            squares[group]=norm2;
        }
    """,
)


def prepare(model):
    tables=[]
    for block in model.layers:
        proj=block.self_attn.o_proj
        w=proj.weight
        if type(proj) is not nn.Linear or w.shape!=(2304,3072) or w.dtype!=mx.bfloat16:
            continue
        saved=getattr(proj,"_output_tiles",None)
        if saved is None or saved[0] is not w:
            encoded=packed_rows.encode(w)
            object.__setattr__(proj,"_output_tiles",(w,encoded))
            tables.extend(encoded)
    if tables:
        mx.eval(tables)


def apply(projection, x, post):
    if (post is None or type(projection) is not nn.Linear or "bias" in projection
            or projection.weight.shape != (2304,3072) or projection.weight.dtype != mx.bfloat16
            or x.dtype != mx.bfloat16 or x.shape != (1,1,3072)):
        return None
    layer,history,hidden=post
    conv=layer.post_attention_convolution.conv1d.weight
    saved=getattr(projection,"_output_tiles",None)
    compact=saved is not None and saved[0] is projection.weight
    kernel=_compact if compact else _kernel
    inputs=[x,projection.weight,history,conv,hidden]
    if compact:
        inputs.extend(saved[1])
    return tuple(kernel(inputs=inputs,
                         template=[("W",conv.dtype),("BOUNDARY",layer.is_block_boundary),("COLS",3072)],
                         grid=((1152 if compact else 576)*32,1,1),threadgroup=(128,1,1),
                         output_shapes=[hidden.shape,(1,4,2304),(576,)],
                         output_dtypes=[mx.float32]*3))
