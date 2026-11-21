"""Apply attention output projection, temporal filter, and residual in one pass."""
import mlx.core as mx
import mlx.nn as nn

_pack = mx.fast.metal_kernel(
    name="inkling_pack_output_tiles",input_names=["weight"],
    output_names=["low","exponents","bases"],
    source="""
        uint i=thread_position_in_grid.x*4, row=i/3072, col=i%3072;
        uint block=(row/4*24+col/128)*4+row%4;
        uint dst=block*128+col%128;
        ushort4 raw=*(device const ushort4*)((device const ushort*)weight+i);
        ushort4 exp=(raw>>7)&255;
        ushort minexp=simd_min(min(min(exp.x,exp.y),min(exp.z,exp.w)));
        ushort maxexp=simd_max(max(max(exp.x,exp.y),max(exp.z,exp.w)));
        if (!thread_index_in_simdgroup) bases[block]=maxexp-minexp>15 ? 255 : minexp;
        *(device uchar4*)(low+dst)=uchar4((raw&127)|((raw>>8)&128));
        ushort4 nib=(exp-minexp)&15;
        exponents[dst/4]=nib.x | (nib.y<<4) | (nib.z<<8) | (nib.w<<12);
    """,
)


_kernel = mx.fast.metal_kernel(
    name="inkling_project_attention_residual",
    input_names=["x","weight","low","exponents","bases","history","conv","hidden"],
    output_names=["residual","window","squares"],
    source="""
        uint lane=thread_index_in_simdgroup;
        uint group=thread_position_in_grid.x/32, first=group*4;
        float4 dots=0;
        for (uint d=lane*4; d<3072; d+=128) {
            float4 a=float4(*(device const bfloat4*)(x+d));
            for (uint r=0;r<4;++r) {
                uint block=(group*24+d/128)*4+r, col=block*128+d%128;
                ushort base=bases[block];
                ushort4 raw;
                if (base==255) {
                    raw=*(device const ushort4*)((device const ushort*)weight+(first+r)*3072+d);
                } else {
                    ushort4 m=ushort4(*(device const uchar4*)(low+col));
                    ushort bits=exponents[col/4];
                    ushort4 ex=ushort4(bits,bits>>4,bits>>8,bits>>12)&15;
                    raw=((m&128)<<8)|((base+ex)<<7)|(m&127);
                }
                float4 b=float4(as_type<bfloat4>(raw));
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


def prepare(model):
    arrays=[]
    for layer in model.layers:
        proj=layer.self_attn.o_proj
        if (type(proj) is not nn.Linear or "bias" in proj or proj.weight.shape != (2304,3072)
                or proj.weight.dtype != mx.bfloat16):
            continue
        w=proj.weight
        saved=getattr(proj,"_compact_output",None)
        if saved is None or saved[0] is not w:
            count=w.size
            packed=_pack(inputs=[w],grid=(count//4,1,1),threadgroup=(128,1,1),
                         output_shapes=[(count,),(count//4,),(count//128,)],
                         output_dtypes=[mx.uint8,mx.uint16,mx.uint8])
            object.__setattr__(proj,"_compact_output",(w,packed))
            arrays.extend(packed)
    if arrays:
        mx.eval(arrays)


def apply(projection, x, post):
    if (post is None or type(projection) is not nn.Linear or "bias" in projection
            or projection.weight.shape != (2304,3072) or projection.weight.dtype != mx.bfloat16
            or x.dtype != mx.bfloat16 or x.shape != (1,1,3072)):
        return None
    layer,history,hidden=post
    conv=layer.post_attention_convolution.conv1d.weight
    w=projection.weight
    saved=getattr(projection,"_compact_output",None)
    if saved is None or saved[0] is not w:
        return None
    return tuple(_kernel(inputs=[x,w,*saved[1],history,conv,hidden],
                         template=[("W",conv.dtype),("BOUNDARY",layer.is_block_boundary)],
                         grid=(576*32,1,1),threadgroup=(128,1,1),
                         output_shapes=[hidden.shape,(1,4,2304),(576,)],
                         output_dtypes=[mx.float32]*3))
