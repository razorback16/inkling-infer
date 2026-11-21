"""Apply attention output projection, temporal filter, and residual in one pass."""
import mlx.core as mx
import mlx.nn as nn

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


_pack_rows = mx.fast.metal_kernel(
    name="inkling_output_row_bytes", input_names=["w"], output_names=["payload","formats"],
    source="""
        uint lane=thread_index_in_simdgroup, row=thread_position_in_grid.x/32;
        ushort highest=0;
        for (uint d=lane*4;d<3072;d+=128) {
            ushort4 bits=*(device const ushort4*)((device const ushort*)w+row*3072+d);
            ushort4 e=(bits>>7)&255;
            highest=max(highest,max(max(e.x,e.y),max(e.z,e.w)));
        }
        highest=simd_max(highest);
        ushort base=ushort(max(0,int(highest)-15));
        uint raw_tiles=0;
        for (uint tile=0;tile<24;++tile) {
            uint d=tile*128+lane*4;
            ushort4 bits=*(device const ushort4*)((device const ushort*)w+row*3072+d);
            ushort4 delta=((bits>>7)&255)-base;
            uint id=(row/4*24+tile)*4+row%4, addr=id*48;
            if (simd_any(any(delta>15))) {
                raw_tiles|=1u<<tile;
                uint word=lane<24 ? addr+lane*2 : 2304*24*48+id*16+(lane-24)*2;
                *(device ushort4*)(payload+word)=bits;
            } else {
                payload[addr+lane]=as_type<uint>(uchar4((bits<<1)|(bits>>15)));
                ((device ushort*)(payload+addr+32))[lane]=delta.x|(delta.y<<4)|(delta.z<<8)|(delta.w<<12);
            }
        }
        if (!lane) formats[row]=(uint(base)<<24)|raw_tiles;
    """,
)
_compact_rows = mx.fast.metal_kernel(
    name="inkling_stream_output_row",input_names=["x","history","conv","hidden","payload","formats"],
    output_names=["residual","window","squares"],
    source="""
        uint lane=thread_index_in_simdgroup, row=thread_position_in_grid.x/32;
        uint sg=simdgroup_index_in_threadgroup;
        uint format=formats[row];
        ushort base=ushort(format>>24)<<8;
        float dot=0;
        for (uint tile=0;tile<24;++tile) {
            uint d=tile*128+lane*4,id=(row/4*24+tile)*4+row%4,addr=id*48;
            float4 a=float4(*(device const bfloat4*)(x+d));
            ushort4 bits;
            if (format&(1u<<tile)) {
                uint word=lane<24 ? addr+lane*2 : 2304*24*48+id*16+(lane-24)*2;
                bits=*(device const ushort4*)(payload+word);
            } else {
                ushort h=((device const ushort*)(payload+addr+32))[lane];
                ushort4 n=ushort4(h,h>>4,h>>8,h>>12)&15;
                ushort4 code=(n<<8)|ushort4(as_type<uchar4>(payload[addr+lane]));
                ushort4 rot=code+base;
                bits=(rot>>1)|(rot<<15);
            }
            float4 w=float4(as_type<bfloat4>(bits));
            for (uint j=0;j<4;++j) dot=fma(a[j],w[j],dot);
        }
        for (ushort step=16;step;step>>=1) dot+=simd_shuffle_down(dot,step);
        threadgroup float values[4];
        if (!lane) {
            float v=float(bfloat16_t(dot)),filtered=0;
            for (uint t=0;t<3;++t) {
                float old=history[t*2304+row];
                window[t*2304+row]=old;
                filtered=fma(float(W(old)),float(conv[row*4+t]),filtered);
            }
            filtered=fma(float(W(v)),float(conv[row*4+3]),filtered);
            window[3*2304+row]=v;
            float out=float(bfloat16_t((float(W(filtered))+v)*0.4472135954999579f));
            if (!BOUNDARY) out+=float(hidden[row]);
            residual[row]=out;
            values[sg]=out;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!sg && !lane) {
            float norm2=0;
            for (uint i=0;i<4;++i) norm2+=values[i]*values[i];
            squares[row/4]=norm2;
        }
    """,
)


def prepare(model):
    arrays=[]
    for block in model.layers:
        proj=block.self_attn.o_proj
        w=proj.weight
        if type(proj) is not nn.Linear or w.shape!=(2304,3072) or w.dtype!=mx.bfloat16:
            continue
        saved=getattr(proj,"_output_rows",None)
        if saved is None or saved[0] is not w:
            encoded=_pack_rows(inputs=[w],grid=(2304*32,1,1),threadgroup=(128,1,1),
                               output_shapes=[(2304,24,64),(2304,)],output_dtypes=[mx.uint32,mx.uint32])
            object.__setattr__(proj,"_output_rows",(w,encoded))
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
    saved=getattr(projection,"_output_rows",None)
    compact=saved is not None and saved[0] is projection.weight
    kernel=_compact_rows if compact else _kernel
    inputs=([x,history,conv,hidden,*saved[1]] if compact else
            [x,projection.weight,history,conv,hidden])
    return tuple(kernel(inputs=inputs,
                         template=[("W",conv.dtype),("BOUNDARY",layer.is_block_boundary)],
                         grid=((2304 if compact else 576)*32,1,1),threadgroup=(128,1,1),
                         output_shapes=[hidden.shape,(1,4,2304),(576,)],
                         output_dtypes=[mx.float32]*3))
