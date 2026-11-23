// The arm64 PyTorch BF16 dot uses 32 FP32 partial sums. The last four
// lanes reduce in adjacent pairs, after the eight four-lane vectors combine.
inline float reference_sum(float x) {
    x+=simd_shuffle_xor(x,16); x+=simd_shuffle_xor(x,8);
    x+=simd_shuffle_xor(x,4); x+=simd_shuffle_xor(x,1);
    return x+simd_shuffle_xor(x,2);
}
// Reduce four independent rows in the reference XOR order. The first two
// exchanges assign one row to each group of eight lanes.
inline float reduce_channel_row(float4 rows,uint lane){
    float2 keep=(lane&16)?rows.zw:rows.xy;
    float2 give=(lane&16)?rows.xy:rows.zw;
    float2 pair=keep+simd_shuffle_xor(give,16);
    float value=(lane&8)?pair.y:pair.x;
    value+=simd_shuffle_xor((lane&8)?pair.x:pair.y,8);
    value+=simd_shuffle_xor(value,4);
    value+=simd_shuffle_xor(value,1);
    value+=simd_shuffle_xor(value,2);
    return value;
}

// Tiled BF16 checkpoint projection. Accumulate in FP32 using Metal's SIMD
// matrix instructions. Rows at a partial tile boundary use a masked store.
using DenseMMA = mlx::steel::BlockMMA<bfloat,bfloat,64,32,32,2,2,false,true,40,40>;

template<bool Packed>
inline void dense_product(device const bfloat *x, device const bfloat *w, uint3 dims,
                          uint2 group, uint tid, uint sg, uint lane, threadgroup bfloat *panel,
                          thread DenseMMA &mma) {
    constexpr int BM=64, BN=32, BK=32, WM=2, WN=2, stride=BK+8;
    uint m=dims.x, n=dims.y, k=dims.z;
    uint row=group.y*BM, col=group.x*BN;
    uint input[4];
    uint fn=((lane/4)&2)*2+(lane%2)*2;
    for (uint i=0;i<4;++i) input[i]=min(row+uint(mma.sm)+8*WM*i,m-1)*k+fn;
    for (uint start=0; start<k; start+=BK) {
        if constexpr(!Packed){
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j=tid*8;j<BN*BK;j+=WM*WN*32*8) {
            uint r=j/BK, d=j%BK;
            *(threadgroup uint4*)(panel+r*stride+d) = col+r<n ?
                *(device const uint4*)(w+(col+r)*k+start+d) : uint4(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        for (short kk=0;kk<BK;kk+=8) {
            for (short i=0;i<4;++i)
                mma.Atile.frag_at(i,0)=float2(*(device const bfloat2*)(x+input[i]+start+kk));
            if constexpr(Packed){
                uint panel_id=(group.x*(k/8)+(start+kk)/8)*4+sg%WN;
                for(ushort j=0;j<2;++j)
                    mma.Btile.frag_at(0,j)=float2(((device const bfloat2*)w)[(panel_id+j*WN)*32+lane]);
            }else mma.Btile.template load<bfloat,1,WN,1,stride>(panel+mma.Bs_offset+kk);
            mlx::steel::tile_matmad(mma.Ctile,mma.Atile,mma.Btile,mma.Ctile);
        }
    }
}

template<typename Out,bool Packed=false>
inline void dense_tile(device const bfloat *x, device const bfloat *w, device Out *y,
                       uint3 dims, uint2 group, uint tid, uint sg, uint lane,
                       threadgroup bfloat *panel) {
    uint m=dims.x, n=dims.y, k=dims.z;
    uint row=group.y*64, col=group.x*32;
    DenseMMA mma(sg,lane);
    dense_product<Packed>(x,w,dims,group,tid,sg,lane,panel,mma);
    for (short i=0;i<4;++i) {
        uint r=row+mma.sm+16*i;
        for (short j=0;j<2;++j) {
            uint c=col+mma.sn+16*j;
            float2 value=mma.Ctile.frag_at(i,j);
            if (metal::is_same_v<Out,bfloat>) {
                for(uint e=0;e<2;++e) {
                    float v=value[e];
                    float mid=as_type<float>((as_type<uint>(v)&0xffff0000u)|0x8000u);
                    bool close=v!=0 && abs(v-mid)<=(2e-5f+6.e-6f*abs(v));
                    uint mask=uint((simd_vote::vote_t)simd_ballot(r<m && c<n && close));
                    while(mask){
                        ushort owner=ctz(mask);uint mr=simd_shuffle(r,owner),nc=simd_shuffle(c+e,owner);
                        float sum=0;
                        for(uint d=lane;d<k;d+=32)sum=fma(float(x[mr*k+d]),float(w[nc*k+d]),sum);
                        sum=reference_sum(sum);if(lane==owner)value[e]=sum;
                        mask&=mask-1;
                    }
                }
            }
            if(r<m && c<n)*(device vec<Out,2>*)(y+r*n+c)=vec<Out,2>(value);
        }
    }
}
#define DENSE_TILE(NAME,T) \
kernel void NAME(device const bfloat *x, device const bfloat *w, device T *y, constant uint3 &dims, \
    uint2 g [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup bfloat panel[32*40]; dense_tile(x,w,y,dims,g,tid,sg,lane,panel); }
DENSE_TILE(dense_bf16,bfloat)
DENSE_TILE(dense_fp32,float)

kernel void prefill_packed_dense(device const bfloat *x,device const bfloat *weight,device float *out,
                                constant uint3 &dims,uint2 group [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]],uint sg [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]){
    dense_tile<float,true>(x,weight,out,dims,group,tid,sg,lane,nullptr);
}

// Only near-midpoint results need the CPU reduction order. Give those
// rows separate SIMD groups so a single repair cannot stall a GEMM tile.
inline bfloat4 unpack_dense_word(uint m,ushort nibbles,uint h){
    ushort4 mant=ushort4(as_type<uchar4>(m));
    ushort4 exps=((ushort4(nibbles)>>ushort4(0,4,8,12))&15)+ushort(h);
    return as_type<bfloat4>(((mant&128)<<8)|(exps<<7)|(mant&127));
}
inline bfloat4 read_dense_block(device const uchar *payload,device const uint *header,
                                device const bfloat4 *exception,uint block,uint lane){
    uint h=header[block];
    if(h>>31)return exception[(h&0x7fffffffu)*32+lane];
    device const uint *pair=((device const uint*)payload)+((block/2)*32+lane)*3;
    return unpack_dense_word(pair[block&1],ushort(pair[2]>>((block&1)*16)),h);
}
inline bfloat4 paired_row_weight(device const uchar *payload,device const uint *header,
                                 device const bfloat4 *exception,uint row,uint d,uint lane){
    uint block=row*18+d/128,h=header[block];
    if(h>>31)return exception[(h&0x7fffffffu)*32+lane];
    device const uint *tile=(device const uint*)payload+((row/2)*9+d/256)*192;
    uint m=tile[lane*4+(row&1)*2+(block&1)],e=tile[128+lane*2+(row&1)];
    return unpack_dense_word(m,ushort(e>>((block&1)*16)),h);
}
inline uint packed_channel(uint d){return (d/128)*128+(d%32)*4+(d%128)/32;}
template<bool Packed=false>
inline float packed_projection(device const bfloat *x,device const bfloat *weights,uint rows,uint k,uint row,uint lane){
    uint blocks=rows*(k/128);
    device const uchar *payload=(device const uchar*)weights;
    device const uint *header=(device const uint*)(payload+((blocks+1)&~1u)*192);
    device const bfloat4 *exception=(device const bfloat4*)(header+blocks);
    float sum=0;
    for(uint d=0;d<k;d+=128){
        uint block=(row*k+d)/128;
        bfloat4 w=(k==2304 && !(rows%4))?paired_row_weight(payload,header,exception,row,d,lane):read_dense_block(payload,header,exception,block,lane);
        if(Packed){
            float4 v=float4(*(device const bfloat4*)(x+d+lane*4));
            for(ushort j=0;j<4;++j)sum=fma(v[j],float(w[j]),sum);
        }else for(ushort j=0;j<4;++j)sum=fma(float(x[d+lane+j*32]),float(w[j]),sum);
    }
    return reference_sum(sum);
}
template<bool Packed=true>
inline float packed_projection_channel(device const bfloat *x,device const bfloat *weights,uint rows,uint first,uint lane){
    constexpr uint k=2304;
    uint blocks=rows*18;
    device const uchar *payload=(device const uchar*)weights;
    device const uint *header=(device const uint*)(payload+((blocks+1)&~1u)*192);
    device const bfloat4 *exception=(device const bfloat4*)(header+blocks);
    float4 acc=0;
    for(uint d=0;d<k;d+=256) {
        float4 x0,x1;
        if constexpr(Packed){
            x0=float4(*(device const bfloat4*)(x+d+lane*4));
            x1=float4(*(device const bfloat4*)(x+d+128+lane*4));
        }else for(ushort j=0;j<4;++j){
            x0[j]=float(x[d+lane+j*32]);
            x1[j]=float(x[d+128+lane+j*32]);
        }
        for(ushort r=0;r<4;r+=2){
            uint row=first+r,block=row*18+d/128;
            device const uint *tile=(device const uint*)payload+(row/2*9+d/256)*192;
            uint4 mant=((device const uint4*)tile)[lane];
            uint2 exp=((device const uint2*)(tile+128))[lane];
            for(ushort rr=0;rr<2;++rr){
                uint2 h=((device const uint2*)header)[block/2+rr*9];
                bfloat4 a=unpack_dense_word(mant[rr*2],ushort(exp[rr]),h.x);
                bfloat4 b=unpack_dense_word(mant[rr*2+1],ushort(exp[rr]>>16),h.y);
                if((h.x|h.y)>>31){
                    if(h.x>>31)a=exception[(h.x&0x7fffffffu)*32+lane];
                    if(h.y>>31)b=exception[(h.y&0x7fffffffu)*32+lane];
                }
                for(ushort j=0;j<4;++j)acc[r+rr]=fma(x0[j],float(a[j]),acc[r+rr]);
                for(ushort j=0;j<4;++j)acc[r+rr]=fma(x1[j],float(b[j]),acc[r+rr]);
            }
        }
    }
    return reduce_channel_row(acc,lane);
}
template<bool Packed=true,typename Batch=device const bfloat4*,typename Input=device const bfloat*>
inline float high_pair_channels(Input x,device const bfloat *weights,uint rows,uint first,uint lane){
    constexpr uint k=2304,group_rows=32;
    device const uchar *buf=(device const uchar*)weights;
    device const uint *table=(device const uint*)(buf+rows*k*3/2)+(first/group_rows)*256;
    float4 acc=0;
    for(uint d=0;d<k;d+=128){
        device const uchar *tile=buf+(first/4*18+d/128)*768;
        uint4 low=((device const uint4*)tile)[lane];
        ushort4 codes=((device const ushort4*)(tile+512))[lane];
        float4 act;
        if constexpr(Packed)act=float4(((Batch)(x+d))[lane]);
        else for(ushort j=0;j<4;++j)act[j]=float(x[d+lane+j*32]);
        for(ushort r=0;r<4;++r){
            uchar4 v=as_type<uchar4>(low[r]);uchar2 c=as_type<uchar2>(codes[r]);
            uint2 words=uint2(table[c.x],table[c.y])|uint2(uint(v.x)|(uint(v.y)<<16),uint(v.z)|(uint(v.w)<<16));
            float4 w=float4(as_type<bfloat4>(words));
            for(ushort j=0;j<4;++j)acc[r]=fma(act[j],w[j],acc[r]);
        }
    }
    return reduce_channel_row(acc,lane);
}
inline void round_projection_block(device const bfloat *x,device const bfloat *weights,
                                   device const float *approx,device bfloat *out,uint3 dims,
                                   uint group,uint tid,uint sg,uint lane,
                                   threadgroup uint *pending,threadgroup atomic_uint *count){
    if(!tid)atomic_store_explicit(count,0u,memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint j=tid;j<1024;j+=256){
        uint idx=group*1024+j;
        float v=idx<dims.x*dims.y?approx[idx]:0.f;
        float mid=as_type<float>((as_type<uint>(v)&0xffff0000u)|0x8000u);
        bool fix=v!=0 && abs(v-mid)<=4.e-6f+5.e-6f*abs(v);
        uint mask=uint((simd_vote::vote_t)simd_ballot(fix));
        if(mask){
            uint base=0;if(!lane)base=atomic_fetch_add_explicit(count,uint(popcount(mask)),memory_order_relaxed);
            base=simd_shuffle(base,0);
            if(fix)pending[base+popcount(mask&((1u<<lane)-1))]=idx;
        }
        if(!fix && idx<dims.x*dims.y)out[idx]=bfloat(v);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint size=atomic_load_explicit(count,memory_order_relaxed);
    for(uint j=sg;j<size;j+=8){
        uint idx=pending[j];
        device const bfloat *act=x+(idx/dims.y)*dims.z,*weight=weights+(idx%dims.y)*dims.z;
        float v=0;
        for(uint d=0;d<dims.z;d+=128){
            bfloat4 w=((device const bfloat4*)(weight+d))[lane];
            for(ushort j=0;j<4;++j)v=fma(float(act[d+lane+j*32]),float(w[j]),v);
        }
        float sum=reference_sum(v);
        if(!lane)out[idx]=bfloat(sum);
    }
}

kernel void round_projection(device const bfloat *x, device const bfloat *weights,
                             device const float *approx, device bfloat *out, constant uint3 &dims,
                             uint group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){
    threadgroup uint pending[1024];
    threadgroup atomic_uint count;
    round_projection_block(x,weights,approx,out,dims,group,tid,sg,lane,pending,&count);
}
// Dispatch all six independent attention input projections in one grid.
constant uint attention_columns[6] = {3072,384,24,3072,1024,1024};


// One SIMD group computes four output channels. The four adjacent input
// elements per lane and shuffle reduction match the reference GEMV order.
template<typename Out>
inline void dense_vector(device const bfloat *x, device const bfloat *w, device Out *y,
                         constant uint3 &dims, uint2 group, uint sg, uint lane) {
    uint n=dims.y,k=dims.z,base=(group.x*4+sg)*4;
    if (base>=n) return;
    float4 acc=0;
    for (uint d=lane;d<k;d+=32) {
        float v=float(x[group.y*k+d]);
        for(uint r=0;r<4;++r) acc[r]=fma(v,float(w[(base+r)*k+d]),acc[r]);
    }
    for (uint r=0;r<4;++r) {
        float sum=reference_sum(acc[r]);
        if (!lane) y[group.y*n+base+r]=Out(sum);
    }
}
#define DENSE_VECTOR(NAME,T) \
kernel void NAME(device const bfloat *x,device const bfloat *w,device T *y,constant uint3 &dims, \
   uint2 g [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], \
   uint lane [[thread_index_in_simdgroup]]) { dense_vector(x,w,y,dims,g,sg,lane); }
DENSE_VECTOR(vector_bf16,bfloat)
DENSE_VECTOR(vector_fp32,float)

// Eight adjacent lanes cooperate on one output row. Four contiguous loads
// retain the reference's 32 partial sums while coalescing weight reads.
inline float projection_row(device const bfloat *x,device const bfloat *weight,uint k,uint lane){
    float4 acc=0;
    for(uint d=(lane%8)*4;d<k;d+=32)
        acc=fma(float4(*(device const bfloat4*)(x+d)),float4(*(device const bfloat4*)(weight+d)),acc);
    acc+=simd_shuffle_xor(acc,4);acc+=simd_shuffle_xor(acc,2);acc+=simd_shuffle_xor(acc,1);
    return (acc.x+acc.y)+(acc.z+acc.w);
}
#define COALESCED(NAME,T) \
kernel void NAME(device const bfloat *x,device const bfloat *w,device T *y,constant uint3 &dims, \
    uint2 g [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){ \
    uint r=g.x*4+lane/8; if(r>=dims.y)return; \
    float sum=projection_row(x+g.y*dims.z,w+r*dims.z,dims.z,lane); \
    if(lane%8==0)y[g.y*dims.y+r]=T(sum); }
COALESCED(coalesced_bf16,bfloat)
COALESCED(coalesced_fp32,float)

kernel void controller_output(device const bfloat *z,device const bfloat *forget_w,device const bfloat *gain_w,
                              device const float *x,device const float *msg,device float *out,
                              uint2 g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]){
    uint row=g.x*4+lane/8,at=g.y*2304+row;
    float f=projection_row(z+g.y*64,forget_w+row*64,64,lane);
    float v=projection_row(z+g.y*64,gain_w+row*64,64,lane);
    if(lane%8==0)out[at]=(1.f-reference_tanh(f))*x[at]+reference_tanh(v)*msg[at];
}

kernel void packed_vocabulary_logits(device const bfloat *x,device const bfloat *weights,
                                     device float *out,constant uint3 &dims,
                                     uint g [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]],
                                     uint lane [[thread_index_in_simdgroup]]) {
    uint row=g*8+sg;
    if(row>=dims.y)return;
    float sum=packed_projection(x,weights,dims.y,dims.z,row,lane);
    if(!lane)out[row]=30.f*precise::tanh(float(bfloat(sum))/30.f);
}

kernel void pack_affine6_vocabulary(device const bfloat *src,device uchar *dst,device ushort *headers,
           constant uint2 &range,uint t [[thread_position_in_grid]],uint lane [[thread_index_in_simdgroup]]){
    uint local_block=t/32;
    if(local_block>=range.y)return;
    uint block=range.x+local_block;
    uint low=0,high=0;
    for(ushort g=0;g<8;++g){
        float value=float(src[local_block*256+g*32+lane]);
        float peak=simd_max(max(value,0.f)),bottom=simd_min(min(value,0.f));
        float span=peak-bottom;
        int ideal=span?int(rint(-bottom*63.f/span)):32;
        int center=clamp(int(rint(float(ideal-8)/3.f)),0,15);
        float best=INFINITY,best_scale=0;int best_z=32;
        // Fit adjacent grid origins using the exact scale precision stored on device.
        for(int delta=-1;delta<=1;++delta){
            int z=8+3*clamp(center+delta,0,15);
            float scale=max(peak/(63-z),-bottom/z);
            for(ushort iter=0;iter<3;++iter){
                float rounded=float(as_type<half>(ushort((as_type<ushort>(half(scale))+4u)&0xfff8u)));
                float q=rounded?clamp(rint(value/rounded),float(-z),float(63-z)):0.f;
                float diff=fma(-q,rounded,value);
                float err=simd_sum(diff*diff);
                if(err<best){best=err;best_scale=rounded;best_z=z;}
                float numer=simd_sum(value*q),denom=simd_sum(q*q);
                if(denom>0)scale=numer/denom;
            }
        }
        // The upper twelve bits encode the rounded positive half scale. The low
        // nibble chooses a zero point in [8,53] in steps of three.
        if(!lane)headers[block*8+g]=ushort((uint(as_type<ushort>(half(best_scale)))<<1)|uint((best_z-8)/3));
        uint code=uint(clamp((best_scale?rint(value/best_scale):0.f)+best_z,0.f,63.f));
        low|=(code&15)<<(4*g);high|=(code>>4)<<(2*g);
    }
    // Eight consecutive weights per lane share one scale and zero point.
    // Reorder once while loading, before the vocabulary matrix is reused.
    uint first=0,second=0;
    for(ushort j=0;j<8;++j){
        ushort src_lane=(lane*8+j)%32,part=lane/4;
        uint lo=simd_shuffle(low,src_lane),hi=simd_shuffle(high,src_lane);
        uint q=((lo>>(4*part))&15)|(((hi>>(2*part))&3)<<4);
        if(j<4)first|=q<<(6*j);
        else second|=q<<(6*(j-4));
    }
    device packed_ushort3 *data=(device packed_ushort3*)(dst+block*192);
    data[lane]=packed_ushort3(ushort(first),ushort(second),ushort((first>>16)|((second>>16)<<8)));
}
kernel void affine6_vocabulary_logits(device const bfloat4 *act,device const uchar *weights,device const ushort *headers,device float *out,
                constant uint3 &dims,uint group [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]],
                uint lane [[thread_index_in_simdgroup]]){
    uint row=group*8+sg;
    if(row>=dims.y)return;
    uint blocks=dims.z/256;float sum=0;
    for(ushort k=0;k<blocks;++k){
        uint b=row*blocks+k;
        device const uchar *tile=weights+b*192;
        ushort3 bits=((device const packed_ushort3*)tile)[lane];uint lo=uint(bits.x)|(uint(bits.y)<<16),hi=bits.z;
        uint h=headers[b*8+lane/4];
        float scale=float(as_type<half>(ushort((h>>1)&0x7ff8u)));
        float zero=float(8+3*int(h&15));
        uint2 v=uint2(lo&65535,lo>>16)|(uint2(hi&255,hi>>8)<<16);uint4 shift=uint4(0,6,12,18),q0=(uint4(v.x)>>shift)&63,q1=(uint4(v.y)>>shift)&63;
        float4 a=float4(act[(k*32+lane)*2]),bvec=float4(act[(k*32+lane)*2+1]);
        float subtotal=dot(a,float4(q0)-zero)+dot(bvec,float4(q1)-zero);
        sum=fma(subtotal,scale,sum);
    }
    sum=simd_sum(sum);if(!lane)out[row]=30.f*precise::tanh(float(bfloat(sum))/30.f);
}

// Batch rows and controller channels tile the two dimensions of each matrix.
// Keep the 32 CPU accumulation streams separate until the ordered reduction.
template<uint Groups>
inline void prompt_controller_product(device const bfloat *state,device const bfloat *msg,device const bfloat2 *w,
                                     device bfloat *out,uint rows,uint2 g,uint sg,uint lane,
                                     threadgroup float *partial){
    uint r=((lane/4)&4)+(lane/2)%4,c=((lane/4)&2)*2+(lane%2)*2;
    uint row=min(g.y*8+r,rows-1);
    for(uint stream=sg;stream<32;stream+=Groups){
        simdgroup_float8x8 dot=make_filled_simdgroup_matrix<float,8,8>(0.f);
        for(ushort k=0;k<18;++k){
            simdgroup_float8x8 a,b;
            device const bfloat *input_row=(k<9?state:msg)+row*2304+stream*72;
            float2 input=float2(*(device const bfloat2*)(input_row+(k%9)*8+c));
            float2 weight=float2(w[((g.x*32+stream)*18+k)*32+lane]);
            for(ushort e=0;e<2;++e){a.thread_elements()[e]=input[e];b.thread_elements()[e]=weight[e];}
            simdgroup_multiply_accumulate(dot,a,b,dot);
        }
        for(ushort e=0;e<2;++e)partial[(r*8+c+e)*32+stream]=dot.thread_elements()[e];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint cell=sg;cell<64;cell+=Groups){
        float value=reference_sum(partial[cell*32+lane]);
        uint m=g.y*8+cell/8;
        if(!lane && m<rows)out[m*64+g.x*8+cell%8]=bfloat(value);
    }
}
#define PROMPT_CONTROLLER(NAME,GROUPS) \
kernel void NAME(device const bfloat *state,device const bfloat *msg,device const bfloat2 *w,device bfloat *out,constant uint &rows, \
                 uint2 g [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){ \
    threadgroup float partial[64*32];prompt_controller_product<GROUPS>(state,msg,w,out,rows,g,sg,lane,partial); }
PROMPT_CONTROLLER(prompt_controller_down,8)

// Share the tile's near-midpoint outputs across all four SIMD groups.
inline void dense_tile_repaired(device const bfloat *x, device const bfloat *w, device bfloat *y,
                                uint3 dims, uint2 group, uint tid, uint sg, uint lane,
                                device const bfloat *exact, threadgroup ushort *list,
                                threadgroup atomic_uint *size) {
    uint m=dims.x, n=dims.y, k=dims.z;
    uint row=group.y*64, col=group.x*32;
    DenseMMA mma(sg,lane);
    dense_product<true>(x,w,dims,group,tid,sg,lane,nullptr,mma);
    if(!tid)atomic_store_explicit(size,0u,memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ushort flags=0;
    for(ushort i=0;i<4;++i){
        uint r=row+mma.sm+16*i;
        for(ushort j=0;j<2;++j){
            uint c=col+mma.sn+16*j;
            float2 val=mma.Ctile.frag_at(i,j);
            bool2 near=(val!=0.f) && (abs(val-as_type<float2>((as_type<uint2>(val)&0xffff0000u)|0x8000u))<=4.e-6f+5.e-6f*abs(val));
            if(r<m && c<n){
                flags|=ushort(uint(near.x)|(uint(near.y)<<1))<<(i*4+j*2);
                *(device bfloat2*)(y+r*n+c)=bfloat2(val);
            }
        }
    }
    uint length=popcount(flags);
    uint offset=simd_prefix_exclusive_sum(length);
    uint total=simd_sum(length),base=0;
    if(!lane && total)base=atomic_fetch_add_explicit(size,total,memory_order_relaxed);
    offset+=simd_shuffle(base,0);
    while(flags){
        uint b=ctz(flags);
        list[offset++]=ushort((mma.sm+16*(b/4))*32+mma.sn+16*((b/2)%2)+b%2);
        flags&=flags-1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    uint count=atomic_load_explicit(size,memory_order_relaxed);
    for(uint idx=sg;idx<count;idx+=4){
        uint rc=list[idx],r=row+rc/32,c=col+rc%32;
        device const bfloat *act=x+r*k,*weight=exact+c*k;
        float sum=0;
        for(uint d=0;d<k;d+=128){
            float4 w=float4(((device const bfloat4*)(weight+d))[lane]);
            for(ushort j=0;j<4;++j)sum=fma(float(act[d+lane+j*32]),w[j],sum);
        }
        sum=reference_sum(sum);
        if(!lane)y[r*n+c]=bfloat(sum);
    }
}
kernel void repaired_attention_projections(
    device const bfloat *x0, device const bfloat *w0, device const bfloat *e0, device bfloat *y0,
    device const bfloat *x1, device const bfloat *w1, device const bfloat *e1, device bfloat *y1,
    device const bfloat *x2, device const bfloat *w2, device const bfloat *e2, device bfloat *y2,
    device const bfloat *x3, device const bfloat *w3, device const bfloat *e3, device bfloat *y3,
    device const bfloat *x4, device const bfloat *w4, device const bfloat *e4, device bfloat *y4,
    device const bfloat *x5, device const bfloat *w5, device const bfloat *e5, device bfloat *y5, constant uint &rows,
    uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){
    device const bfloat *x[6]={x0,x1,x2,x3,x4,x5};
    device const bfloat *w[6]={w0,w1,w2,w3,w4,w5};
    device const bfloat *e[6]={e0,e1,e2,e3,e4,e5};
    device bfloat *y[6]={y0,y1,y2,y3,y4,y5};
    uint index=0, tile=group.x;
    for(;index<5;++index){
        uint groups=(attention_columns[index]+31)/32;
        if(tile<groups)break;
        tile-=groups;
    }
    uint3 dims=uint3(rows,attention_columns[index],2304);
    threadgroup ushort list[2048]; threadgroup atomic_uint count;
    dense_tile_repaired(x[index],w[index],y[index],dims,uint2(tile,group.y),tid,sg,lane,e[index],list,&count);
}

// Select the first occurrence of the largest finite logit, as numpy.argmax does.
inline uint2 greedy_winner(float x,uint idx,threadgroup uint2 *tile,uint sg,uint lane){
    float maximum=simd_max(x);
    uint best=simd_min(x==maximum?idx:0xffffffffu);
    if(!lane)tile[sg]=uint2(as_type<uint>(maximum),best);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    x=lane<8?as_type<float>(tile[lane].x):-INFINITY;
    idx=lane<8?tile[lane].y:0xffffffffu;
    maximum=simd_max(x);
    return uint2(as_type<uint>(maximum),simd_min(x==maximum?idx:0xffffffffu));
}
kernel void greedy_max_tiles(device const float *scores,device uint2 *tiles,constant uint &vocab,
                             uint group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){
    threadgroup uint2 part[8];float score=-INFINITY;uint id=0xffffffffu;
    for(ushort j=0;j<4;++j){
        uint i=group*1024+tid+uint(j)*256;
        if(i<vocab && scores[i]>score){score=scores[i];id=i;}
    }
    uint2 win=greedy_winner(score,id,part,sg,lane);
    if(!tid)tiles[group]=win;
}
kernel void greedy_embedding(device const uint2 *tiles,device const bfloat *weights,device bfloat *input,device uint *token,
                             constant uint &count,uint tid [[thread_index_in_threadgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){
    threadgroup uint2 part[8];
    uint2 candidate=tid<count?tiles[tid]:uint2(as_type<uint>(-INFINITY),0xffffffffu);
    uint2 win=greedy_winner(as_type<float>(candidate.x),candidate.y,part,sg,lane);
    uint id=win.y==0xffffffffu?0:win.y;
    if(!tid)*token=id;
    for(uint d=tid;d<576;d+=256)((device bfloat4*)input)[d]=((device const bfloat4*)weights)[id*576+d];
}
