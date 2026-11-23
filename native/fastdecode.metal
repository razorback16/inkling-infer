kernel void packed_bf16(device const bfloat *x,device const bfloat *w,device bfloat *out,constant uint3 &dims,
                        uint g [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]){
    uint row=g*8+sg;if(row>=dims.y)return;
    float sum=packed_projection<true>(x,w,dims.y,dims.z,row,lane);
    if(!lane)out[row]=bfloat(sum);
}
// Apply the attention output projection, filter and residual together, as in
// the MLX filtered_output kernel. Round both BF16 intermediates in CPU order.
inline void store_output_residual(float projected,uint d,device const bfloat *hidden,
        device const bfloat *conv,device const bfloat *old,device bfloat *next,
        device float *residual,uint skip){
    #pragma clang fp contract(off)
    bfloat raw=bfloat(projected);float sum=0;
    for(ushort j=0;j<3;++j){
        bfloat tap=old[j*2304+d];
        sum=fma(float(tap),float(conv[d*4+j]),sum);
        if(j)next[(j-1)*2304+d]=tap;
    }
    next[2*2304+d]=raw;sum=fma(float(raw),float(conv[d*4+3]),sum);
    float filtered=float(bfloat((sum+float(raw))/sqrt(5.f)));
    residual[d]=filtered+(skip?float(hidden[d]):0.f);
}
kernel void decode_output_residual(device const bfloat *x,device const bfloat *weight,device const bfloat *hidden,
                                  device const bfloat *conv,device const bfloat *old,device bfloat *next,
                                  device float *residual,constant uint &skip,uint group [[threadgroup_position_in_grid]],
                                  uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){
    #pragma clang fp contract(off)
    uint d=group*8+sg;
    float projected=packed_projection<true>(x,weight,2304,3072,d,lane);
    if(!lane)store_output_residual(projected,d,hidden,conv,old,next,residual,skip);
}
template<bool HighPairs>
inline void global_head_projections_impl(device const bfloat *q,device const bfloat *k,device const bfloat *v,
      device const bfloat *wq,device const bfloat *wr,device const bfloat *ws,
      device const bfloat *wk,device const bfloat *wv,device bfloat *oq,device bfloat *orr,
      device bfloat *os,device bfloat *ok,device bfloat *ov,
      device const bfloat *wk_taps,device const bfloat *wv_taps,device bfloat *taps,
      uint gid,uint sg,uint lane){
    uint row=(gid*8+sg)*4,rows=3072;
    if(row>=5592)return;
    device const bfloat *src=q,*w=wq;device bfloat *out=oq;
    if(row>=5560){row-=5560;src=v;w=wv_taps;out=taps+32;rows=32;}
    else if(row>=5528){row-=5528;src=k;w=wk_taps;out=taps;rows=32;}
    else if(row>=4504){row-=4504;src=v;w=wv;out=ov;rows=1024;}
    else if(row>=3480){row-=3480;src=k;w=wk;out=ok;rows=1024;}
    else if(row>=3456){row-=3456;w=ws;out=os;rows=24;}
    else if(row>=3072){row-=3072;w=wr;out=orr;rows=384;}
    float result;
    if constexpr(HighPairs)result=high_pair_channels<false>(src,w,rows,row,lane);
    else result=packed_projection_channel<false>(src,w,rows,row,lane);
    if(!(lane&7))out[row+lane/8]=bfloat(result);
}
kernel void global_head_projections(device const bfloat *q,device const bfloat *k,device const bfloat *v,
      device const bfloat *wq,device const bfloat *wr,device const bfloat *ws,
      device const bfloat *wk,device const bfloat *wv,device bfloat *oq,device bfloat *orr,
      device bfloat *os,device bfloat *ok,device bfloat *ov,
      device const bfloat *wk_taps,device const bfloat *wv_taps,device bfloat *taps,
      uint gid [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){
    global_head_projections_impl<false>(q,k,v,wq,wr,ws,wk,wv,oq,orr,os,ok,ov,wk_taps,wv_taps,taps,gid,sg,lane);
}
kernel void paired_global_head_projections(device const bfloat *q,device const bfloat *k,device const bfloat *v,
      device const bfloat *wq,device const bfloat *wr,device const bfloat *ws,
      device const bfloat *wk,device const bfloat *wv,device bfloat *oq,device bfloat *orr,
      device bfloat *os,device bfloat *ok,device bfloat *ov,
      device const bfloat *wk_taps,device const bfloat *wv_taps,device bfloat *taps,
      uint gid [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){
    global_head_projections_impl<true>(q,k,v,wq,wr,ws,wk,wv,oq,orr,os,ok,ov,wk_taps,wv_taps,taps,gid,sg,lane);
}

// Parallelize narrow projections over the reduction dimension. Re-evaluate
// candidate BF16 midpoint crossings with the baseline's ordered FP32 reduction.
kernel void split_bf(device const bfloat *x,device const bfloat *w,device bfloat *y,constant uint3 &dims,
                     uint row [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){
    uint width=dims.z;w+=row*width;threadgroup float partial[8],magnitudes[8],result;
    threadgroup uint fix;
    float a=0,mag=0;
    for(uint j=tid*4;j<width;j+=256*4){
        float4 v=float4(*(device const bfloat4*)(x+j)),u=float4(*(device const bfloat4*)(w+j));
        for(ushort z=0;z<4;++z){float term=u[z]*v[z];a+=term;mag+=abs(term);}
    }
    a=simd_sum(a);mag=simd_sum(mag);
    if(!lane){partial[sg]=a;magnitudes[sg]=mag;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(!sg){
        float sum=simd_sum(lane<8?partial[lane]:0.f),bound=simd_sum(lane<8?magnitudes[lane]:0.f);
        if(!lane){
            uint bits=as_type<uint>(sum);
            float mid=as_type<float>((bits&0xffff0000u)|0x8000);
            fix=abs(mid-sum)<=bound*(width/32+24)*0x1p-24f;
            result=sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(!sg && fix){
        float v=0;for(uint j=lane;j<width;j+=32)v=fma(float(x[j]),float(w[j]),v);
        v=reference_sum(v);if(!lane)result=v;
    }
    if(!tid)y[row]=bfloat(result);
}
// A single activation row uses FP32 GEMV for the router and message gates.
inline float reduce_router_streams(float lo,float hi,uint lane){
    ushort i=lane%16;
    float v=simd_shuffle(lo,i)+simd_shuffle(lo,i+16);
    v+=simd_shuffle(hi,i);v+=simd_shuffle(hi,i+16);
    ushort four=i&~3;
    float group=simd_shuffle(v,four)+simd_shuffle(v,four+1);
    group+=simd_shuffle(v,four+2);group+=simd_shuffle(v,four+3);
    float total=simd_shuffle(group,0)+simd_shuffle(group,4);
    total+=simd_shuffle(group,8);return total+simd_shuffle(group,12);
}
inline float fp32_gemv_row(device const bfloat *x,device const bfloat *w,uint width,uint lane){
    // The CPU FP32 projection accumulates 64 independent FMA streams.
    float lo=0,hi=0;
    for(uint d=lane;d<width;d+=64){
        lo=fma(float(x[d]),float(w[d]),lo);
        hi=fma(float(x[d+32]),float(w[d+32]),hi);
    }
    return reduce_router_streams(lo,hi,lane);
}
// Store eight consecutive CPU FMA inputs per SIMD lane in one vector load.
template<class Input>
inline float packed_router_dot(Input x,device const bfloat *w,uint lane){
    float lo=0,hi=0;
    for(uint d=0;d<2304;d+=256){
        uint4 bits=((device const uint4*)(w+d))[lane];
        for(ushort j=0;j<4;++j){
            float2 weight=float2(as_type<bfloat2>(bits[j]));
            lo=fma(float(x[d+j*64+lane]),weight.x,lo);
            hi=fma(float(x[d+j*64+32+lane]),weight.y,hi);
        }
    }
    return reduce_router_streams(lo,hi,lane);
}
kernel void router_gemv(device const bfloat *x,device const bfloat *w,device float *out,
                         constant uint3 &dims,uint row [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]){
    float v=fp32_gemv_row(x,w+row*dims.z,dims.z,lane);
    if(!lane)out[row]=v;
}
// Sliding decode groups the projection, recurrent taps, and normalization of a
// whole head in one workgroup, following the MLX serving kernel's partition.
inline float blend_rms_inverse(device const float *parts,uint lane);
template<bool HighPairs,bool Blend=false,bool Prepared=false>
inline void sliding_heads_impl(
    device const bfloat *q, device const bfloat *k, device const bfloat *v,
    device const bfloat *wq, device const bfloat *wr, device const bfloat *ws,
    device const bfloat *wg, device const bfloat *wk, device const bfloat *wv,
    device const bfloat *qs, device const bfloat *rs,
    device const bfloat *kw, device const bfloat *vw,
    device const bfloat *old, device bfloat *next, device uchar *keys, device uchar *vals,
    device const uchar *table, device const uint *state,
    device bfloat *queries, device float *rels, device bfloat *gates, device bfloat *sinks,
    uint bank, uint sg,
    uint lane,threadgroup float *scratch,device const float *state_rows=nullptr,
    device const bfloat *norm_weight=nullptr,device const float *parts=nullptr,
    device bfloat *normalized_query=nullptr,threadgroup bfloat *normalized=nullptr) {
    #pragma clang fp contract(off)
    if(bank>=27)bank+=24;
    bool query=bank<24, rel=bank>=24 && bank<27, gate=bank>=27 && bank<51;
    bool key=bank>=51 && bank<59, val=bank>=59 && bank<67, sink=bank==67;
    uint head=query?bank:rel?bank-24:gate?bank-27:key?bank-51:val?bank-59:0;
    device const bfloat *x=key?k:val?v:q;
    device const bfloat *w=query?wq:rel?wr:gate?wg:key?wk:val?wv:ws;

    if constexpr(Blend){
        uint source=key?1:val?2:0;
        float inv=blend_rms_inverse(parts+source*160,lane);
        for(uint d=sg*32+lane;d<2304;d+=1024){
            bfloat value=bfloat(state_rows[source*2304+d]*inv*float(norm_weight[d]));
            uint at=packed_channel(d);
            normalized[at]=value;
            // The value projection later reads the normalized query for its attention gate.
            if(bank==0)normalized_query[at]=value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(sink && sg>=6)return;
    uint rows=(query || gate)?3072:rel?384:sink?24:1024;
    float dot;
    if constexpr(Blend || Prepared)dot=high_pair_channels<true,threadgroup const bfloat4*>(normalized,w,rows,head*128+sg*4,lane);
    else if constexpr(HighPairs)dot=high_pair_channels(x,w,rows,head*128+sg*4,lane);
    else dot=packed_projection_channel(x,w,rows,head*128+sg*4,lane);
    if(!(lane&7)){
        uint channel=sg*4+lane/8, d=head*128+channel;
        float raw=float(bfloat(dot));
        if(sink)sinks[d]=bfloat(raw);
        else if(gate)gates[d]=bfloat(raw);
        else if(query || rel)scratch[channel]=raw*float((query?qs:rs)[d]);
        else {
            uint idx=(key?0:3072)+d;
            device const bfloat *conv=(key?kw:vw)+d*4;
            float y=0;
            for(uint tap=0;tap<3;++tap) y=fma(float(old[idx+tap*1024]),float(conv[tap]),y);
            y=fma(raw,float(conv[3]),y);
            next[idx]=old[idx+1024];
            next[idx+1024]=old[idx+2048];
            next[idx+2048]=bfloat(raw);
            scratch[channel]=float(bfloat((y+raw)/sqrt(5.f)));
        }
    }
    if(sink || gate)return;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(rel?sg>=8:sg!=0)return;
    uint width=rel?16:128, offset=rel?sg*16:0;
    float square=0;
    if(val) {
        for(uint d=lane;d<128;d+=32)square+=scratch[d]*scratch[d];
        float b=simd_shuffle(square,lane%8+8),c=simd_shuffle(square,lane%8+16),e=simd_shuffle(square,lane%8+24);
        square=((square+b)+c)+e;
    } else if(lane<8) {
        for(uint d=lane;d<width;d+=8)square+=scratch[offset+d]*scratch[offset+d];
    }
    float total=0;
    for(ushort j=0;j<8;++j)total+=simd_shuffle(square,j);
    float scale=val?1.f/precise::sqrt(total/128.f+1e-6f):
                    min((rel?8.f:sqrt(512.f))*(1.f/(precise::sqrt(total)+1e-6f)),1.f);
    for(uint j=lane;j<width;j+=32) {
        uint d=head*128+offset+j;
        bfloat value=bfloat(scratch[offset+j]*scale);
        if(query)queries[d]=value;
        else if(rel)rels[d]=float(value);
        else {
            uint at=state[1]*1024+d;
            device uchar *cache=key?keys:vals;
            if(state[5])cache[at]=table[as_type<ushort>(value)];
            else ((device bfloat*)cache)[at]=value;
        }
    }
}
kernel void sliding_heads(
    device const bfloat *q, device const bfloat *k, device const bfloat *v,
    device const bfloat *wq, device const bfloat *wr, device const bfloat *ws,
    device const bfloat *wg, device const bfloat *wk, device const bfloat *wv,
    device const bfloat *qs, device const bfloat *rs,
    device const bfloat *kw, device const bfloat *vw,
    device const bfloat *old, device bfloat *next, device uchar *keys, device uchar *vals,
    device const uchar *table, device const uint *state,
    device bfloat *queries, device float *rels, device bfloat *gates, device bfloat *sinks,
    uint bank [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float scratch[128];
    sliding_heads_impl<false>(q,k,v,wq,wr,ws,wg,wk,wv,qs,rs,kw,vw,old,next,keys,vals,table,state,queries,rels,gates,sinks,bank,sg,lane,scratch);
}
kernel void paired_sliding_heads(
    device const bfloat *q, device const bfloat *k, device const bfloat *v,
    device const bfloat *wq, device const bfloat *wr, device const bfloat *ws,
    device const bfloat *wg, device const bfloat *wk, device const bfloat *wv,
    device const bfloat *qs, device const bfloat *rs,
    device const bfloat *kw, device const bfloat *vw,
    device const bfloat *old, device bfloat *next, device uchar *keys, device uchar *vals,
    device const uchar *table, device const uint *state,
    device bfloat *queries, device float *rels, device bfloat *gates, device bfloat *sinks,
    uint bank [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float scratch[128];
    sliding_heads_impl<true>(q,k,v,wq,wr,ws,wg,wk,wv,qs,rs,kw,vw,old,next,keys,vals,table,state,queries,rels,gates,sinks,bank,sg,lane,scratch);
}
kernel void blended_sliding_heads(
    device const float *raw, device const bfloat *norm, device const float *parts,
    device const bfloat *wq, device const bfloat *wr, device const bfloat *ws,
    device bfloat *normalized_query, device const bfloat *wk, device const bfloat *wv,
    device const bfloat *qs, device const bfloat *rs,
    device const bfloat *kw, device const bfloat *vw,
    device const bfloat *old, device bfloat *next, device uchar *keys, device uchar *vals,
    device const uchar *table, device const uint *state,
    device bfloat *queries, device float *rels, device bfloat *gates, device bfloat *sinks,
    uint bank [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float scratch[128];
    threadgroup bfloat normalized[2304];
    sliding_heads_impl<true,true>(nullptr,nullptr,nullptr,wq,wr,ws,nullptr,wk,wv,qs,rs,kw,vw,old,next,keys,vals,table,state,queries,rels,gates,sinks,bank,sg,lane,scratch,raw,norm,parts,normalized_query,normalized);
}

kernel void expert_router(device const bfloat *x,device const bfloat *weight,device float *logits,
                          device bfloat4 *packed,uint row [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]){
    if(row<18){
        bfloat4 a;for(ushort j=0;j<4;++j)a[j]=x[row*128+j*32+lane];
        packed[row*32+lane]=a;
    }
    float y=packed_router_dot(x,weight+row*2304,lane);
    if(!lane)logits[row]=y;
}

kernel void paired_output_residual(device const bfloat *x,device const bfloat *weight,device const bfloat *hidden,
                                  device const bfloat *conv,device const bfloat *old,device bfloat *next,
                                  device float *residual,constant uint &skip,uint group [[threadgroup_position_in_grid]],
                                  uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){
    #pragma clang fp contract(off)
    uint d=group*8+sg;
    device const uint *table=(device const uint*)((device const uchar*)weight+2304*3072*3/2)+(d/32)*256;
    float sum=0;
    for(uint k=0;k<24;++k){
        device const uchar *panel=(device const uchar*)weight+(d*24+k)*192;
        uint4 low=uint4(((device const uchar4*)panel)[lane]);
        uchar2 codes=((device const uchar2*)(panel+128))[lane];
        uint2 packed=uint2(table[codes.x],table[codes.y])|uint2(low.x|(low.y<<16),low.z|(low.w<<16));
        float4 w=float4(as_type<bfloat4>(packed));
        float4 act=float4(((device const bfloat4*)x)[k*32+lane]);
        for(ushort j=0;j<4;++j)sum=fma(act[j],w[j],sum);
    }
    float projected=reference_sum(sum);
    if(!lane)store_output_residual(projected,d,hidden,conv,old,next,residual,skip);
}
