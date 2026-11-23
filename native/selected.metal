// Each SIMD group shares eight activation elements across four output rows.
// Tables contain the same BF16 weights as an eager checkpoint dequantization.
template<uint K,bool Packed=false>
inline float selected_channel_dot(device const bfloat *x,device const uchar *w,device const uchar *scales,
                           device const bfloat *bank,device const bfloat *codes,uint expert,uint c,uint lane) {
    if(nvfp4)bank+=expert*2048;
    float4 acc=0;
    if(nvfp4){
        uint first=(expert*2304+c)*(K/128);
        device const ushort *words=((device const ushort*)w)+first*32+lane;
        device const uchar4 *factors=((device const uchar4*)scales)+first*2+lane/16;

            ushort4 rawc; uint4 swc;
            for(ushort r=0;r<4;++r){
                rawc[r]=words[0*128+r*32];
                swc[r]=as_type<uint>(factors[0*8+r*2]);
            }
            float4 ac;
            if constexpr(Packed)ac=float4(*(device const bfloat4*)(x+0+lane*4));
            else for(ushort j=0;j<4;++j)ac[j]=float(x[0+lane+32*j]);
            for(uint d=128;d<K;d+=128){ words+=128;factors+=8;
            ushort4 rawn; uint4 swn;
            for(ushort r=0;r<4;++r){
                rawn[r]=words[0*128+r*32];
                swn[r]=as_type<uint>(factors[0*8+r*2]);
            }
            float4 an;
            if constexpr(Packed)an=float4(*(device const bfloat4*)(x+d+lane*4));
            else for(ushort j=0;j<4;++j)an[j]=float(x[d+lane+32*j]);

            float4 wtc[4];
            for(ushort r=0;r<4;++r){
                uint4 index=(uint4(as_type<uchar4>(swc[r]))<<4)|((uint4(rawc[r])>>uint4(0,4,8,12))&15);
                for(ushort j=0;j<4;++j)wtc[r][j]=float(bank[index[j]]);
            }

            for(ushort r=0;r<4;++r)for(ushort j=0;j<4;++j)acc[r]=fma(ac[j],wtc[r][j],acc[r]);
            ac=an;rawc=rawn;swc=swn;}
            float4 wtc[4];
            for(ushort r=0;r<4;++r){
                uint4 index=(uint4(as_type<uchar4>(swc[r]))<<4)|((uint4(rawc[r])>>uint4(0,4,8,12))&15);
                for(ushort j=0;j<4;++j)wtc[r][j]=float(bank[index[j]]);
            }

            for(ushort r=0;r<4;++r)for(ushort j=0;j<4;++j)acc[r]=fma(ac[j],wtc[r][j],acc[r]);
            }else for(uint d=0;d<K;d+=128){
        float4 act;
        if(Packed)act=float4(*(device const bfloat4*)(x+d+lane*4));
        else for(ushort j=0;j<4;++j)act[j]=float(x[d+lane+j*32]);
        for(ushort r=0;r<4;++r){
            uint row=expert*2304+c+r,block=expert_panel(row,K/128,d/128);
            float4 wt;
                uchar4 raw=*(device const uchar4*)(w+(block*32+lane)*4);
                uchar4 factors=*(device const uchar4*)(scales+block*4);
                for(ushort j=0;j<4;++j)wt[j]=float(codes[uint(factors[j])*256+raw[j]]);
            for(ushort j=0;j<4;++j)acc[r]=fma(act[j],wt[j],acc[r]);
        }
    }
    return reduce_channel_row(acc,lane);
}
template<bool Packed>
inline void select_gate_rows(device const bfloat *x,device const uint *ids,device const uchar *w,
                             device const uchar *scales,device const bfloat *bank,device const bfloat *codes,
                             device const float *activation,device bfloat *out,uint3 g,uint groups,uint sg,uint lane){
    uint slot=g.z*8+g.y,c=(g.x*groups+sg)*4;
    bfloat value=bfloat(selected_channel_dot<2304,Packed>(x+g.z*2304,w,scales,bank,codes,ids[slot],c,lane));
    ushort gate=as_type<ushort>(value),up=simd_shuffle_xor(gate,8);
    if(!(lane&15)){
        uint d=c/2+lane/16;
        out[slot*1152+packed_channel(d)]=bfloat(activation[2*uint(gate)]*activation[2*uint(up)+1]);
    }
}
kernel void selected_gate(device const bfloat *x,device const uint *ids,device const uchar *w,
                          device const uchar *scales,device const bfloat *bank,device const bfloat *codes,
                          device const float *activation,device bfloat *out,uint3 g [[threadgroup_position_in_grid]],
                          uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){
    select_gate_rows<false>(x,ids,w,scales,bank,codes,activation,out,g,4,sg,lane);
}
kernel void selected_gate_packed(device const bfloat *x,device const uint *ids,device const uchar *w,
                          device const uchar *scales,device const bfloat *bank,device const bfloat *codes,
                          device const float *activation,device bfloat *out,uint3 g [[threadgroup_position_in_grid]],
                          uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){
    // Interleave the eight MXFP8 experts before advancing to the next row tile.
    if(!nvfp4)g.xy=g.yx;
    select_gate_rows<true>(x,ids,w,scales,bank,codes,activation,out,g,nvfp4?4:8,sg,lane);
}
inline void select_down_partial(device const bfloat *x,device const uint *ids,device const uchar *w,
                                  device const uchar *scales,device const bfloat *bank,device const bfloat *codes,
                                  device const float *weights,threadgroup float4 *partial,uint2 g,uint sg,uint lane) {
    uint slot=g.y*8+sg,c=g.x*4;
    float p=float(bfloat(selected_channel_dot<1152,true>(x+slot*1152,w,scales,bank,codes,ids[slot],c,lane)));
    if(!(lane&7))((threadgroup float*)partial)[sg*4+lane/8]=p*weights[slot];
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
inline bfloat mixed_channel(threadgroup const float4 *partial,uint lane) {
    float v=0;for(uint j=0;j<8;++j)v+=partial[j][lane];
    return bfloat(v);
}
kernel void selected_down(device const bfloat *x,device const uint *ids,device const uchar *w,
                          device const uchar *scales,device const bfloat *bank,device const bfloat *codes,
                          device const float *weights,device bfloat *out,uint2 g [[threadgroup_position_in_grid]],
                          uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){
    threadgroup float4 partial[8];
    select_down_partial(x,ids,w,scales,bank,codes,weights,partial,g,sg,lane);
    if(!sg && lane<4)out[g.y*2304+g.x*4+lane]=mixed_channel(partial,lane);
}

kernel void finish_selected_down(device const bfloat *x,device const uint *ids,device const uchar *w,
                                 device const uchar *scales,device const bfloat *bank,device const bfloat *codes,
                                 device const float *weights,device bfloat *out,device const bfloat *history,
                                 device bfloat *next,device const bfloat *conv,device const float *residual,
                                 uint2 g [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float4 shared[16];
    threadgroup float4 *partial=shared;
    if(!nvfp4){
        // Each set of eight SIMD groups reduces one independent four-channel tile.
        g.x=g.x*2+sg/8;
        partial+=8*(sg/8);
        sg%=8;
    }
    select_down_partial(x,ids,w,scales,bank,codes,weights,partial,g,sg,lane);
    if(!sg && lane<4){
        uint d=g.x*4+lane;
        bfloat current=mixed_channel(partial,lane);
        float sum=0;
        for(uint j=0;j<3;++j){
            bfloat old=history[j*2304+d];
            sum=fma(float(old),float(conv[d*4+j]),sum);
            if(j)next[(j-1)*2304+d]=old;
        }
        next[2*2304+d]=current;
        sum=fma(float(current),float(conv[d*4+3]),sum);
        bfloat filtered=bfloat((sum+float(current))/sqrt(5.f));
        out[d]=bfloat(residual[d]+float(filtered));
    }
}

// Transpose four consecutive SIMD reduction steps into one packed lane load.
kernel void pack_decode_bank(device const uchar *src,device const uchar *scales,device uchar *dst,
                             device uchar *packed_scales,constant uint2 &dims,uint t [[thread_position_in_grid]]){
    uint samples=nvfp4?16:32,block=t/samples,l=t%samples;
    if(block>=dims.x)return;
    uint dest=expert_panel(block/dims.y,dims.y,block%dims.y);
    if(nvfp4){
        uint a=src[block*64+l],b=src[block*64+16+l],c=src[block*64+32+l],d=src[block*64+48+l];
        ushort2 pack=ushort2((a&15)|((b&15)<<4)|((c&15)<<8)|((d&15)<<12),
                             (a>>4)|(b&240)|((c&240)<<4)|((d&240)<<8));
        *(device ushort2*)(dst+dest*64+l*4)=pack;
        if(l<2){for(ushort j=0;j<4;++j)packed_scales[dest*8+l*4+j]=scales[block*8+j*2+l];}
    }else{
        if(l<4)packed_scales[dest*4+l]=scales[block*4+l];
        uchar4 pack;
        for(ushort j=0;j<4;++j)pack[j]=src[block*128+j*32+l];
        *(device uchar4*)(dst+dest*128+l*4)=pack;
    }
}
