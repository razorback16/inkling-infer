// PyTorch eager attention rounds QK, scaled QK, biased scores and normalized
// probabilities to BF16. Compute a compact score panel for each 32-query tile.
inline uint tile_start(uint row,device const uint *state) {
    uint end=state[1]+(row&~31u)+1, window=state[4];
    return window && end>window ? (end-window)&~31u : 0;
}
kernel void attention_scores(device const bfloat *q,device const uchar *keys,device const bfloat *bias,
                             device bfloat *scores,device const uint *s,uint3 g [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]],uint sg [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
    uint head=g.z*3+sg/4, row=g.y*32+(sg%4)*8+((lane/4)&4)+(lane/2)%4;
    uint col=((lane/4)&2)*2+(lane%2)*2, k0=tile_start(g.y*32,s)+g.x*32;
    if(k0>s[1]+min((g.y+1)*32,s[2])-1)return;
    threadgroup atomic_uint count;
    threadgroup ushort pending[3072];
    if(!tid)atomic_store_explicit(&count,0u,memory_order_relaxed);
    threadgroup bfloat panel[128][40];
    if(!s[5] && k0+32<=s[3]){
        for(ushort tile=sg;tile<64;tile+=12){
            ushort row=tile/16*8,d=tile%16*8;
            simdgroup_matrix<bfloat,8,8> data;
            simdgroup_load(data,((device const bfloat*)keys)+((k0+row)*8+g.z)*128+d,1024);
            simdgroup_store(data,&panel[d][row],40,ulong2(0),true);
        }
    }else{
        for(uint j=tid*4;j<32*128;j+=384*4){
            uint k=k0+j/128,d=j%128;
            bfloat4 vec=bfloat4(0);
            if(k<s[3]){
                size_t at=(size_t(k)*8+g.z)*128+d;
                if(s[5]){
                    uchar4 raw=*(device const uchar4*)(keys+at);
                    for(ushort e=0;e<4;++e)vec[e]=bfloat(e4m3(raw[e]));
                }else vec=*(device const bfloat4*)((device const bfloat*)keys+at);
            }
            for(ushort e=0;e<4;++e)panel[d+e][j/128]=vec[e];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 acc[4];
    for(uint i=0;i<4;++i)acc[i]=make_filled_simdgroup_matrix<float,8,8>(0.f);
    for(uint d=0;d<128;d+=8){
        simdgroup_matrix<bfloat,8,8> a;
        bfloat2 pair=row<s[2]?*(device const bfloat2*)(q+(row*24+head)*128+d+col):bfloat2(0);
        for(ushort e=0;e<2;++e)a.thread_elements()[e]=pair[e];
        for(uint i=0;i<4;++i){
            simdgroup_matrix<bfloat,8,8> b;
            simdgroup_load(b,&panel[d][i*8],40);
            simdgroup_multiply_accumulate(acc[i],a,b,acc[i]);
        }
    }
    for(uint i=0;i<4;++i)for(uint e=0;e<2;++e){
        uint column=g.x*32+i*8+col+e,k=k0+i*8+col+e;
        int dist=int(s[1]+row)-int(k);
        bool valid=row<s[2] && dist>=0 && (!s[4] || uint(dist)<s[4]);
        float dot=acc[i].thread_elements()[e];
        float mid=as_type<float>((as_type<uint>(dot)&0xffff0000u)|0x8000u);
        bool fix=valid && dot!=0 && abs(dot-mid)<3.0e-5f+2.e-6f*abs(dot);
        uint mask=uint((simd_vote::vote_t)simd_ballot(fix));
        if(mask){
            uint slot=0;
            if(!lane)slot=atomic_fetch_add_explicit(&count,uint(popcount(mask)),memory_order_relaxed);
            slot=simd_shuffle(slot,0);
            if(fix)pending[slot+popcount(mask&((1u<<lane)-1))]=ushort((head%3)*1024+(row%32)*32+(k-k0));
        }
        if(row<s[2] && !fix){
            float v=-INFINITY;
            if(valid){
                v=float(bfloat(float(bfloat(dot))*rsqrt(128.f)));
                if(dist<1024)v+=bias[(row*24+head)*1024+dist];
            }
            scores[(row*24+head)*s[10]+column]=bfloat(v);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint size=atomic_load_explicit(&count,memory_order_relaxed);
    for(uint j=sg;j<size;j+=12){
        uint entry=pending[j],h=g.z*3+entry/1024,r=g.y*32+(entry%1024)/32,k=k0+entry%32;
        float sum=0;
        for(uint d=lane;d<128;d+=32)sum=fma(float(q[(r*24+h)*128+d]),float(panel[d][entry%32]),sum);
        sum=reference_sum(sum);
        if(!lane){
            float v=float(bfloat(float(bfloat(sum))*rsqrt(128.f)));
            uint dist=s[1]+r-k;
            if(dist<1024)v+=float(bias[(r*24+h)*1024+dist]);
            scores[(r*24+h)*s[10]+g.x*32+entry%32]=bfloat(v);
        }
    }
}
// One SIMD matrix scores the three queries of a KV head against eight keys.
enum class ScoreKeyPanel { bf16, fp8, tail };
template<ScoreKeyPanel mode,ushort tiles>
inline void accumulate_key_scores(thread simdgroup_float8x8 (&acc)[tiles],device const bfloat *q,device const uchar *keys,
    device const uint *s,uint3 g,uint head,uint rr,uint cc,uint base,uint first){
    #pragma clang loop unroll_count(2)
    for(ushort d=0;d<128;d+=8){
        simdgroup_matrix<bfloat,8,8> a;
        // Each key tile consumes only the first three matrix rows.
        bfloat2 query=*(device const bfloat2*)(q+(head+rr%3)*128+d+cc);
        for(ushort j=0;j<2;++j)a.thread_elements()[j]=query[j];
        for(ushort tile=0;tile<tiles;++tile){
            uint key0=base+first+tile*8;
            simdgroup_matrix<bfloat,8,8> b;
            if constexpr(mode==ScoreKeyPanel::bf16){
                device const bfloat *block=((device const bfloat*)keys)+(key0*8+g.z)*128+d;
                simdgroup_load(b,block,1024,ulong2(0),true);
            }else{
                for(ushort j=0;j<2;++j){
                    uint key=key0+cc+j;
                    uint at=(key*8+g.z)*128+d+rr;
                    if constexpr(mode==ScoreKeyPanel::fp8)b.thread_elements()[j]=bfloat(e4m3(keys[at]));
                    else b.thread_elements()[j]=key<=s[1]?bfloat(cache_value(keys,at,s[5])):bfloat(0);
                }
            }
            simdgroup_multiply_accumulate(acc[tile],a,b,acc[tile]);
        }
    }
}
template<ushort tiles>
inline void score_key_tiles(device const bfloat *q,device const uchar *keys,device const float *relative,device const bfloat *weights,
                          device bfloat *scores,device const uint *s,uint3 g,
                          uint sg,uint lane){
    uint first=g.x*32*tiles+sg*8*tiles,base=tile_start(0,s),head=g.z*3;
    if(base+first>s[1])return;
    uint rr=((lane/4)&4)+(lane/2)%4,cc=((lane/4)&2)*2+(lane%2)*2;
    simdgroup_float8x8 acc[tiles];
    for(ushort tile=0;tile<tiles;++tile)acc[tile]=make_filled_simdgroup_matrix<float,8,8>(0.f);
    if(base+first+8*tiles<=s[1]+1){
        if(s[5])accumulate_key_scores<ScoreKeyPanel::fp8,tiles>(acc,q,keys,s,g,head,rr,cc,base,first);
        else accumulate_key_scores<ScoreKeyPanel::bf16,tiles>(acc,q,keys,s,g,head,rr,cc,base,first);
    }else accumulate_key_scores<ScoreKeyPanel::tail,tiles>(acc,q,keys,s,g,head,rr,cc,base,first);
    for(ushort tile=0;tile<tiles;++tile){
        uint start=first+tile*8,key0=base+start;
        if(key0>s[1])return;
        uint r=lane/8,c=lane%8,key=key0+c,dist=s[1]-key;
        uint src=(r&4)*4+(r&3)*2+(c&4)*2+(c&2)/2;
        float lo=simd_shuffle(acc[tile].thread_elements()[0],src),hi=simd_shuffle(acc[tile].thread_elements()[1],src);
        float dot=c%2?hi:lo;
        bool valid=r<3 && key<=s[1] && (!s[4] || dist<s[4]);
        float mid=as_type<float>((as_type<uint>(dot)&0xffff0000u)|0x8000u);
        uint mask=uint((simd_vote::vote_t)simd_ballot(valid && dot!=0 && abs(dot-mid)<3.e-5f+2.e-6f*abs(dot)));
        while(mask){
            uint owner=ctz(mask),k=key0+owner%8,h=head+owner/8;
            float exact=0;
            for(ushort d=lane;d<128;d+=32)exact=fma(float(q[h*128+d]),cache_value(keys,(k*8+g.z)*128+d,s[5]),exact);
            exact=simd_shuffle(reference_sum(exact),0);
            if(lane==owner)dot=exact;
            mask&=mask-1;
        }
        if(valid){
            float score=float(bfloat(float(bfloat(dot))*rsqrt(128.f)));
            if(dist<1024){
                float4 sums=0;
                for(ushort j=0;j<16;++j)sums[j%4]=fma(relative[(head+r)*16+j],float(weights[j*1024+dist]),sums[j%4]);
                score+=float(bfloat(((sums.x+sums.y)+sums.z)+sums.w));
            }
            scores[(head+r)*s[10]+start+c]=bfloat(score);
        }
    }
}

kernel void decode_scores(device const bfloat *q,device const uchar *keys,device const float *relative,device const bfloat *weights,
                          device bfloat *scores,device const uint *s,uint3 g [[threadgroup_position_in_grid]],
                          uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){
    score_key_tiles<1>(q,keys,relative,weights,scores,s,g,sg,lane);
}
kernel void decode_score_pairs(device const bfloat *q,device const uchar *keys,device const float *relative,device const bfloat *weights,
                          device bfloat *scores,device const uint *s,uint3 g [[threadgroup_position_in_grid]],
                          uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){
    score_key_tiles<2>(q,keys,relative,weights,scores,s,g,sg,lane);
}

// One SIMD group handles each sliding prompt row. Keep its exponents in
// lane registers and use the matrix unit to advance the four ordered sums.
template<bool Full>
inline void normalize_prompt_window(device bfloat *data,float sink,uint length,uint first,uint sink_lane,uint lane){
    float saved[32],peak=sink;
    uint count=Full?1024u:length;
    if constexpr(Full){
        for(ushort i=0;i<32;++i){
            saved[i]=float(data[i*32+lane]);
            peak=max(peak,saved[i]);
        }
    }else{
        for(uint k=lane;k<count;k+=32)peak=max(peak,float(data[k]));
    }
    peak=simd_max(peak);
    uint r=((lane/4)&4)+(lane/2)%4,c=((lane/4)&2)*2+(lane%2)*2;
    simdgroup_float8x8 acc=make_filled_simdgroup_matrix<float,8,8>(0.f);
    simdgroup_float8x8 ones=make_filled_simdgroup_matrix<float,8,8>(1.f);
    for(uint k=0;k<count;k+=32){
        float v;
        if constexpr(Full)v=reference_exp(saved[k/32]-peak);
        else v=k+lane<count?reference_exp(float(data[k+lane])-peak):0.f;
        if(Full || k<1024)saved[k/32]=v;
        simdgroup_float8x8 terms;
        for(ushort e=0;e<2;++e)terms.thread_elements()[e]=simd_shuffle(v,(c+e)*4+((r+4-first)&3));
        simdgroup_multiply_accumulate(acc,terms,ones,acc);
    }
    float sum=simd_shuffle(acc.thread_elements()[0],2*(lane&3));
    if((lane&3)==sink_lane)sum+=reference_exp(sink-peak);
    sum+=simd_shuffle_xor(sum,2);sum+=simd_shuffle_xor(sum,1);
    float inv=1.f/simd_broadcast_first(sum);
    for(uint k=0;k<count;k+=32){
        if(!Full && k+lane>=count)break;
        float v=Full || k<1024?saved[k/32]:reference_exp(float(data[k+lane])-peak);
        data[k+lane]=bfloat(v*inv);
    }
}
kernel void sliding_row_probabilities(device bfloat *scores,device const bfloat *sinks,device const uint *s,
                                      uint group [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]],
                                      uint lane [[thread_index_in_simdgroup]]){
    uint row=group*4+sg,token=row/24,base=tile_start(token,s);
    uint hi=s[1]+token+1-base,lo=hi>s[4]?hi-s[4]:0,length=hi-lo;
    uint origin=s[1]>=s[4]?s[1]-s[4]+1:0;
    uint first=(base+lo-origin)&3,sink_lane=(s[3]-origin)&3;
    device bfloat *data=scores+size_t(row)*s[10]+lo;
    if(length==1024)normalize_prompt_window<true>(data,float(sinks[row]),length,first,sink_lane,lane);
    else normalize_prompt_window<false>(data,float(sinks[row]),length,first,sink_lane,lane);
}
kernel void attention_probabilities(device bfloat *scores,device const bfloat *sinks,device const uint *s,
                                    uint row [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]],uint sg [[simdgroup_index_in_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]]){
    uint width=s[10],token=row/24,base=tile_start(token,s);
    uint hi=s[1]+token+1-base,lo=s[4] && hi>s[4]?hi-s[4]:0;
    scores+=row*width;
    threadgroup float part[8];
    float sink=float(bfloat(sinks[row])),peak=sink;
    for(uint i=lo+tid;i<hi;i+=256)peak=max(peak,float(scores[i]));
    peak=simd_max(peak);if(!lane)part[sg]=peak;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    peak=simd_max(lane<8?part[lane]:-INFINITY);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // arm64 eager softmax sums four interleaved FP32 streams, including
    // the sink at the end of the complete key tensor (after causal padding).
    uint origin=s[4] && s[1]>=s[4]?s[1]-s[4]+1:0;
    uint first=(base+lo-origin)&3, sink_lane=(s[3]-origin)&3;
    threadgroup float tile[256];
    float sum=0;
    for(uint j=lo;j<hi;j+=256){
        uint i=j+tid;
        tile[tid]=i<hi?reference_exp(float(scores[i])-peak):0.f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(tid<4)for(uint z=(tid+4-first)&3;z<min(256u,hi-j);z+=4)sum+=tile[z];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(!sg){
        if(lane==sink_lane)sum+=reference_exp(sink-peak);
        sum+=simd_shuffle_xor(sum,2);sum+=simd_shuffle_xor(sum,1);
        if(!lane)part[0]=1.f/sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv=part[0];
    for(uint i=lo+tid;i<hi;i+=256)scores[i]=bfloat(reference_exp(float(scores[i])-peak)*inv);
}
inline void accumulate_softmax_panels(thread simdgroup_float8x8 &acc,simdgroup_float8x8 ones,
    threadgroup const float *stored,uint length,uint first,uint r,uint c){
    uint stream=(r+4-first)&3,full=length&~31u,k=0;
    for(;k+8*32<=full;k+=8*32){
        #pragma clang loop unroll(full)
        for(ushort i=0;i<8;++i){
            simdgroup_float8x8 terms;
            terms.thread_elements()[0]=stored[k+i*32+c*4+stream];
            terms.thread_elements()[1]=stored[k+i*32+(c+1)*4+stream];
            simdgroup_multiply_accumulate(acc,terms,ones,acc);
        }
    }
    for(;k<full;k+=32){
        simdgroup_float8x8 terms;
        terms.thread_elements()[0]=stored[k+c*4+stream];
        terms.thread_elements()[1]=stored[k+(c+1)*4+stream];
        simdgroup_multiply_accumulate(acc,terms,ones,acc);
    }
    if(k<length){
        simdgroup_float8x8 terms;
        for(ushort j=0;j<2;++j){
            uint at=k+(c+j)*4+stream;
            terms.thread_elements()[j]=at<length?stored[at]:0.f;
        }
        simdgroup_multiply_accumulate(acc,terms,ones,acc);
    }
}
kernel void decode_probabilities(device bfloat *scores,device const bfloat *sinks,device const uint *s,
                                    uint row [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]],uint sg [[simdgroup_index_in_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]]){
    threadgroup float stored[6144];
    uint width=s[10],token=row/24,base=tile_start(token,s);
    uint hi=s[1]+token+1-base,lo=s[4] && hi>s[4]?hi-s[4]:0;
    scores+=row*width;
    threadgroup float part[8];
    float sink=float(bfloat(sinks[row])),peak=sink;
    for(uint i=lo+tid;i<hi;i+=256)peak=max(peak,float(scores[i]));
    peak=simd_max(peak);if(!lane)part[sg]=peak;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    peak=simd_max(lane<8?part[lane]:-INFINITY);
    // arm64 eager softmax sums four interleaved FP32 streams, including
    // the sink at the end of the complete key tensor (after causal padding).
    uint origin=s[4] && s[1]>=s[4]?s[1]-s[4]+1:0;
    uint first=(base+lo-origin)&3, sink_lane=(s[3]-origin)&3;
    if(hi-lo<=6144){
        for(uint j=tid;j<hi-lo;j+=256)
            stored[j]=reference_exp(float(scores[lo+j])-peak);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(!sg){
            uint r=((lane/4)&4)+(lane/2)%4,c=((lane/4)&2)*2+(lane%2)*2;
            simdgroup_float8x8 acc=make_filled_simdgroup_matrix<float,8,8>(0.f);
            simdgroup_float8x8 ones=make_filled_simdgroup_matrix<float,8,8>(1.f);
            accumulate_softmax_panels(acc,ones,stored,hi-lo,first,r,c);
            float sum=simd_shuffle(acc.thread_elements()[0],(lane&3)*2);
            if(lane==sink_lane)sum+=reference_exp(sink-peak);
            sum+=simd_shuffle_xor(sum,2);sum+=simd_shuffle_xor(sum,1);
            if(!lane)part[0]=1.f/sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float inv=part[0];
        for(uint j=tid;j<hi-lo;j+=256)scores[lo+j]=bfloat(stored[j]*inv);
        return;
    }
    // Carry the four ordered sums across pages for arbitrarily long global KV.
    uint r=((lane/4)&4)+(lane/2)%4,c=((lane/4)&2)*2+(lane%2)*2;
    simdgroup_float8x8 acc=make_filled_simdgroup_matrix<float,8,8>(0.f);
    simdgroup_float8x8 ones=make_filled_simdgroup_matrix<float,8,8>(1.f);
    for(uint page=lo;page<hi;page+=2048){
        uint length=min(2048u,hi-page);
        for(uint j=tid;j<length;j+=256)stored[j]=reference_exp(float(scores[page+j])-peak);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(!sg){
            accumulate_softmax_panels(acc,ones,stored,length,first,r,c);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(!sg){
        float sum=simd_shuffle(acc.thread_elements()[0],(lane&3)*2);
        if(lane==sink_lane)sum+=reference_exp(sink-peak);
        sum+=simd_shuffle_xor(sum,2);sum+=simd_shuffle_xor(sum,1);
        if(!lane)part[0]=1.f/sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv=part[0];
    for(uint i=lo+tid;i<hi;i+=256)scores[i]=bfloat(reference_exp(float(scores[i])-peak)*inv);
}
// One SIMD group per residue stream and eight output channels. The three
// query heads in a KV group share the same value tile. No partial KV panels
// are materialized between dispatches.
template<bool FP8>
inline void accumulate_value_panel(thread simdgroup_float8x8 &acc,device const bfloat *prob,
    device const uchar *values,uint pitch,uint ghead,uint head,uint channel,uint origin,uint base,
    uint k,uint sg,uint r,uint c){
    simdgroup_float8x8 a,b;
    uint pos=origin+k+sg;
    device const bfloat *p=prob+(head+r%3)*pitch+(pos-base)+c*4;
    a.thread_elements()[0]=p[0];a.thread_elements()[1]=p[4];
    uint at=((pos+r*4)*8+ghead)*128+channel+c;
    if constexpr(FP8){
        uchar2 codes=*(device const uchar2*)(values+at);
        b.thread_elements()[0]=e4m3(codes.x);b.thread_elements()[1]=e4m3(codes.y);
    }else{
        bfloat2 v=*(device const bfloat2*)(((device const bfloat*)values)+at);
        b.thread_elements()[0]=v.x;b.thread_elements()[1]=v.y;
    }
    simdgroup_multiply_accumulate(acc,a,b,acc);
}
template<bool FP8,ushort Panels>
inline void accumulate_full_value_panels(thread simdgroup_float8x8 &acc,device const bfloat *prob,
    device const uchar *values,uint pitch,uint gh,uint head,uint channel,uint origin,uint base,uint full,uint sg,uint r,uint c){
    uint k=0;
    for(;k+Panels*32<=full;k+=Panels*32){
        #pragma clang loop unroll(full)
        for(ushort step=0;step<Panels;++step)
            accumulate_value_panel<FP8>(acc,prob,values,pitch,gh,head,channel,origin,base,k+step*32,sg,r,c);
    }
    for(;k<full;k+=32)accumulate_value_panel<FP8>(acc,prob,values,pitch,gh,head,channel,origin,base,k,sg,r,c);
}
template<bool Project,bool Packed=true,bool HighPairs=false>
inline void attended_value_tile(device const bfloat *prob,device const uchar *values,device const bfloat *gate,device bfloat *out,
                               device const uint *s,device const bfloat *q,device const bfloat *gate_weights,device const float *sigmoid,uint2 g,uint sg,uint lane,
                               threadgroup float *part,threadgroup float *gate_tile){

    uint r=((lane/4)&4)+(lane/2)%4,c=((lane/4)&2)*2+(lane%2)*2;
    uint base=tile_start(0,s),origin=s[4] && s[1]>=s[4]?s[1]-s[4]+1:0;
    uint count=s[3]-origin,aligned=count&~3u,channel=g.x*8,head=g.y*3;
    if(sg<4){
    simdgroup_float8x8 acc=make_filled_simdgroup_matrix<float,8,8>(0.f);
    // The interior contains complete 32-key panels. The last partial
    // panel uses the same four ordered accumulations with masked loads.
    constexpr ushort panels=Packed?1:4;
    uint full=aligned&~31u;
    if(s[5])accumulate_full_value_panels<true,panels>(acc,prob,values,s[10],g.y,head,channel,origin,base,full,sg,r,c);
    else accumulate_full_value_panels<false,panels>(acc,prob,values,s[10],g.y,head,channel,origin,base,full,sg,r,c);
    if(full<aligned){
        uint k=full;
        simdgroup_float8x8 a,b;
        uint key=k+r*4+sg;
        for(uint e=0;e<2;++e){
            uint pos=k+(c+e)*4+sg;
            a.thread_elements()[e]=r<3 && pos<aligned?float(prob[(head+r)*s[10]+origin+pos-base]):0.f;
            b.thread_elements()[e]=key<aligned?cache_value(values,((origin+key)*8+g.y)*128+channel+c+e,s[5]):0.f;
        }
        simdgroup_multiply_accumulate(acc,a,b,acc);
    }
    if(r<3)for(uint e=0;e<2;++e)part[(sg*3+r)*8+c+e]=acc.thread_elements()[e];
    }else if constexpr(Project){
        uint local_head=(sg-4)/2,start=((sg-4)%2)*4;
        float value;
        uint first=(head+local_head)*128+channel+start;
        if constexpr(HighPairs)value=high_pair_channels<Packed>(q,gate_weights,3072,first,lane);
        else value=packed_projection_channel<Packed>(q,gate_weights,3072,first,lane);
        if(!(lane&7))gate_tile[local_head*8+start+lane/8]=sigmoid[as_type<ushort>(bfloat(value))];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(!sg && lane<24){
        uint r=lane/8,c=lane%8;
        uint d=channel+c;
        float sum=part[r*8+c];
        for(uint j=aligned;j<count;++j)
            sum=fma(float(prob[(head+r)*s[10]+origin+j-base]),cache_value(values,((origin+j)*8+g.y)*128+d,s[5]),sum);
        sum+=part[(1*3+r)*8+c];sum+=part[(2*3+r)*8+c];sum+=part[(3*3+r)*8+c];
        uint at=(head+r)*128+d;
        out[packed_channel(at)]=bfloat(float(bfloat(sum))*(Project?gate_tile[r*8+c]:1.f/(1.f+reference_exp(-float(gate[at])))));
    }
}
kernel void decode_av(device const bfloat *prob,device const uchar *values,device const bfloat *gate,device bfloat *out,
                       device const uint *s,uint2 g [[threadgroup_position_in_grid]],
                       uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){
    threadgroup float part[96];
    attended_value_tile<false>(prob,values,gate,out,s,nullptr,nullptr,nullptr,g,sg,lane,part,nullptr);
}
kernel void projected_decode_av(device const bfloat *prob,device const uchar *values,device const bfloat *q,
                                device const bfloat *gate_weights,device bfloat *out,device const uint *s,device const float *sigmoid,
                                uint2 g [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]){
    threadgroup float part[96];threadgroup float gate_tile[24];
    attended_value_tile<true>(prob,values,nullptr,out,s,q,gate_weights,sigmoid,g,sg,lane,part,gate_tile);
}
kernel void paired_projected_decode_av(device const bfloat *prob,device const uchar *values,device const bfloat *q,
                                device const bfloat *gate_weights,device bfloat *out,device const uint *s,device const float *sigmoid,
                                uint2 g [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]){
    threadgroup float part[96];threadgroup float gate_tile[24];
    attended_value_tile<true,true,true>(prob,values,nullptr,out,s,q,gate_weights,sigmoid,g,sg,lane,part,gate_tile);
}
kernel void global_projected_decode_av(device const bfloat *prob,device const uchar *values,device const bfloat *q,
                                device const bfloat *gate_weights,device bfloat *out,device const uint *s,device const float *sigmoid,
                                uint2 g [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]){
    threadgroup float part[96];threadgroup float gate_tile[24];
    attended_value_tile<true,false>(prob,values,nullptr,out,s,q,gate_weights,sigmoid,g,sg,lane,part,gate_tile);
}
kernel void paired_global_projected_decode_av(device const bfloat *prob,device const uchar *values,device const bfloat *q,
                                device const bfloat *gate_weights,device bfloat *out,device const uint *s,device const float *sigmoid,
                                uint2 g [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]){
    threadgroup float part[96];threadgroup float gate_tile[24];
    attended_value_tile<true,false,true>(prob,values,nullptr,out,s,q,gate_weights,sigmoid,g,sg,lane,part,gate_tile);
}


// Distribute the four CPU accumulation streams across independent SIMD groups.
// The 64-key staging tile gathers both inputs once and keeps each stream contiguous.
kernel void ordered_attention_values(device const bfloat *probs,device const uchar *values,device float *out,
                                     device const uint *s,uint3 g [[threadgroup_position_in_grid]],
                                     uint tid [[thread_index_in_threadgroup]],uint sg [[simdgroup_index_in_threadgroup]],
                                     uint lane [[thread_index_in_simdgroup]]){
    uint warp=sg/4,stream=sg%4,head=g.z,first=g.y*16;
    uint start=tile_start(first,s),width=s[10],origin=s[4]&&s[1]>=s[4]?s[1]-s[4]+1:0;
    uint aligned=s[3]-((s[3]-origin)&3u);
    uint end=min(aligned,s[1]+min(first+16,s[2]));
    threadgroup bfloat pa[16][72],vb[64][72];
    simdgroup_float8x8 acc[8];
    for(ushort i=0;i<8;++i)acc[i]=make_filled_simdgroup_matrix<float,8,8>(0.f);
    for(uint pos=start;pos<end;pos+=64){
        // Fully visible BF16 panels can be read directly into matrix registers.
        bool full=first+16<=s[2] && pos+64<=end;
        bool interior=full && pos+63<=s[1]+first && (!s[4] || s[1]+first+15-pos<s[4]);
        if(!s[5] && interior){
            uint r=((lane/4)&4)+(lane/2)%4,c=((lane/4)&2)*2+(lane%2)*2;
            uint row=first+warp*8+r;
            for(ushort k=0;k<16;k+=8){
                simdgroup_matrix<bfloat,8,8> a;
                for(ushort j=0;j<2;++j){
                    uint key=pos+stream+(k+c+j)*4;
                    a.thread_elements()[j]=probs[(row*24+head)*width+key-start];
                }
                for(ushort i=0;i<8;++i){
                    simdgroup_matrix<bfloat,8,8> b;
                    uint key=pos+stream+k*4;
                    simdgroup_load(b,((device const bfloat*)values)+(key*8+head/3)*128+g.x*64+i*8,4096);
                    simdgroup_multiply_accumulate(acc[i],a,b,acc[i]);
                }
            }
            continue;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint i=tid*4;i<64*64;i+=256*4){
            uint t=i/64,key=pos+t,d=g.x*64+i%64;
            bfloat4 b=bfloat4(0);
            if(key<end){
                size_t at=((size_t(key)*8+head/3)*128+d);
                if(s[5]){uchar4 raw=*(device const uchar4*)(values+at);for(ushort j=0;j<4;++j)b[j]=bfloat(e4m3(raw[j]));}
                else b=*(device const bfloat4*)((device const bfloat*)values+at);
            }
            *(threadgroup bfloat4*)&vb[(t%4)*16+t/4][i%64]=b;
        }
        for(uint i=tid*4;i<16*64;i+=256*4){
            uint y=i/64,row=first+y,t=i%64,diag=s[1]+row;
            for(ushort j=0;j<4;++j){
                uint key=pos+t+j;
                pa[y][j*16+t/4]=row<s[2] && key<end && key<=diag && (!s[4] || diag-key<s[4])
                    ?probs[(row*24+head)*width+key-start]:bfloat(0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(ushort k=0;k<16;k+=8){
            simdgroup_matrix<bfloat,8,8> a;
            simdgroup_load(a,&pa[warp*8][stream*16+k],72);
            for(ushort i=0;i<8;++i){
                simdgroup_matrix<bfloat,8,8> b;
                simdgroup_load(b,&vb[stream*16+k][i*8],72);
                simdgroup_multiply_accumulate(acc[i],a,b,acc[i]);
            }
        }
    }
    threadgroup float partial[4][16][64];
    for(ushort i=0;i<8;++i)simdgroup_store(acc[i],&partial[stream][warp*8][i*8],64);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=tid;i<1024;i+=256){
        uint row=first+i/64,d=g.x*64+i%64,diag=s[1]+row;
        if(row>=s[2])continue;
        float v=partial[origin%4][i/64][i%64];
        for(uint key=aligned;key<=diag;++key)if(key>=start && (!s[4] || diag-key<s[4]))
            v=fma(float(probs[(row*24+head)*width+key-start]),cache_value(values,((size_t(key)*8+head/3)*128+d),s[5]),v);
        for(ushort j=1;j<4;++j)v+=partial[(origin+j)%4][i/64][i%64];
        out[(row*24+head)*128+d]=v;
    }
}

// A standard 1024-key window fits two logits per thread. Preserve the four
// interleaved eager sums by zero-padding the final panel.
kernel void cached_window_probabilities(device bfloat *scores,device const bfloat *sinks,device const uint *s,
                        uint row [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){
    threadgroup float stored[1024],part[16];
    uint total=s[1]+1, count=min(total,1024u), lo=(total-count)&31u;
    scores+=row*s[10]+lo;
    float sink=float(sinks[row]),peak=sink;
    float vals[2];
    for(ushort i=0;i<2;++i){
        uint k=tid+i*512;
        vals[i]=k<count?float(scores[k]):-INFINITY;
        peak=max(peak,vals[i]);
    }
    peak=simd_max(peak);
    if(!lane)part[sg]=peak;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    peak=simd_max(lane<16?part[lane]:-INFINITY);
    for(ushort i=0;i<2;++i){
        uint k=tid+i*512;
        vals[i]=reference_exp(vals[i]-peak);
        stored[k]=vals[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(!sg){
        uint r=((lane/4)&4)+(lane/2)%4,c=((lane/4)&2)*2+(lane%2)*2;
        simdgroup_float8x8 acc=make_filled_simdgroup_matrix<float,8,8>(0.f);
        simdgroup_float8x8 ones=make_filled_simdgroup_matrix<float,8,8>(1.f);
        accumulate_softmax_panels(acc,ones,stored,1024,0,r,c);
        float sum=simd_shuffle(acc.thread_elements()[0],(lane&3)*2);
        if(lane==(count&3))sum+=reference_exp(sink-peak);
        sum+=simd_shuffle_xor(sum,2);sum+=simd_shuffle_xor(sum,1);
        if(!lane)part[0]=1.f/sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv=part[0];
    for(ushort i=0;i<2;++i){uint k=tid+i*512;if(k<count)scores[k]=bfloat(vals[i]*inv);}
}
