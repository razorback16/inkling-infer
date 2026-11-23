// Keep the same K panel of adjacent output rows contiguous. Decode reads
// each row tile as a unit and prompt matrix products reuse the same layout.
inline uint expert_panel(uint row,uint cols,uint k){
    constexpr uint rows=4;
    return ((row/rows)*cols+k)*rows+row%rows;
}
// NVFP4 magnitudes are 0, .5, 1, 1.5, 2, 3, 4 and 6. Power-of-two
// multiples of the rounded unit and 1.5-unit bases retain BF16 rounding.
inline bfloat4 unpack_four_codes(ushort4 code, ushort2 base) {
    ushort4 mag = code & 7;
    ushort4 significand = select(ushort4(base.x), ushort4(base.y), (mag > 1) & ((mag & 1) != 0));
    ushort4 bits = significand + ((mag / 2 - 1) << 7);
    bits = select(ushort4(0), bits, (significand != 0) & (mag != 0));
    return as_type<bfloat4>(bits | ((code & 8) << 12));
}
// Group routed rows into 128-token tiles without reading the routing decisions
// back to the host. The expert index is data-dependent on every invocation.
kernel void expert_counts(device const uint *ids,device uint *counts,constant uint &n,
                          uint expert [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    threadgroup uint partial[8]; uint count=0;
    for(uint i=tid;i<n;i+=256) count+=ids[i]==expert;
    count=simd_sum(count);
    if (!lane) partial[sg]=count;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!sg) { count=simd_sum(lane<8?partial[lane]:0u); if (!lane) counts[expert]=count; }
}
kernel void expert_schedule(device const uint *counts,device uint *cursors,device uint4 *tiles,
                            constant uint &capacity,uint expert [[thread_position_in_grid]]) {
    uint start=0, tile=0;
    for(uint i=0;i<expert;++i) { start+=counts[i]; tile+=(counts[i]+127)/128; }
    cursors[expert]=start;
    for(uint j=0;j<counts[expert];j+=128)
        tiles[tile++]=uint4(start+j,expert,min(128u,counts[expert]-j),0);
    if(expert==127) while(tile<capacity) tiles[tile++]=0;
}
kernel void expert_order(device const uint *ids,device atomic_uint *cursors,
                         device uint *order,constant uint &n,uint i [[thread_position_in_grid]]) {
    if(i<n){
        uint pos=atomic_fetch_add_explicit(cursors+ids[i],1u,memory_order_relaxed);
        order[pos]=i;order[n+i]=pos;
    }
}

// Quantized GEMM on the original weight bank. Each loaded panel is decoded
// once and reused across up to 128 routed rows. Adapted from prefill_qmm.py.
template<bool Gate, uint BM>
inline void grouped_project(device const bfloat *x,device const uint *w,device const uchar *scales,
        device const bfloat *bank_codes,device const uint4 *tiles,device const uint *order,device bfloat *out,
        device const float *activation,device const bfloat *codes,device ushort *events,uint capacity,uint2 group,uint tid,uint sg,uint lane,
        threadgroup bfloat *panel,threadgroup atomic_uint *used) {
    constexpr uint BN=32, BK=128, WM=(BM==128?4:BM%16==0?2:1), WN=4/WM, N=2304, K=Gate?2304:1152;
    uint tile=group.y*16+group.x%16;
    if(tile>=capacity) return;
    uint4 desc=tiles[tile]; uint row=desc.x,expert=desc.y,count=desc.z,column=group.x/16*BN;
    if(!count) return;
    if(!tid)atomic_store_explicit(used,0u,memory_order_relaxed);
    uint header=tile*(N/BN)+column/BN;
    uint event_base=capacity*(N/BN)+row*(N/2)+count*(column/2);
    mlx::steel::BlockMMA<bfloat,bfloat,BM,BN,BK,WM,WN,false,true,BK,BK> mma(sg,lane);
    uint fn=((lane/4)&2)*2+(lane%2)*2;
    uint inrow[BM/(8*WM)];
    for(uint i=0;i<BM/(8*WM);++i) {
        uint r=row+min(uint(mma.sm)+i*8*WM,count-1);
        inrow[i]=Gate ? order[r]/8 : r;
    }
    for(uint k=0;k<K;k+=BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint j=tid*4;j<BN*32;j+=128*4){
            uint wr=j/32,stream=j%32,row=expert*N+column+wr;
            uint block=expert_panel(row,K/128,k/128);
            if(nvfp4){
                ushort4 words=*(device const ushort4*)((device const ushort*)w+block*32+stream);
                uchar4 factors=*(device const uchar4*)(scales+block*8+(stream/16)*4);
                for(ushort part=0;part<4;++part){
                    uint base=(expert*128+factors[part])*16;
                    ushort2 bases=as_type<ushort2>(*(device const bfloat2*)(bank_codes+base+2));
                    ushort4 code=(words>>(part*4))&15;
                    *(threadgroup bfloat4*)(panel+wr*BK+part*32+stream)=unpack_four_codes(code,bases);
                }
            }else{
                uint4 words=*(device const uint4*)(w+block*32+stream);
                uchar4 factors=*(device const uchar4*)(scales+block*4);
                for(ushort part=0;part<4;++part){
                    uint4 code=(words>>(part*8))&255;
                    bfloat4 values;for(ushort i=0;i<4;++i)values[i]=codes[uint(factors[part])*256+code[i]];
                    *(threadgroup bfloat4*)(panel+wr*BK+part*32+stream)=values;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for(short kk=0;kk<BK;kk+=8) {
            for(short i=0;i<BM/(8*WM);++i)
                mma.Atile.frag_at(i,0)=float2(*(device const bfloat2*)(x+inrow[i]*K+k+kk+fn));
            mma.Btile.template load<bfloat,1,WN,1,BK>(panel+mma.Bs_offset+kk);
            mlx::steel::tile_matmad(mma.Ctile,mma.Atile,mma.Btile,mma.Ctile);
        }
    }
    for(short i=0;i<BM/(8*WM);++i) {
        uint r=mma.sm+i*8*WM;

        for(short j=0;j<BN/(8*WN);++j) {
            uint c=column+mma.sn+j*8*WN;
            float2 value=mma.Ctile.frag_at(i,j);
            uint middle=(as_type<uint>(value.x)&0xffff0000u)|0x8000u;
            bool low=value.x!=0 && abs(value.x-as_type<float>(middle))<=4.e-6f*max(abs(value.x),.5f);
            middle=(as_type<uint>(value.y)&0xffff0000u)|0x8000u;
            bool high=value.y!=0 && abs(value.y-as_type<float>(middle))<=4.e-6f*max(abs(value.y),.5f);
            uint flags=uint(low)|(uint(high)<<1);
            uint ballot=uint((simd_vote::vote_t)simd_ballot(r<count && flags));
            if constexpr(!Gate){
                ushort packed=ushort(flags);
                packed|=simd_shuffle_xor(packed,1)<<2;
                packed|=simd_shuffle_xor(packed,8)<<4;
                if(r<count && !(c%8)){
                    uint slot=order[row+r],addr=((slot/8*36+c/64)*8+slot%8)*64+c%64;
                    ((device uchar*)events)[addr/8]=uchar(packed);
                }
            }else if(ballot){
                uint offset=0;
                if(!lane)offset=atomic_fetch_add_explicit(used,uint(popcount(ballot)),memory_order_relaxed);
                offset=simd_shuffle(offset,0);
                if(r<count && flags)events[event_base+offset+popcount(ballot&((1u<<lane)-1))]=ushort((flags<<11)|(r*(BN/2)+(c-column)/2));
            }
            if(r>=count)continue;
            bfloat2 pair=bfloat2(value);
            if(Gate){
                ushort2 bits=as_type<ushort2>(pair);
                out[(row+r)*1152+c/2]=flags==1?pair.y:flags==2?pair.x:bfloat(activation[2*uint(bits.x)]*activation[2*uint(bits.y)+1]);
            }else{
                uint slot=order[row+r],addr=((slot/8*36+c/64)*8+slot%8)*64+c%64;
                *(device bfloat2*)(out+addr)=pair;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(Gate && !tid)events[header]=ushort(atomic_load_explicit(used,memory_order_relaxed));
}
#define GROUPED(NAME,GATE) \
kernel void NAME(device const bfloat *x,device const uint *w,device const uchar *scales,device const bfloat *bank_codes, \
    device const uint4 *tiles,device const uint *order,device bfloat *out,device const float *activation, \
    device const bfloat *codes,device ushort *events,constant uint &capacity,uint2 group [[threadgroup_position_in_grid]], \
    uint tid [[thread_index_in_threadgroup]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup atomic_uint used; threadgroup bfloat panel[32*128]; uint tile=group.y*16+group.x%16; if(tile>=capacity)return; \
    uint count=tiles[tile].z; if(!count)return; \
    if(count<=8)grouped_project<GATE,8>(x,w,scales,bank_codes,tiles,order,out,activation,codes,events,capacity,group,tid,sg,lane,panel,&used); \
    else if(count<=16)grouped_project<GATE,16>(x,w,scales,bank_codes,tiles,order,out,activation,codes,events,capacity,group,tid,sg,lane,panel,&used); \
    else if(count<=24)grouped_project<GATE,24>(x,w,scales,bank_codes,tiles,order,out,activation,codes,events,capacity,group,tid,sg,lane,panel,&used); \
    else if(count<=32)grouped_project<GATE,32>(x,w,scales,bank_codes,tiles,order,out,activation,codes,events,capacity,group,tid,sg,lane,panel,&used); \
    else if(count<=40)grouped_project<GATE,40>(x,w,scales,bank_codes,tiles,order,out,activation,codes,events,capacity,group,tid,sg,lane,panel,&used); \
    else if(count<=48)grouped_project<GATE,48>(x,w,scales,bank_codes,tiles,order,out,activation,codes,events,capacity,group,tid,sg,lane,panel,&used); \
    else if(count<=56)grouped_project<GATE,56>(x,w,scales,bank_codes,tiles,order,out,activation,codes,events,capacity,group,tid,sg,lane,panel,&used); \
    else if(count<=64)grouped_project<GATE,64>(x,w,scales,bank_codes,tiles,order,out,activation,codes,events,capacity,group,tid,sg,lane,panel,&used); \
    else if(count<=80)grouped_project<GATE,80>(x,w,scales,bank_codes,tiles,order,out,activation,codes,events,capacity,group,tid,sg,lane,panel,&used); \
    else if(count<=96)grouped_project<GATE,96>(x,w,scales,bank_codes,tiles,order,out,activation,codes,events,capacity,group,tid,sg,lane,panel,&used); \
    else if(count<=112)grouped_project<GATE,112>(x,w,scales,bank_codes,tiles,order,out,activation,codes,events,capacity,group,tid,sg,lane,panel,&used); \
    else grouped_project<GATE,128>(x,w,scales,bank_codes,tiles,order,out,activation,codes,events,capacity,group,tid,sg,lane,panel,&used); }
GROUPED(grouped_gate,true)
GROUPED(grouped_down,false)
kernel void expert_mix(device const bfloat *states,device const float *weights,device bfloat *out,
                       uint2 t [[thread_position_in_grid]]) {
    #pragma clang fp contract(off)
    uint c=t.x*4;
    float4 mixed=0;
    for(uint e=0;e<8;++e) {
        uint addr=((t.y*36+c/64)*8+e)*64+c%64;
        mixed+=float4(*(device const bfloat4*)(states+addr))*weights[t.y*8+e];
    }
    *(device bfloat4*)(out+t.y*2304+c)=bfloat4(mixed);
}


template<uint K,ushort Channels>
inline bfloat2 repair_expert_channels(device const bfloat *x,device const uchar *w,device const uchar *scales,
                                      device const bfloat *bank,device const bfloat *codes,uint expert,uint c,uint lane){
        float2 sum=0;
        for(uint k=0;k<K;k+=128){
            float4 act;for(ushort j=0;j<4;++j)act[j]=float(x[k+lane+j*32]);
            for(ushort channel=0;channel<Channels;++channel){
                uint block=expert_panel(expert*2304+c+channel,K/128,k/128);
                if(nvfp4){
                    ushort word=((device const ushort*)w)[block*32+lane];
                    uchar4 factors=*(device const uchar4*)(scales+block*8+(lane/16)*4);
                    for(ushort j=0;j<4;++j)
                        sum[channel]=fma(act[j],float(bank[(expert*128+factors[j])*16+((word>>(4*j))&15)]),sum[channel]);
                }else{
                    uchar4 raw=*(device const uchar4*)(w+(block*32+lane)*4);
                    uchar4 factors=*(device const uchar4*)(scales+block*4);
                    for(ushort j=0;j<4;++j)sum[channel]=fma(act[j],float(codes[uint(factors[j])*256+raw[j]]),sum[channel]);
                }
            }
        }
        return bfloat2(float2(reference_sum(sum.x),Channels==2?reference_sum(sum.y):0.f));
}
template<bool Gate>
inline void repair_grouped(device const ushort *events,device const bfloat *x,device const uchar *w,
                           device const uchar *scales,device const bfloat *bank,device const bfloat *codes,
                           device const uint *order,device const uint4 *tiles,device const float *activation,
                           device bfloat *out,uint cap,uint2 g,uint sg,uint lane){
    constexpr uint K=Gate?2304:1152;
    uint tile=g.y*16+g.x%16;
    if(tile>=cap)return;
    uint4 desc=tiles[tile];if(!desc.z)return;
    uint column=g.x/16*32,expert=desc.y,base=cap*72+desc.x*1152+desc.z*(column/2);
    uint n=events[tile*72+g.x/16];
    for(uint i=sg;i<n;i+=8){
        uint entry=events[base+i],r=desc.x+(entry&2047u)/16,c=column+(entry%16)*2,slot=order[r];
        uint xr=Gate?slot/8:r;
        uint flags=entry>>11,first=c+uint(flags==2);
        bfloat2 pair=flags==3?repair_expert_channels<K,2>(x+xr*K,w,scales,bank,codes,expert,first,lane):
                              repair_expert_channels<K,1>(x+xr*K,w,scales,bank,codes,expert,first,lane);
        if(!lane){
            if(Gate){
                if(flags==1)pair.y=out[r*1152+c/2];
                if(flags==2)pair=bfloat2(out[r*1152+c/2],pair.x);
                ushort2 bits=as_type<ushort2>(pair);
                out[r*1152+c/2]=bfloat(activation[2*uint(bits.x)]*activation[2*uint(bits.y)+1]);
            }else{
                uint addr=((slot/8*36+c/64)*8+slot%8)*64+c%64;
                if(flags==3)*(device bfloat2*)(out+addr)=pair;
                else out[addr+uint(flags==2)]=pair.x;
            }
        }
    }
}
#define REPAIR_GROUPED(NAME,GATE) \
kernel void NAME(device const ushort *events,device const bfloat *x,device const uchar *w,device const uchar *scales, \
                 device const bfloat *bank,device const bfloat *codes,device const uint *order,device const uint4 *tiles, \
                 device const float *activation,device bfloat *out,constant uint &cap,uint2 g [[threadgroup_position_in_grid]], \
                 uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){ \
    repair_grouped<GATE>(events,x,w,scales,bank,codes,order,tiles,activation,out,cap,g,sg,lane); }
REPAIR_GROUPED(repair_grouped_gate,true)
REPAIR_GROUPED(repair_grouped_down,false)

// Final rounding after the weighted expert sum can absorb a BF16 dot change.
// Replay all flagged terms whenever the two bounding mixtures round apart.
kernel void mix_down_replay(device const bfloat *states,device const float *routes,device const uchar *flags,
    device const bfloat *input,device const uchar *w,device const uchar *scales,device const bfloat *bank,
    device const bfloat *codes,device const uint *ids,device const uint *inverse,device bfloat *out,
    uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){
    #pragma clang fp contract(off)
    threadgroup ushort pending[256];
    threadgroup atomic_uint used;
    if(!tid)atomic_store_explicit(&used,0u,memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint c=g.x*256+tid;
    float sum=0,lower=0,upper=0;
    uint mask=0;
    for(ushort e=0;e<8;++e){
        uint addr=((g.y*36+c/64)*8+e)*64+c%64;
        bfloat raw=states[addr];float v=float(raw),lo=v,hi=v;
        if((flags[addr/8]>>(c%8))&1){
            mask|=1u<<e;
            ushort b=as_type<ushort>(raw);
            // Include both adjacent BF16 values and the full absolute band
            // used at small/cancelling dots.
            float adj1=float(as_type<bfloat>(ushort(max(b,ushort(1))+1)));
            float adj2=float(as_type<bfloat>(ushort(max(b,ushort(1))-1)));
            float bound=4.e-6f*max(.5f,abs(v));
            lo=min(v-bound,min(adj1,adj2));hi=max(v+bound,max(adj1,adj2));
            if((b&0x7fffu)==0){lo=-bound;hi=bound;}
        }
        float weight=routes[g.y*8+e];
        sum+=v*weight;lower+=lo*weight;upper+=hi*weight;
    }
    bool fix=mask && bfloat(lower)!=bfloat(upper);
    uint ballot=uint((simd_vote::vote_t)simd_ballot(fix));
    if(ballot){
        uint base=0;
        if(!lane)base=atomic_fetch_add_explicit(&used,uint(popcount(ballot)),memory_order_relaxed);
        base=simd_shuffle(base,0);
        if(fix)pending[base+popcount(ballot&((1u<<lane)-1))]=ushort((mask<<8)|tid);
    }
    if(!fix)out[g.y*2304+c]=bfloat(sum);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint n=atomic_load_explicit(&used,memory_order_relaxed);
    for(uint j=sg;j<n;j+=8){
        uint event=pending[j],col=g.x*256+(event&255u);
        float mixed=0;
        for(ushort e=0;e<8;++e){
            uint slot=g.y*8+e,addr=((g.y*36+col/64)*8+e)*64+col%64;
            bfloat value;
            if((event>>(8+e))&1){
                uint row=inverse[slot],expert=ids[slot];
                value=repair_expert_channels<1152,1>(input+row*1152,w,scales,bank,codes,expert,col,lane).x;
            }else value=states[addr];
            if(!lane)mixed+=float(value)*routes[slot];
        }
        if(!lane)out[g.y*2304+col]=bfloat(mixed);
    }
}
