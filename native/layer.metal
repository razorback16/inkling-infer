inline float reference_exp(float x);
// Per-token reductions use one threadgroup per row. Hidden vectors have 2304 channels.
template<class T, class U>
inline void normalize_row(device const T *x, device const bfloat *weight, device U *out,
                          uint4 dims, uint row, uint tid, uint sg, uint lane,
                          threadgroup float *sums) {
    #pragma clang fp contract(off)
    uint stride=dims.y, offset=dims.z;
    x+=row*2304; out+=row*stride+(dims.w==2?offset/32:offset);
    // Contiguous arm64 CPU reductions use eight four-lane accumulators
    // and cascade chunks of sixteen vectors into a second accumulator.
    if(sg<5){
        uint start=sg*512+lane;
        float subtotal=0;
        for(uint d=start;d<min(start+512,2304u);d+=32){
            float v=float(x[d]); subtotal+=v*v;
        }
        sums[sg*32+lane]=subtotal;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(!sg){
        float v=sums[lane]+sums[32+lane];
        v+=sums[64+lane];v+=sums[96+lane];v+=sums[128+lane];
        float a=simd_shuffle(v,(lane%8)+8),b=simd_shuffle(v,(lane%8)+16),c=simd_shuffle(v,(lane%8)+24);
        v=((v+a)+b)+c;
        float total=0;
        for(ushort i=0;i<8;++i)total+=simd_shuffle(v,i);
        if(!lane)sums[0]=1.f/precise::sqrt(total/2304.f+1e-6f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv=sums[0];
    if(dims.w==0){
        for(uint d=tid*4;d<2304;d+=1024){
            float4 input=float4(*(device const vec<T,4>*)(x+d));
            float4 scale=dims.x?float4(*(device const bfloat4*)(weight+d)):float4(1);
            *(device vec<U,4>*)(out+d)=vec<U,4>(input*inv*scale);
        }
        return;
    }

    // The controller stores each group of 32 channel streams as 72 adjacent samples.
    if(dims.w==2){
        for(uint i=tid*4;i<2304;i+=1024){
            uint stream=i/72,sample=i%72,d=stream+sample*32;
            float4 input,scale=1;
            for(ushort j=0;j<4;++j){
                input[j]=float(x[d+j*32]);
                if(dims.x)scale[j]=float(weight[d+j*32]);
            }
            *(device vec<U,4>*)(out+stream*(stride/32)+sample)=vec<U,4>(input*inv*scale);
        }
        return;
    }
    for(uint d=tid;d<2304;d+=256)
        out[packed_channel(d)]=U(float(x[d])*inv*(dims.x?float(weight[d]):1.f));
}
#define NORM(NAME,T,U) \
kernel void NAME(device const T *x,device const bfloat *weight,device U *out,constant uint4 &dims, \
                 uint row [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]], \
                 uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup float sums[160];normalize_row(x,weight,out,dims,row,tid,sg,lane,sums); }
NORM(norm_bf_float,bfloat,float)
NORM(norm_float_float,float,float)
NORM(norm_bf_bf,bfloat,bfloat)
NORM(norm_float_bf,float,bfloat)

kernel void depth_mix(device const float *s0,device const float *s1,device const float *s2,
                      device const float *s3,device const float *s4,device const float *query,
                      device float *output,constant uint3 &dims,uint2 group [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]],uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    #pragma clang fp contract(off)
    uint row=group.x;
    query+=group.y*2304;output+=group.y*dims.z*2304;
    device const float *states[5]={s0,s1,s2,s3,s4};
    threadgroup float scores[5][9],weights[5];
    uint count=dims.x,queries=dims.y;
    bool matrix=queries>1 && (count>=4 || queries==4);
    bool small=queries==3 && count<=3;
    if(small){
        uint j=tid/96,panel=(tid%96)/16,l=tid%16,ch=l%4;
        if(j<count){
            device const float *a=query+panel*384;
            device const float *b=states[j]+row*2304+panel*384;
            float v=0;
            if(l>=8)v=l<12?a[ch]*b[ch]:a[380+ch]*b[380+ch];
            for(uint k=4;k<372;k+=16)v=fma(a[k+l],b[k+l],v);
            v+=simd_shuffle_xor(v,8);
            uint tail=372+(l%8);
            v=fma(a[tail],b[tail],v);
            v+=simd_shuffle_xor(v,4);
            v+=simd_shuffle_xor(v,2);
            v+=simd_shuffle_xor(v,1);
            if(!l)scores[j][panel]=v;
        }
    }else if(matrix){
        uint width=queries==4 && count==2?2304:1024, panels=(2304+width-1)/width;
        if(tid<count*panels){
            uint j=tid/panels,panel=tid%panels;
            float acc=0;
            for(uint d=panel*width;d<min(2304u,(panel+1)*width);++d)
                acc=fma(query[d],states[j][row*2304+d],acc);
            scores[j][panel]=acc;
        }
    }else if(tid<20*count){
        uint j=tid/20,panel=tid%20/4,ch=tid%4;
        uint start=panel*512+ch,stop=min((panel+1)*512,2304u);
        float acc=0;
        for(uint d=start;d<stop-4;d+=4)acc=fma(query[d],states[j][row*2304+d],acc);
        uint tail=stop-4+ch;
        acc+=query[tail]*states[j][row*2304+tail];
        bool odd=(count%2 && j==count-1);
        float p=simd_shuffle_xor(acc,1),q=simd_shuffle_xor(acc,2);
        acc+=odd?q:p;
        p=simd_shuffle_xor(acc,1);q=simd_shuffle_xor(acc,2);acc+=odd?p:q;
        if(!ch)scores[j][panel]=acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(!tid){
        float peak=-INFINITY,z=0;
        for(uint j=0;j<count;++j){float s=0; for(uint i=0;i<(small?6u:matrix?(queries==4 && count==2?1u:3u):5u);++i)s+=scores[j][i];
            weights[j]=s*(1.f/48.f); peak=max(peak,weights[j]);}
        for(uint j=0;j<count;++j){weights[j]=reference_exp(weights[j]-peak);z+=weights[j];}
        if(count>=4)z=((weights[0]+(count==5?weights[4]:0.f))+weights[2])+(weights[1]+weights[3]);
        for(uint j=0;j<count;++j)weights[j]*=(1.f/z);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint d=tid;d<2304;d+=288){float v=0;
        for(uint j=0;j<count;++j) v=fma(weights[j],states[j][row*2304+d],v);
        output[row*2304+d]=v;}
}
kernel void message_blend(device const float *x,device const float *message,device const float *forget,
                          device const float *gain,device float *out,uint i [[thread_position_in_grid]]) {
    #pragma clang fp contract(off)
    out[i]=(1.f-reference_tanh(forget[i]))*x[i]+reference_tanh(gain[i])*message[i];
}

// old and next hold disjoint snapshots, so prefix branches can share the old state.
kernel void short_conv(device const bfloat *x,device const bfloat *old,device bfloat *next,
                       device const bfloat *weights,device const bfloat *delta,device bfloat *out,
                       constant uint3 &dims,uint2 t [[thread_position_in_grid]]) {
    #pragma clang fp contract(off)
    uint d=t.x,row=t.y,width=dims.x,len=dims.y;
    if(d>=width)return;
    float total=0;
    for(int i=0;i<4;++i){
        int r=dims.z?int(row)-i:int(row)-3+i;
        float v=r<0?float(old[(r+3)*width+d]):float(x[r*width+d]);
        float w=dims.z?float(weights[i*width+d])+float(delta[(row*8+d/128)*4+i])
                      :float(weights[d*4+i]);
        total=dims.z?total+v*w:fma(v,w,total);
    }
    out[row*width+d]=bfloat((total+float(x[row*width+d]))/(dims.z?3.f:sqrt(5.f)));
    if(row==0)for(int i=0;i<3;++i){
        int r=int(len)-3+i;
        next[i*width+d]=r<0?old[(r+3)*width+d]:x[r*width+d];
    }
}
// Apply the static depthwise filter and its residual using the same BF16
// rounding point as the separate convolution. Only retained rows need output.
template<typename Residual, typename Output>
inline void prompt_filter_residual(device const bfloat4 *x,device const bfloat4 *old,
        device bfloat4 *next,device const bfloat4 *weights,device const vec<Residual,4> *residual,
        device vec<Output,4> *out,uint3 dims,uint2 t){
    #pragma clang fp contract(off)
    constexpr uint stride=2304/4;
    uint column=t.x,row=t.y+dims.y;
    float4 sum=0;
    bfloat4 taps[4];
    for(ushort lane=0;lane<4;++lane)taps[lane]=weights[column*4+lane];
    for(ushort tap=0;tap<4;++tap){
        int source=int(row)-3+tap;
        bfloat4 v=source<0?old[(source+3)*stride+column]:x[uint(source)*stride+column];
        float4 w=float4(taps[0][tap],taps[1][tap],taps[2][tap],taps[3][tap]);
        sum=fma(float4(v),w,sum);
    }
    uint idx=t.y*stride+column;
    float4 filtered=float4(bfloat4((sum+float4(x[row*stride+column]))/sqrt(5.f)));
    float4 add=dims.z?float4(residual[idx]):float4(0);
    out[idx]=vec<Output,4>(filtered+add);
    if(t.y==0){
        for(int tap=0;tap<3;++tap){
            int source=int(dims.x)-3+tap;
            next[tap*stride+column]=source<0?old[(source+3)*stride+column]:x[uint(source)*stride+column];
        }
    }
}
#define PROMPT_FILTER(NAME,R,O) \
kernel void NAME(device const bfloat4 *x,device const bfloat4 *old,device bfloat4 *next, \
                 device const bfloat4 *weights,device const vec<R,4> *residual,device vec<O,4> *out, \
                 constant uint3 &dims,uint2 point [[thread_position_in_grid]]) { \
    prompt_filter_residual(x,old,next,weights,residual,out,dims,point); }
PROMPT_FILTER(prompt_attention_residual,bfloat,float)
PROMPT_FILTER(prompt_expert_residual,float,bfloat)

kernel void residual_float(device const bfloat *x,device const bfloat *filtered,device float *out,
                           constant uint &add,uint i [[thread_position_in_grid]]) {
    out[i]=float(filtered[i])+(add?float(x[i]):0.f);
}
kernel void residual_bf(device const float *x,device const bfloat *filtered,device bfloat *out,
                        uint i [[thread_position_in_grid]]){out[i]=bfloat(x[i]+float(filtered[i]));}

template<typename Out>
inline void clip_head(device const bfloat *input,device const bfloat *scale,device Out *out,
                      device const uint *state,uint width,bool scaled,uint2 g,uint lane){
    #pragma clang fp contract(off)
    uint row=g.y*24+g.x;
    float ss=0;
    if(lane<8)for(uint d=lane;d<width;d+=8){
        float v=float(input[row*width+d])*float(scale[g.x*width+d]);
        ss+=v*v;
    }
    float total=0;
    for(ushort j=0;j<8;++j)total+=simd_shuffle(ss,j);
    float factor=min((width==16?8.f:sqrt(512.f))*(1.f/(precise::sqrt(total)+1e-6f)),1.f);
    float tau=scaled?1.f+.1f*precise::log(max(float(state[0]+g.y+1)/8192.f,1.f)):1.f;
    for(uint d=lane;d<width;d+=32){
        float v=float(input[row*width+d])*float(scale[g.x*width+d]);
        out[row*width+d]=Out(bfloat(float(bfloat(v*factor))*tau));
    }
}
kernel void query_clip(device const bfloat *input,device const bfloat *scale,device bfloat *out,
                       device const uint *state,constant uint2 &dims,uint2 g [[threadgroup_position_in_grid]],
                       uint lane [[thread_index_in_simdgroup]]){
    clip_head(input,scale,out,state,dims.x,dims.y,g,lane);
}

kernel void clip_decode_heads(device const bfloat *q,device const bfloat *r,device const bfloat *qs,
                              device const bfloat *rs,device float *qo,device float *ro,device const uint *state,
                              constant uint &scaled,uint2 g [[threadgroup_position_in_grid]],
                              uint lane [[thread_index_in_simdgroup]]){
    clip_head(g.y?r:q,g.y?rs:qs,g.y?ro:qo,state,g.y?16:128,scaled,uint2(g.x,0),lane);
}
kernel void store_kv(device const bfloat *k,device const bfloat *v,device uchar *keys,device uchar *values,
                     device const uchar *table,device const uint *state,constant uint &fp8,
                     uint2 g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]){
    #pragma clang fp contract(off)
    uint base=(g.y*8+g.x)*128;
    float ks=0,vs=0;
    if(lane<8)for(uint d=lane;d<128;d+=8){float x=float(k[base+d]);ks+=x*x;}
    for(uint d=lane;d<128;d+=32){float x=float(v[base+d]);vs+=x*x;}
    float v8=simd_shuffle(vs,(lane%8)+8),v16=simd_shuffle(vs,(lane%8)+16),v24=simd_shuffle(vs,(lane%8)+24);
    vs=((vs+v8)+v16)+v24;
    float kv=0,vv=0;
    for(ushort j=0;j<8;++j){kv+=simd_shuffle(ks,j);vv+=simd_shuffle(vs,j);}
    if(state[4] && state[2]>1){
        // Depthwise convolution in eager prefill returns channel-major
        // storage, so each head reduces a strided vector.
        float square=0;
        if(!lane)for(uint d=0;d<128;++d){
            float x=float(k[base+d]);square=fma(x,x,square);
        }
        kv=simd_shuffle(square,0);
        float subtotal=0;
        if(lane<8)for(uint d=(lane/4)*64+lane%4;d<(lane/4+1)*64;d+=4){
            float x=float(v[base+d]);subtotal+=x*x;
        }
        subtotal+=simd_shuffle_xor(subtotal,4);
        vv=0;for(ushort j=0;j<4;++j)vv+=simd_shuffle(subtotal,j);
    }
    float kf=min(sqrt(512.f)*(1.f/(precise::sqrt(kv)+1e-6f)),1.f);
    float vf=1.f/precise::sqrt(vv/128.f+1e-6f);
    uint at=base+lane*4;
    bfloat4 ko=bfloat4(float4(*(device const bfloat4*)(k+at))*kf),vo=bfloat4(float4(*(device const bfloat4*)(v+at))*vf);
    at+=state[1]*1024;
    if(fp8){for(uint d=0;d<4;++d){keys[at+d]=table[as_type<ushort>(ko[d])];values[at+d]=table[as_type<ushort>(vo[d])];}}
    else {*(device bfloat4*)((device bfloat*)keys+at)=ko;*(device bfloat4*)((device bfloat*)values+at)=vo;}
}
// Store the filtered global KV heads and clip Q and relative position heads
// in parallel, consuming the tap projections from the same dense dispatch.
kernel void decode_global_heads(device const bfloat *kp,device const bfloat *vp,device const bfloat *taps,
    device const bfloat *ks,device const bfloat *vs,device const bfloat *old,device bfloat *next,
    device uchar *keys,device uchar *values,device const uchar *table,device const uint *state,
    device const bfloat *q,device const bfloat *rel,device const bfloat *qs,device const bfloat *rs,
    device bfloat *qo,device float *ro,uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){
    #pragma clang fp contract(off)
    if(head>=8){
        uint h=(head-8)*8+sg;
        bool relative=h>=24;
        if(relative)clip_head(rel,rs,ro,state,16,true,uint2(h-24,0),lane);
        else clip_head(q,qs,qo,state,128,true,uint2(h,0),lane);
        return;
    }
    threadgroup bfloat filtered[256];
    bool value=sg>=4;
    uint d=head*128+tid%128,h=d+(value?3072:0);
    device const bfloat *fixed=value?vs:ks;
    bfloat raw=(value?vp:kp)[d];
    float sum=0;
    for(ushort t=0;t<4;++t){
        bfloat x=t?old[h+(3-t)*1024]:raw;
        sum+=float(x)*(float(fixed[t*1024+d])+float(taps[(value?32:0)+head*4+t]));
    }
    filtered[tid]=bfloat((sum+float(raw))/3.f);
    next[h]=old[h+1024];next[h+1024]=old[h+2048];next[h+2048]=raw;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(sg>1)return;
    float square=0;
    if(sg==1){
        for(ushort j=lane;j<128;j+=32){float x=float(filtered[128+j]);square+=x*x;}
        float b=simd_shuffle(square,lane%8+8),c=simd_shuffle(square,lane%8+16),e=simd_shuffle(square,lane%8+24);
        square=((square+b)+c)+e;
    }else if(lane<8){
        for(ushort j=lane;j<128;j+=8){float x=float(filtered[j]);square+=x*x;}
    }
    float total=0;
    for(ushort j=0;j<8;++j)total+=simd_shuffle(square,j);
    float scale=sg?1.f/precise::sqrt(total/128.f+1e-6f):
        min(sqrt(512.f)*(1.f/(precise::sqrt(total)+1e-6f)),1.f);
    bfloat4 result=bfloat4(float4(*(threadgroup bfloat4*)(filtered+sg*128+lane*4))*scale);
    uint at=state[1]*1024+head*128+lane*4;
    device uchar *cache=sg?values:keys;
    if(state[5])for(ushort j=0;j<4;++j)cache[at+j]=table[as_type<ushort>(result[j])];
    else *(device bfloat4*)((device bfloat*)cache+at)=result;
}
kernel void relative_bias(device const bfloat *r,device const bfloat *w,device bfloat *out,
                          uint2 t [[thread_position_in_grid]]){
    uint distance=t.x*4,row=t.y;
    float4 sum[4]={};
    for(ushort d=0;d<16;++d){
        float4 w4=float4(*(device const bfloat4*)(w+d*1024+distance));
        sum[d%4]=fma(float4(r[row*16+d]),w4,sum[d%4]);
    }
    *(device bfloat4*)(out+row*1024+distance)=bfloat4(((sum[0]+sum[1])+sum[2])+sum[3]);
}
kernel void gate_context(device const float *context,device const bfloat *gate,device bfloat *out,
                         uint i [[thread_position_in_grid]]){
    float v=float(gate[i]);out[i]=bfloat(float(bfloat(context[i]))*(1.f/(1.f+reference_exp(-v))));
}
// expf polynomial adapted from SLEEF 3.9.0 (Boost Software License 1.0).
// Copyright Naoki Shibata and contributors 2010–2025. See SLEEF-LICENSE.txt.
inline float reference_exp(float x) {
    if(x < -104.f) return 0.f;
    if(x > 88.722839f) return INFINITY;
    float q=rint(x*1.4426950408889634f);
    float s=fma(q,-0.693145751953125f,x);
    s=fma(q,-1.428606765330187045e-6f,s);
    float u=0.00019852761761285365f;
    u=fma(u,s,0.0013930435525253415f);
    u=fma(u,s,0.008333360776305199f);
    u=fma(u,s,0.041666485369205475f);
    u=fma(u,s,0.1666666716337204f);
    u=fma(u,s,0.5f);
    return ldexp(1.f+fma(s*s,u,s),int(q));
}

// Follow the median-of-three quickselect partition used by arm64 libc++
// nth_element. Eager CPU sums the eight selected sigmoid scores in this order.
// Adapted from libc++ __algorithm/nth_element.h. See LIBCXX-LICENSE.txt.
inline void swap_index(thread ushort *p,uint a,uint b) {ushort v=p[a];p[a]=p[b];p[b]=v;}
inline void select_eight(thread ushort *p,threadgroup const float *rank) {
    uint first=0,last=128;
    while(last-first>7){
        uint m=(first+last)/2,tail=last-1;
        bool changed=false;
        if(rank[p[m]]>rank[p[first]]){swap_index(p,m,first);changed=true;}
        if(rank[p[tail]]>rank[p[m]]){
            swap_index(p,tail,m);changed=true;
            if(rank[p[m]]>rank[p[first]])swap_index(p,m,first);
        }
        uint i=first,j=tail;
        if(rank[p[i]]==rank[p[m]]){
            while(--j>i && rank[p[j]]<=rank[p[m]]){}
            if(j!=i){swap_index(p,i,j);changed=true;}
            else {
                i=first+1;j=tail;
                if(rank[p[first]]<=rank[p[j]]){
                    while(i<j && rank[p[first]]<=rank[p[i]])++i;
                    if(i==j)return;
                    swap_index(p,i++,j);
                }
                if(i==j)return;
                while(true){
                    while(rank[p[first]]<=rank[p[i]])++i;
                    do{--j;}while(rank[p[first]]>rank[p[j]]);
                    if(i>=j)break;
                    swap_index(p,i++,j);
                }
                if(7<i)return;
                first=i;continue;
            }
        }
        ++i;
        if(i<j)while(true){
            while(rank[p[i]]>rank[p[m]])++i;
            do{--j;}while(rank[p[j]]<=rank[p[m]]);
            if(i>=j)break;
            swap_index(p,i,j);changed=true;
            if(i==m)m=j;
            ++i;
        }
        if(i!=m && rank[p[m]]>rank[p[i]]){swap_index(p,i,m);changed=true;}
        if(i==7)return;
        uint left=7<i?first:i+1,right=7<i?i:last;
        if(!changed){
            uint a=7<i?first:i;
            while(a+1<right && rank[p[a+1]]<=rank[p[a]])++a;
            if(a+1==right)return;
        }
        first=left;last=right;
    }
    for(uint a=first;a+1<last;++a){
        uint best=a;
        for(uint b=a+1;b<last;++b)if(rank[p[b]]>rank[p[best]])best=b;
        swap_index(p,a,best);
    }
}
inline uint route_low_bits(uint limit,uint warp){
    uint start=warp*32;
    return limit<=start?0u:limit>=start+32?~0u:(1u<<(limit-start))-1u;
}
struct ExpertRouteScratch {
    float keys[128], scores[128];
    ushort orders[2][128], left[64], selected[8];
    ushort2 right[64];
    uint masks[4];
};

inline void normalize_expert_route(float probability,float key,device uint *ids,device float *route,
                                   uint tid,uint lane,uint sg,threadgroup ExpertRouteScratch &scratch){
    threadgroup float *keys=scratch.keys,*scores=scratch.scores;
    threadgroup auto &orders=scratch.orders;
    threadgroup ushort *left=scratch.left,*selected=scratch.selected,*p=orders[0];
    threadgroup ushort2 *right=scratch.right;
    threadgroup uint *masks=scratch.masks;
    uint phase=0;
    scores[tid]=probability;keys[tid]=key;p[tid]=tid;
    uint begin=0,end=128;bool slow=false;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    while(end-begin>7){
        uint first=begin,last=end-1,m=(begin+end)/2;
        ushort a=p[first],pivot=p[m],c=p[last];
        if(keys[a]<keys[pivot]){ushort v=a;a=pivot;pivot=v;}
        if(keys[pivot]<keys[c]){ushort v=pivot;pivot=c;c=v;}
        if(keys[a]<keys[pivot]){ushort v=a;a=pivot;pivot=v;}
        slow=keys[a]==keys[pivot] || keys[pivot]==keys[c];
        if(slow)break;
        ushort item=tid==first?a:tid==last?c:tid==m?pivot:p[tid];
        bool high=tid>=first && tid<=last && keys[item]>keys[pivot];
        uint mask=uint((simd_vote::vote_t)simd_ballot(high));
        if(!lane)masks[sg]=mask;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint4 high_bits=uint4(masks[0],masks[1],masks[2],masks[3]);
        uint4 counts=popcount(high_bits);
        uint bound=first+counts.x+counts.y+counts.z+counts.w;
        bool l=tid>=first && tid<bound && !high;
        bool r=tid>=bound && high;
        uint index=0;
        if(l){
            index=popcount((~mask)&route_low_bits(bound,sg)&~route_low_bits(first,sg)&((1u<<lane)-1));
            for(ushort j=0;j<sg;++j)index+=popcount((~high_bits[j])&route_low_bits(bound,j)&~route_low_bits(first,j));
            left[index]=item;
        }else if(r){
            index=popcount(mask&(~1u<<lane));
            for(ushort j=sg+1;j<4;++j)index+=popcount(high_bits[j]&~route_low_bits(bound,j));
            right[index]=ushort2(item,tid);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // The median sits at m before the partition. Follow its single
        // possible swap directly, then place it at the partition boundary.
        uint at=m,other=bound==first?a:bound==m?pivot:bound==last?c:p[bound];
        if(m<bound){
            uint order=0;
            for(ushort j=0;j<4;++j)order+=popcount((~high_bits[j])&route_low_bits(m,j)&~route_low_bits(first,j));
            at=right[order].y;
        }
        if((high_bits[bound/32]>>(bound%32))&1){
            uint order=0;
            for(ushort j=0;j<4;++j)order+=popcount(high_bits[j]&~route_low_bits(bound+1,j));
            other=left[order];
        }
        if(l)item=right[index].x;else if(r)item=left[index];
        if(at!=bound && keys[pivot]>keys[other]){
            if(tid==at)item=other;
            if(tid==bound)item=pivot;
        }
        phase^=1;p=orders[phase];
        p[tid]=item;
        if(bound==7)end=begin;
        else if(bound<7)begin=bound+1;
        else end=bound;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(!tid){
        ushort result[8];
        if(!slow){
            for(uint i=begin;i+1<end;++i){
                uint best=i;
                for(uint j=i+1;j<end;++j){if(keys[p[j]]==keys[p[best]])slow=1;if(keys[p[j]]>keys[p[best]])best=j;}
                ushort tmp=p[i];p[i]=p[best];p[best]=tmp;
            }
        }
        if(slow){
            ushort order[128];for(uint i=0;i<128;++i)order[i]=i;
            select_eight(order,keys);for(uint i=0;i<8;++i)result[i]=order[i];
        }else for(uint i=0;i<8;++i)result[i]=p[i];
        for(ushort j=0;j<8;++j)selected[j]=result[j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(!sg){
        uint id=selected[lane%8];float value=scores[id];
        float total=0;uint rank=0;
        for(ushort j=0;j<8;++j){total+=simd_shuffle(value,j);rank+=simd_shuffle(id,j)<id;}
        if(lane<8){ids[rank]=id;route[rank]=value/total*8.f;}
    }
}
kernel void expert_route(device const float *logits,device const float *bias,device uint *ids,
                                   device float *route,uint row [[threadgroup_position_in_grid]],
                                   uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],
                                   uint sg [[simdgroup_index_in_threadgroup]]){
    threadgroup ExpertRouteScratch scratch;
    float value=1.f/(1.f+reference_exp(-logits[row*128+tid]));
    normalize_expert_route(value,value+bias[tid],ids+row*8,route+row*8,tid,lane,sg,scratch);
}



// Vision features replace the normalized text embedding at the image token.
kernel void normalize_embedding(device const bfloat *input,device const bfloat *weight,
                                device const uchar *image,device bfloat *out,
                                uint row [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
                                uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){
    if(image[row]){
        for(uint d=tid;d<2304;d+=256)out[row*2304+d]=input[row*2304+d];
        return;
    }
    threadgroup float sums[160];
    normalize_row(input,weight,out,uint4(1,2304,0,0),row,tid,sg,lane,sums);
}

// All attention-residual queries share a tiny FP32 matrix product per token.
// Consecutive, eight-element MMA fragments preserve the panel accumulation order.
kernel void depth_matrix(device const float *s0,device const float *s1,device const float *s2,
                         device const float *s3,device const float *s4,device const float *queries,
                         device float *out,constant uint4 &dims,uint2 group [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]],uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]){
    device const float *states[5]={s0,s1,s2,s3,s4};
    uint sources=dims.x,nq=dims.y,rows=dims.z;
    uint token=group.x;
    queries+=group.y*nq*2304;
    out+=group.y*nq*rows*2304;
    uint r=((lane/4)&4)+(lane/2)%4,c=((lane/4)&2)*2+(lane%2)*2;
    bool single=sources==2 && nq==4;
    uint nk=single?2304:1024,panels=(2304+nk-1)/nk;
    threadgroup float part[3][4][8],weight[4][5];
    if(sg<panels){
        simdgroup_float8x8 acc=make_filled_simdgroup_matrix<float,8,8>(0.f);
        for(uint k=sg*nk;k<min(2304u,(sg+1)*nk);k+=8){
            simdgroup_float8x8 a,b;
            for(uint e=0;e<2;++e){
                a.thread_elements()[e]=r<nq?queries[r*2304+k+c+e]:0.f;
                b.thread_elements()[e]=c+e<sources?states[c+e][token*2304+k+r]:0.f;
            }
            simdgroup_multiply_accumulate(acc,a,b,acc);
        }
        if(r<nq)for(uint e=0;e<2;++e)part[sg][r][c+e]=acc.thread_elements()[e];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid<nq){
        float peak=-INFINITY,z=0;
        for(uint j=0;j<sources;++j){
            float v=0;for(uint p=0;p<panels;++p)v+=part[p][tid][j];
            v*=1.f/48.f;weight[tid][j]=v;peak=max(peak,v);
        }
        for(uint j=0;j<sources;++j){float v=reference_exp(weight[tid][j]-peak);weight[tid][j]=v;z+=v;}
        if(sources>=4)z=((weight[tid][0]+(sources==5?weight[tid][4]:0.f))+weight[tid][2])+(weight[tid][1]+weight[tid][3]);
        for(uint j=0;j<sources;++j)weight[tid][j]*=1.f/z;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint d=tid;d<2304;d+=256){
        float result[4]={0,0,0,0};
        for(uint j=0;j<sources;++j){
            float value=states[j][token*2304+d];
            for(uint q=0;q<nq;++q)result[q]=fma(value,weight[q][j],result[q]);
        }
        for(uint q=0;q<nq;++q)out[(q*rows+token)*2304+d]=result[q];
    }
}

// The CPU single-query dot divides each residual source into 512-channel
// panels, each with four ordered FMA streams. One SIMD group evaluates
// all sources for a panel and stream together.
kernel void depth_single(device const float *s0,device const float *s1,device const float *s2,
                         device const float *s3,device const float *s4,device const float *query,
                         device float *out,constant uint2 &dims,uint row [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]],uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]){
    #pragma clang fp contract(off)
    device const float *states[5]={s0,s1,s2,s3,s4};
    uint count=dims.x,panel=sg/4,residue=sg%4;
    uint r=((lane/4)&4)+(lane/2)%4,c=((lane/4)&2)*2+(lane%2)*2;
    uint start=panel*512,stop=min(start+512,2304u),n=(stop-start)/4;
    simdgroup_float8x8 acc=make_filled_simdgroup_matrix<float,8,8>(0.f);
    for(uint k=0;k<n;k+=8){
        simdgroup_float8x8 a,b;
        for(ushort e=0;e<2;++e){
            uint qa=k+c+e, kb=k+r;
            a.thread_elements()[e]=!r && qa<n-1 ? query[start+qa*4+residue] : 0.f;
            b.thread_elements()[e]=c+e<count && kb<n-1 ? states[c+e][row*2304+start+kb*4+residue] : 0.f;
        }
        simdgroup_multiply_accumulate(acc,a,b,acc);
    }
    threadgroup float parts[5][5][4],weights[5];
    if(!r)for(ushort e=0;e<2;++e)if(c+e<count){
        uint d=stop-4+residue;
        parts[c+e][panel][residue]=acc.thread_elements()[e]+query[d]*states[c+e][row*2304+d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid<count){
        float score=0;
        bool odd=(count%2 && tid==count-1);
        for(ushort p=0;p<5;++p){
            threadgroup const float *v=parts[tid][p];
            score+=odd?(v[0]+v[2])+(v[1]+v[3]):(v[0]+v[1])+(v[2]+v[3]);
        }
        weights[tid]=score*(1.f/48.f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(!sg){
        float v=lane<count?weights[lane]:-INFINITY;
        float peak=simd_max(v);
        float e=lane<count?reference_exp(v-peak):0.f;
        float z=0;for(ushort j=0;j<count;++j)z+=simd_shuffle(e,j);
        if(count>=4)z=(simd_shuffle(e,0)+simd_shuffle(e,4)+simd_shuffle(e,2))+(simd_shuffle(e,1)+simd_shuffle(e,3));
        if(lane<count)weights[lane]=e*(1.f/z);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint d=tid;d<2304;d+=640){
        float v=0;for(ushort j=0;j<count;++j)v=fma(weights[j],states[j][row*2304+d],v);
        out[row*2304+d]=v;
    }
}

// Reduce a threadgroup-resident hidden vector in the same order as the
// contiguous FP32 CPU RMS normalization.
inline float hidden_inverse(threadgroup const float *values,threadgroup float *sums,
                            uint sg,uint lane){
    #pragma clang fp contract(off)
    if(sg<5){
        uint start=sg*512+lane;
        float subtotal=0;
        for(uint d=start;d<min(start+512,2304u);d+=32){
            float v=values[d];subtotal+=v*v;
        }
        sums[sg*32+lane]=subtotal;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(!sg){
        float v=sums[lane]+sums[32+lane];
        v+=sums[64+lane];v+=sums[96+lane];v+=sums[128+lane];
        float a=simd_shuffle(v,lane%8+8),b=simd_shuffle(v,lane%8+16),c=simd_shuffle(v,lane%8+24);
        v=((v+a)+b)+c;
        float total=0;
        for(ushort j=0;j<8;++j)total+=simd_shuffle(v,j);
        if(!lane)sums[0]=1.f/precise::sqrt(total/2304.f+1.e-6f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return sums[0];
}

// Each persistent residual is reused by several layer visits. One matrix
// product scores eight downstream queries against the same residual.
kernel void fixed_depth_eight(device const float *state,device const float *table,device const uint *indices,
                              device float *out,constant uint &queries,uint group [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]],uint sg [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]){
    #pragma clang fp contract(off)
    uint panel=sg/4,residue=sg%4,start=panel*512,stop=min(start+512,2304u),n=(stop-start)/4;
    uint r=((lane/4)&4)+(lane/2)%4,c=((lane/4)&2)*2+(lane%2)*2;
    table+=group*2304*8;
    simdgroup_float8x8 acc=make_filled_simdgroup_matrix<float,8,8>(0.f);
    for(uint k=0;k<n;k+=8){
        simdgroup_float8x8 a,b;
        float x=state[start+(k+r)*4+residue];
        for(ushort e=0;e<2;++e){
            a.thread_elements()[e]=k+c+e<n-1?table[(start+(k+c+e)*4+residue)*8+r]:0.f;
            b.thread_elements()[e]=x;
        }
        simdgroup_multiply_accumulate(acc,a,b,acc);
    }
    threadgroup float part[8][5][4];
    uint d=stop-4+residue;
    if(!c)part[r][panel][residue]=acc.thread_elements()[0]+table[d*8+r]*state[d];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid<8 && group*8+tid<queries){
        float sum=0;
        for(ushort p=0;p<5;++p){
            threadgroup const float *v=part[tid][p];
            sum+=(v[0]+v[1])+(v[2]+v[3]);
        }
        out[indices[group*8+tid]]=sum*(1.f/48.f);
    }
}
constant ushort depth_config [[function_constant(1)]];
// A full 512-channel panel contains two runs of 64 samples for each of
// the four interleaved reduction streams. The short final panel contains one.
inline void accumulate_depth_half_panel(thread simdgroup_float8x8 &sum,
    device const float *query,threadgroup const float *state,uint r,uint c){
    #pragma clang loop unroll(full)
    for(ushort k=0;k<64;k+=8){
        simdgroup_float8x8 a,b;
        float2 weights=*(device const float2*)(query+k+c);
        float value=state[(k+r)*4];
        for(ushort j=0;j<2;++j){
            a.thread_elements()[j]=weights[j];
            b.thread_elements()[j]=value;
        }
        simdgroup_multiply_accumulate(sum,a,b,sum);
    }
}
// Each 2304-channel input supplies nine consecutive groups of eight
// products to every reference stream. Keep both inputs in the same sum.
template<typename Input>
inline void accumulate_controller_input(thread simdgroup_float8x8 &sum,device const bfloat2 *weights,
               Input input,uint stream,uint r,uint c,uint lane){
    #pragma clang loop unroll_count(9)
    for(ushort k=0;k<9;++k){
        simdgroup_matrix<bfloat,8,8> a,b;
        auto w=weights[k*32+lane];
        auto x=input[(k*8+r)*16+(stream+c)/2];
        for(ushort j=0;j<2;++j){
            a.thread_elements()[j]=w[j];
            b.thread_elements()[j]=x[j];
        }
        simdgroup_multiply_accumulate(sum,a,b,sum);
    }
}
// Each lane owns one depth score until the normalized mixture is ready.
inline void normalize_depth_logits(threadgroup float *weights,float logit,uint count,uint lane){
    #pragma clang fp contract(off)
    float peak=simd_max(logit);
    float e=lane<count?reference_exp(logit-peak):0.f;
    float z=0;
    if(count<4){
        for(ushort j=0;j<count;++j)z+=simd_shuffle(e,j);
    }else{
        z=(simd_shuffle(e,0)+simd_shuffle(e,4)+simd_shuffle(e,2))+(simd_shuffle(e,1)+simd_shuffle(e,3));
    }
    if(lane<count)weights[lane]=e*(1.f/z);
}
template<bool Wide>
inline void write_gate_ids(threadgroup const float *keys,threadgroup uint *masks,device uint *ids,
                           uint tid,uint sg,uint lane){
    uint higher=0,mask=0;
    if(!Wide || tid<128){
        float key=keys[tid];
        for(uint i=0;i<128;i+=4){
            float4 v=*(threadgroup const float4*)(keys+i);
            higher+=uint(v.x>key)+uint(v.y>key)+uint(v.z>key)+uint(v.w>key);
        }
        mask=uint((simd_vote::vote_t)simd_ballot(higher<8));
        if(!lane)masks[sg]=mask;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint count=0,offset=0;
    for(ushort j=0;j<4;++j){
        uint n=popcount(masks[j]);
        count+=n;if(j<sg)offset+=n;
    }
    if(count==8){
        if((!Wide || tid<128) && higher<8)ids[offset+popcount(mask&((1u<<lane)-1))]=tid;
    }else{
        if(!tid){
            ushort order[128];
            for(ushort j=0;j<128;++j)order[j]=j;
            select_eight(order,keys);
            for(ushort j=0;j<8;++j){
                ushort best=j;
                for(ushort i=j+1;i<8;++i)if(order[i]<order[best])best=i;
                swap_index(order,j,best);ids[j]=order[j];
            }
        }
    }
}

struct RoutePublication {
    float2 rank[128];
    atomic_uint arrivals;
};
#if __METAL_VERSION__ >= 320
#define INK_ROUTE_COHERENT coherent(device)
struct PublishedGateScratch {
    float keys[128];
    uint masks[4],last;
};

// Each producer releases its ranks before counting its arrival. Only the last
// group acquires all published ranks. Other groups exit without waiting.
template<uint Width>
inline void publish_gate(float logit,device const float *bias,uint group,uint tid,uint sg,uint lane,
                         coherent(device) device RoutePublication *pub,device uint *ids,
                         threadgroup PublishedGateScratch &scratch){
    if(sg<Width && !lane){
        uint row=group*Width+sg;
        float probability=1.f/(1.f+reference_exp(-logit));
        pub->rank[row]=float2(probability+bias[row],probability);
    }
    threadgroup_barrier(mem_flags::mem_device);
    if(!tid){
        atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
        uint arrival=atomic_fetch_add_explicit(&pub->arrivals,1u,memory_order_relaxed);
        scratch.last=arrival==(128/Width-1);
        if(scratch.last){
            atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
            atomic_store_explicit(&pub->arrivals,0u,memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_device|mem_flags::mem_threadgroup);
    if(!scratch.last)return;
    if(tid<128)scratch.keys[tid]=pub->rank[tid].x;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    write_gate_ids<true>(scratch.keys,scratch.masks,ids,tid,sg,lane);
}
#else
#define INK_ROUTE_COHERENT
#endif
template<uint Threads>
inline void mix_decode_depth_row(device const float *s0,device const float *s1,device const float *s2,device const float *s3,
    device const float *f0,device const float *f1,device const float *f2,device const float *f3,
    device const bfloat *hidden,device const float *query,device float *residual,device uchar *output,bool primary,
    uint tid,uint sg,uint lane,threadgroup float *local,threadgroup float *sums,
    threadgroup float (&parts)[5][4],threadgroup float *weights){
    #pragma clang fp contract(off)
    device const float *states[4]={s0,s1,s2,s3};
    uint count=(depth_config & 7u),panel=sg/4,residue=sg%4;
    for(uint d=tid;d<2304;d+=Threads){
        local[d]=(depth_config & 32u)?float(hidden[d]):residual[d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv=hidden_inverse(local,sums,sg,lane);
    for(uint d=tid;d<2304;d+=Threads){
        local[d]*=inv;
        if((depth_config & 32u) && (depth_config & 8u) && primary)residual[d]=local[d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint r=((lane/4)&4)+(lane/2)%4,c=((lane/4)&2)*2+(lane%2)*2;
    if(sg<20){
        uint start=panel*512,stop=min(start+512,2304u);
        simdgroup_float8x8 acc=make_filled_simdgroup_matrix<float,8,8>(0.f);
        device const float *stream=query+sg*128;
        threadgroup const float *state=local+start+residue;
        accumulate_depth_half_panel(acc,stream,state,r,c);
        if(panel<4)accumulate_depth_half_panel(acc,stream+64,state+256,r,c);
        if(!lane){
            uint d=stop-4+residue;
            parts[panel][residue]=acc.thread_elements()[0]+query[20*128+sg]*local[d];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(!sg){
        float logit=-INFINITY;
        if(lane+1<count){
            device const float *fixed[4]={f0,f1,f2,f3};
            logit=fixed[lane][0];
        }else if(lane==count-1){
            float score=0;
            for(ushort p=0;p<5;++p){
                threadgroup const float *v=parts[p];
                score+=count%2?(v[0]+v[2])+(v[1]+v[3]):(v[0]+v[1])+(v[2]+v[3]);
            }
            logit=score*(1.f/48.f);
        }
        normalize_depth_logits(weights,logit,count,lane);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint d=tid;d<2304;d+=Threads){
        float v=0;
        for(ushort j=0;j<count-1;++j)v=fma(weights[j],states[j][d],v);
        v=fma(weights[count-1],local[d],v);
        if(depth_config & (16u|128u))local[d]=v;
        if(!(depth_config & 16u) && primary)((device float*)output)[d]=v;
    }
}
kernel void decode_depth_input(device const float *s0,device const float *s1,device const float *s2,
                             device const float *s3,INK_ROUTE_COHERENT device RoutePublication *publication,device const bfloat *hidden,
                             device const float *route_bias,device uint *gate_ids,device const bfloat *conv,
                             device const float *query,device const bfloat *norm_weight,device float *residual,
                             device uchar *output,device const float *f0,device const float *f1,device const float *f2,device const float *f3,device const bfloat *route_weight,device float *route_logits,device const bfloat *message,device const bfloat *key_weight,device const bfloat *value_weight,
                             uint2 tile [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){
    #pragma clang fp contract(off)
#if __METAL_VERSION__ >= 320
    threadgroup PublishedGateScratch scratch;
#endif
    uint group=tile.x;
    bool primary=!tile.x && !tile.y;
    threadgroup bfloat activated[2304];
    threadgroup float controller_parts[4][32];
    threadgroup float local[2304],sums[160],parts[5][4],weights[5];
    mix_decode_depth_row<640>(s0,s1,s2,s3,f0,f1,f2,f3,hidden,query,residual,output,primary,tid,sg,lane,local,sums,parts,weights);
    uint r=((lane/4)&4)+(lane/2)%4,c=((lane/4)&2)*2+(lane%2)*2;
    if(depth_config & (16u|128u)){
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float scale=hidden_inverse(local,sums,sg,lane);
        for(uint d=tid;d<2304;d+=640){
            if(depth_config & 128u){activated[d]=bfloat(local[d]*scale);continue;}
            bfloat value=bfloat(local[d]*scale*float(norm_weight[d]));
            if(depth_config & 64u){
                activated[d]=value;
                if(primary)((device bfloat*)output)[packed_channel(d)]=value;
            }else ((device bfloat*)output)[(depth_config & 32u)?packed_channel(d):d]=value;
        }
        if(depth_config & 128u){
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(depth_config & 256u){
                route_weight=tile.y==0?route_weight:tile.y==1?key_weight:value_weight;
                message+=tile.y*2304;
                route_logits+=tile.y*32; // 64 BF16 results per bottleneck
            }
            if(sg<16){
                uint row=group*4+sg/4,stream=(sg%4)*8;
                device const bfloat2 *panel=(device const bfloat2*)route_weight+(row*4+sg%4)*18*32;
                simdgroup_float8x8 dot=make_filled_simdgroup_matrix<float,8,8>(0.f);
                accumulate_controller_input(dot,panel,(threadgroup const bfloat2*)activated,stream,r,c,lane);
                accumulate_controller_input(dot,panel+9*32,(device const bfloat2*)message,stream,r,c,lane);
                if(r==c || r==c+1)controller_parts[sg/4][stream+r]=dot.thread_elements()[r%2];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(sg<4){
                float result=reference_sum(controller_parts[sg][lane]);
                if(!lane)((device bfloat*)route_logits)[group*4+sg]=bfloat(result);
            }
        }
        if(depth_config & 64u){
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint row=group*16+sg;
            float logit=0;
            if(sg<16)logit=packed_router_dot(activated,route_weight+row*2304,lane);
#if __METAL_VERSION__ >= 320
            if(depth_config & 512u)
                publish_gate<16>(logit,route_bias,group,tid,sg,lane,publication,gate_ids,scratch);
            else
#endif
            if(sg<16 && !lane)route_logits[row]=logit;
        }
    }
}

// Each group constructs and normalizes one previous-pass message.
kernel void decode_previous_pass(device const float *s0,device const float *s1,device const float *query,
                                 device float *raw,device bfloat *normalized,uint group [[threadgroup_position_in_grid]],
                                 uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){
    #pragma clang fp contract(off)
    constexpr uint sections=1;
    threadgroup float2 factors[sections];
    threadgroup float sums[160];
    if(sg<sections){
        uint r=((lane/4)&4)+(lane/2)%4,c=((lane/4)&2)*2+(lane%2)*2;
        simdgroup_float8x8 acc=make_filled_simdgroup_matrix<float,8,8>(0.f);
        query+=group*2304;
        for(uint j=sg*288/sections;j<(sg+1)*288/sections;++j){
            simdgroup_float8x8 a,b;
            a.thread_elements()[0]=query[j*8+c];a.thread_elements()[1]=query[j*8+c+1];
            b.thread_elements()[0]=s0[j*8+r];b.thread_elements()[1]=s1[j*8+r];
            simdgroup_multiply_accumulate(acc,a,b,acc);
        }
        if(!lane)factors[sg]=float2(acc.thread_elements()[0],acc.thread_elements()[1]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float2 scores=factors[0];
    for(ushort i=1;i<sections;i++)scores+=factors[i];
    scores*=1.f/48.f;
    float peak=max(scores.x,scores.y);
    float2 w=float2(reference_exp(scores.x-peak),reference_exp(scores.y-peak));
    w*=1.f/(w.x+w.y);
    float values[16],subtotal=0;
    uint start=sg*512+lane;
    for(uint j=0;j<(sg==4?8u:16u);j++){
        uint col=start+32*j;
        float v=fma(s1[col],w.y,fma(s0[col],w.x,0.f));
        values[j]=v;raw[group*2304+col]=v;subtotal+=v*v;
    }
    sums[sg*32+lane]=subtotal;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float v=(((sums[lane]+sums[32+lane])+sums[64+lane])+sums[96+lane])+sums[128+lane];
    float a=simd_shuffle(v,(lane%8)+8),b=simd_shuffle(v,(lane%8)+16),c=simd_shuffle(v,(lane%8)+24);
    v=((v+a)+b)+c;
    float total=0;
    for(ushort j=0;j<8;j++)total+=simd_shuffle(v,j);
    float inv=1.f/precise::sqrt(total/2304.f+1e-6f);
    for(uint j=0;j<(sg==4?8u:16u);j++)normalized[group*2304+start+j*32]=bfloat(values[j]*inv);
}

// Each of the three global attention inputs uses its own depth query and
// normalization. Preserve the distinct short- and long-source dot orders.
template<class Source>
inline float global_short_depth_score(device const float *a,Source b,uint l){
    #pragma clang fp contract(off)
    uint ch=l%4;
    float v=0;
    if(l>=8)v=l<12?a[ch]*b[ch]:a[380+ch]*b[380+ch];
    for(uint k=4;k<372;k+=16)v=fma(a[k+l],b[k+l],v);
    v+=simd_shuffle_xor(v,8);
    uint tail=372+l%8;v=fma(a[tail],b[tail],v);
    v+=simd_shuffle_xor(v,4);v+=simd_shuffle_xor(v,2);return v+simd_shuffle_xor(v,1);
}
kernel void global_depth_inputs(device const float *s0,device const float *s1,device const float *s2,
    device const float *s3,device const bfloat *hidden,device const float *queries,
    device const bfloat *w0,device const bfloat *w1,device const bfloat *w2,
    device const bfloat *m0,device const bfloat *m1,device const bfloat *m2,
    device float *raw,device bfloat *out,constant uint2 &dims,uint2 tile [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){
    #pragma clang fp contract(off)
    uint q=tile.y,threads=dims.y?512:288;
    device const float *states[4]={s0,s1,s2,s3};
    device const float *query=queries+q*2304;
    uint count=dims.x;
    threadgroup float parts[5][6],weights[5],mixed[2304],sums[160];
    threadgroup float current[2304];
    for(uint d=tid;d<2304;d+=threads)current[d]=float(hidden[d]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float scale=hidden_inverse(current,sums,sg,lane);
    for(uint d=tid;d<2304;d+=threads)current[d]*=scale;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(count<4){
        uint src=tid/96,panel=(tid%96)/16,l=tid%16,ch=l%4;
        if(src<count){
            device const float *a=query+panel*384;
            float v=src+1==count?global_short_depth_score(a,current+panel*384,l):
                global_short_depth_score(a,states[src]+panel*384,l);
            if(!l)parts[src][panel]=v;
        }
    }else if(sg<3){
        uint r=((lane/4)&4)+(lane/2)%4,c=((lane/4)&2)*2+(lane%2)*2;
        simdgroup_float8x8 acc=make_filled_simdgroup_matrix<float,8,8>(0.f);
        for(uint k=sg*1024;k<min((sg+1)*1024,2304u);k+=8){
            simdgroup_float8x8 a,b;
            for(ushort e=0;e<2;++e){
                a.thread_elements()[e]=query[k+c+e];
                b.thread_elements()[e]=c+e==count-1?current[k+r]:(c+e<count?states[c+e][k+r]:0.f);
            }
            simdgroup_multiply_accumulate(acc,a,b,acc);
        }
        if(!r)for(ushort e=0;e<2;++e)if(c+e<count)parts[c+e][sg]=acc.thread_elements()[e];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(!sg){
        float logit=-INFINITY;
        if(lane<count){
            float score=0;
            for(ushort i=0;i<(count<4?6:3);++i)score+=parts[lane][i];
            logit=score*(1.f/48.f);
        }
        normalize_depth_logits(weights,logit,count,lane);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint d=tid;d<2304;d+=threads){
        float v=0;for(uint j=0;j+1<count;++j)v=fma(states[j][d],weights[j],v);
        v=fma(current[d],weights[count-1],v);
        mixed[d]=v;if(dims.y && tile.x==0)raw[q*2304+d]=v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv=hidden_inverse(mixed,sums,sg,lane);
    device const bfloat *norms[3]={w0,w1,w2},*msgs[3]={m0,m1,m2};
    threadgroup bfloat activated[2304];
    threadgroup float controller_parts[4][32];
    for(uint d=tid;d<2304;d+=threads){
        float v=mixed[d]*inv;
        if(dims.y)activated[d]=bfloat(v);
        else out[q*2304+d]=bfloat(v*float(norms[q][d]));
    }
    if(!dims.y)return;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint r=((lane/4)&4)+(lane/2)%4,c=((lane/4)&2)*2+(lane%2)*2;
    uint row=tile.x*4+sg/4,stream=(sg%4)*8;
    device const bfloat2 *panel=(device const bfloat2*)norms[q]+(row*4+sg%4)*18*32;
    simdgroup_float8x8 dot=make_filled_simdgroup_matrix<float,8,8>(0.f);
    accumulate_controller_input(dot,panel,(threadgroup const bfloat2*)activated,stream,r,c,lane);
    accumulate_controller_input(dot,panel+9*32,(device const bfloat2*)msgs[q],stream,r,c,lane);
    if(r==c || r==c+1)controller_parts[sg/4][stream+r]=dot.thread_elements()[r%2];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(sg<4){
        float result=reference_sum(controller_parts[sg][lane]);
        if(!lane)out[q*64+tile.x*4+sg]=bfloat(result);
    }
}

// The gate can start from membership alone. On an eighth-place tie,
// recover the same winners as the reference partition before projecting.
kernel void gate_expert_indices(device const float *logits,device const float *bias,device uint *ids,
                                device float2 *rank,uint tid [[thread_index_in_threadgroup]],
                                uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){
    threadgroup float keys[128];
    threadgroup uint masks[4];
    float probability=1.f/(1.f+reference_exp(-logits[tid]));
    float key=probability+bias[tid];
    rank[tid]=float2(key,probability);
    keys[tid]=key;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    write_gate_ids<false>(keys,masks,ids,tid,sg,lane);
}

kernel void gate_and_route_weights(device const bfloat *x,device const uint *gate_ids,device const uchar *w,
    device const uchar *scales,device const bfloat *bank,device const bfloat *codes,device const float *activation,
    device bfloat *out,device const float2 *rank,device uint *ids,device float *route,
    uint3 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]){
    threadgroup ExpertRouteScratch scratch;
    if(g.x){
        --g.x;
        select_gate_rows<true>(x,gate_ids,w,scales,bank,codes,activation,out,g,4,sg,lane);
    }else if(!g.y){
        float2 entry=rank[tid];
        normalize_expert_route(entry.y,entry.x,ids,route,tid,lane,sg,scratch);
    }
}

kernel void build_attention_sigmoid(device float *out,uint i [[thread_position_in_grid]]){
    out[i]=1.f/(1.f+reference_exp(-float(as_type<bfloat>(ushort(i)))));
}
kernel void lookup_gate_context(device const float *context,device const bfloat *gate,device bfloat *out,
                                device const float *table,uint i [[thread_position_in_grid]]){
    out[i]=bfloat(float(bfloat(context[i]))*table[as_type<ushort>(gate[i])]);
}

kernel void joint_output_20(device const bfloat *x,device const bfloat *weight,device const bfloat *hidden,
    device const bfloat *conv,device const bfloat *old,device bfloat *next,device float *residual,
    device const float *state,device const float *table,device const uint *indices,device float *out,
    constant uint &queries,uint group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){
    #pragma clang fp contract(off)
    uint query_groups=(queries+7)/8;
    if(group>=query_groups){
        group-=query_groups;
    uint d=group*20+sg;
    if(d>=2304)return;
    device const uint *wt_table=(device const uint*)((device const uchar*)weight+2304*3072*3/2)+(d/32)*256;
    float sum=0;
    for(uint k=0;k<24;++k){
        device const uchar *panel=(device const uchar*)weight+(d*24+k)*192;
        uint4 low=uint4(((device const uchar4*)panel)[lane]);
        uchar2 codes=((device const uchar2*)(panel+128))[lane];
        uint2 packed=uint2(wt_table[codes.x],wt_table[codes.y])|uint2(low.x|(low.y<<16),low.z|(low.w<<16));
        float4 w=float4(as_type<bfloat4>(packed));
        float4 act=float4(((device const bfloat4*)x)[k*32+lane]);
        for(ushort j=0;j<4;++j)sum=fma(act[j],w[j],sum);
    }
    float projected=reference_sum(sum);
    if(!lane)store_output_residual(projected,d,hidden,conv,old,next,residual,0);
        return;
    }
    threadgroup float part[8][5][4];
    table+=group*2304*8;
    for(ushort slot=sg;slot<20;slot+=20){
    uint panel=slot/4,residue=slot%4,start=panel*512,stop=min(start+512,2304u),n=(stop-start)/4;
    uint r=((lane/4)&4)+(lane/2)%4,c=((lane/4)&2)*2+(lane%2)*2;
    simdgroup_float8x8 acc=make_filled_simdgroup_matrix<float,8,8>(0.f);
    for(uint k=0;k<n;k+=8){
        simdgroup_float8x8 a,b;
        float x=state[start+(k+r)*4+residue];
        for(ushort e=0;e<2;++e){
            a.thread_elements()[e]=k+c+e<n-1?table[(start+(k+c+e)*4+residue)*8+r]:0.f;
            b.thread_elements()[e]=x;
        }
        simdgroup_multiply_accumulate(acc,a,b,acc);
    }
    uint d=stop-4+residue;
    if(!c)part[r][panel][residue]=acc.thread_elements()[0]+table[d*8+r]*state[d];

    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid<8 && group*8+tid<queries){
        float sum=0;
        for(ushort p=0;p<5;++p){
            threadgroup const float *v=part[tid][p];
            sum+=(v[0]+v[1])+(v[2]+v[3]);
        }
        out[indices[group*8+tid]]=sum*(1.f/48.f);
    }
}

kernel void depth_sliding_heads(device const bfloat *wq,device const bfloat *wr,device const bfloat *ws,
    device const bfloat *wk,device const bfloat *wv,device const bfloat *qs,device const bfloat *rs,
    device const bfloat *kw,device const bfloat *vw,device const bfloat *old,device bfloat *next,
    device uchar *keys,device uchar *vals,device const uchar *table,device const uint *state,
    device bfloat *queries,device float *rels,device bfloat *sinks,
    device const float *s0,device const float *s1,device const float *s2,device const float *s3,
    device const float *f0,device const float *f1,device const float *f2,device const float *f3,
    device const bfloat *hidden,device const float *depth_query,device const bfloat *norm_weight,
    device float *residual,device bfloat *normalized_query,
    uint bank [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){
    #pragma clang fp contract(off)
    uint tid=sg*32+lane;
    threadgroup bfloat activation[2304];
    threadgroup float head_scratch[128];
    threadgroup float local[2304],sums[160],parts[5][4],weights[5];
    mix_decode_depth_row<1024>(s0,s1,s2,s3,f0,f1,f2,f3,hidden,depth_query,residual,nullptr,bank==0,tid,sg,lane,local,sums,parts,weights);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float scale=hidden_inverse(local,sums,sg,lane);
    for(uint d=tid;d<2304;d+=1024){
        uint at=packed_channel(d);
        bfloat value=bfloat(local[d]*scale*float(norm_weight[d]));
        activation[at]=value;
        if(bank==0)normalized_query[at]=value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sliding_heads_impl<true,false,true>(nullptr,nullptr,nullptr,wq,wr,ws,nullptr,wk,wv,qs,rs,kw,vw,old,next,
        keys,vals,table,state,queries,rels,nullptr,sinks,bank,sg,lane,head_scratch,nullptr,nullptr,nullptr,nullptr,activation);
}
