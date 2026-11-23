// Dispatch the independent query/key/value message controllers together.
kernel void controller_norms(device const float *x0,device const float *x1,device const float *x2,
                             device const float *m0,device const float *m1,device const float *m2,
                             device bfloat *out,uint group [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]],uint sg [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]){
    device const float *rows[6]={x0,m0,x1,m1,x2,m2};
    threadgroup float sum[160];
    normalize_row(rows[group],(device const bfloat*)nullptr,out+group*2304,uint4(0,2304,0,0),0,tid,sg,lane,sum);
}
inline float gate_gemv64(device const bfloat *x,device const bfloat *w,uint lane){
    uint ch=4*(lane%4);
    float4 v=0;
    for(uint j=ch;j<64;j+=16)
        v+=float4(*(device const bfloat4*)(x+j))*float4(*(device const bfloat4*)(w+j));
    float group=((v.x+v.y)+v.z)+v.w;
    uint base=lane&~3;
    float total=simd_shuffle(group,base)+simd_shuffle(group,base+1);
    total+=simd_shuffle(group,base+2);return total+simd_shuffle(group,base+3);
}
kernel void controller_blend(device const bfloat *z,device const bfloat *f0,device const bfloat *f1,
                             device const bfloat *f2,device const bfloat *g0,device const bfloat *g1,
                             device const bfloat *g2,device const float *x0,device const float *x1,
                             device const float *x2,device const float *m0,device const float *m1,
                             device const float *m2,device float *out,uint2 g [[threadgroup_position_in_grid]],
                             uint lane [[thread_index_in_simdgroup]]){
    #pragma clang fp contract(off)
    device const bfloat *fw[3]={f0,f1,f2},*gw[3]={g0,g1,g2};
    device const float *xs[3]={x0,x1,x2},*ms[3]={m0,m1,m2};
    uint c=g.x*4+lane/8,j=g.y;
    float f=gate_gemv64(z+j*64,fw[j]+c*64,lane);
    float v=gate_gemv64(z+j*64,gw[j]+c*64,lane);
    if(lane%8==0)out[j*2304+c]=(1.f-reference_tanh(f))*xs[j][c]+reference_tanh(v)*ms[j][c];
}
kernel void input_norms(device const float *x0,device const float *x1,device const float *x2,
                       device const bfloat *w0,device const bfloat *w1,device const bfloat *w2,
                       device bfloat *out,constant uint &packed,uint group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){
    device const float *xs[3]={x0,x1,x2};device const bfloat *ws[3]={w0,w1,w2};
    threadgroup float sum[160];
    normalize_row(xs[group],ws[group],out+group*2304,uint4(1,2304,0,packed),0,tid,sg,lane,sum);
}
kernel void controller_down(device const bfloat *xs,device const bfloat *w0,device const bfloat *w1,device const bfloat *w2,device bfloat *out,
                     uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){
    device const bfloat *ws[3]={w0,w1,w2}; uint row=g.x,width=4608;
    device const bfloat *w=ws[g.y]+row*width,*x=xs+g.y*width;device bfloat *y=out+g.y*64;threadgroup float partial[8],magnitudes[8],result;
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

kernel void normalize_pass_messages(device const float *messages,device bfloat *out,uint group [[threadgroup_position_in_grid]],
                                   uint tid [[thread_index_in_threadgroup]],uint sg [[simdgroup_index_in_threadgroup]],
                                   uint lane [[thread_index_in_simdgroup]]){
    threadgroup float sum[160];
    normalize_row(messages+group*2304,(device const bfloat*)nullptr,out+group*2304,uint4(0,2304,0,0),0,tid,sg,lane,sum);
}

// Each group produces four of the 32 CPU RMS streams in one 512-channel panel.
kernel void blend_with_sums(device const bfloat *z,device const bfloat4 *w0,device const bfloat4 *w1,
                            device const bfloat4 *w2,device const float *x0,device const float *x1,
                            device const float *x2,device const float *m0,device const float *m1,
                            device const float *m2,device float *out,device float *parts,
                            uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){
    #pragma clang fp contract(off)
    device const bfloat4 *weights[3]={w0,w1,w2};
    device const float *states[3]={x0,x1,x2},*messages[3]={m0,m1,m2};
    uint panel=g.x/8,stream=(g.x%8)*4,rank=tid/4;
    uint col=panel*512+(rank/4)*32+stream+rank%4;
    threadgroup float values[64];
    if(col<2304){
        device const bfloat4 *w=weights[g.y]+((g.x*8+sg)*4*32+lane)*2;
        float4 forget=0,gain=0;
        for(ushort t=0;t<4;t++){
            float4 a=float4(*(device const bfloat4*)(z+g.y*64+t*16+(lane%4)*4));
            forget+=a*float4(w[t*64]);gain+=a*float4(w[t*64+1]);
        }
        float2 v=float2(((forget.x+forget.y)+forget.z)+forget.w,((gain.x+gain.y)+gain.z)+gain.w);
        uint base=lane&~3u;
        float2 total=simd_shuffle(v,base)+simd_shuffle(v,base+1);
        total+=simd_shuffle(v,base+2);total+=simd_shuffle(v,base+3);
        if(lane%4==0){
            float result=(1.f-reference_tanh(total.x))*states[g.y][col]+reference_tanh(total.y)*messages[g.y][col];
            out[g.y*2304+col]=result;values[rank]=result;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid<4){
        float sum=0;
        for(ushort i=0;i<(panel==4?8:16);i++){
            float v=values[i*4+tid];sum+=v*v;
        }
        parts[g.y*160+panel*32+stream+tid]=sum;
    }
}
inline float blend_rms_inverse(device const float *s,uint lane){
    #pragma clang fp contract(off)
    float v=(((s[lane]+s[lane+32])+s[lane+64])+s[lane+96])+s[lane+128];
    float a=simd_shuffle(v,lane%8+8),b=simd_shuffle(v,lane%8+16),c=simd_shuffle(v,lane%8+24);
    v=((v+a)+b)+c;
    float total=0;for(ushort i=0;i<8;i++)total+=simd_shuffle(v,i);
    return 1.f/precise::sqrt(total/2304.f+1e-6f);
}
kernel void normalize_blend_inputs(device const float *x0,device const float *x1,device const float *x2,
                                   device const bfloat *w0,device const bfloat *w1,device const bfloat *w2,
                                   device const float *parts,device bfloat *out,constant uint &packed,
                                   uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
                                   uint lane [[thread_index_in_simdgroup]]){
    device const float *xs[3]={x0,x1,x2};device const bfloat *ws[3]={w0,w1,w2};
    uint d=g.x*128+tid;
    float inv=blend_rms_inverse(parts+g.y*160,lane);
    out[g.y*2304+(packed?packed_channel(d):d)]=bfloat(xs[g.y][d]*inv*float(ws[g.y][d]));
}
inline float blend_router_projection(device const float *x,device const bfloat *weights,device const float *parts,
                                     device const bfloat *matrix,device bfloat *packed,threadgroup bfloat *act,
                                     uint group,uint tid,uint sg,uint lane){
    float inv=blend_rms_inverse(parts,lane);
    for(uint d=tid;d<2304;d+=256){
        bfloat v=bfloat(x[d]*inv*float(weights[d]));
        act[d]=v;if(group==0)packed[packed_channel(d)]=v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint row=group*8+sg;
    float v=packed_router_dot(act,matrix+row*2304,lane);
    return v;
}
kernel void blend_router(device const float *x,device const bfloat *weights,device const float *parts,
                         device const bfloat *matrix,device float *logits,device bfloat *packed,
                         uint group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){
    threadgroup bfloat act[2304];
    float v=blend_router_projection(x,weights,parts,matrix,packed,act,group,tid,sg,lane);
    if(!lane)logits[group*8+sg]=v;
}
#if __METAL_VERSION__ >= 320
kernel void published_blend_router(device const float *x,device const bfloat *weights,device const float *parts,
                         device const bfloat *matrix,coherent(device) device RoutePublication *pub,device bfloat *packed,
                         device const float *bias,device uint *ids,
                         uint group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]){
    threadgroup bfloat act[2304];
    float v=blend_router_projection(x,weights,parts,matrix,packed,act,group,tid,sg,lane);
    threadgroup PublishedGateScratch scratch;
    publish_gate<8>(v,bias,group,tid,sg,lane,pub,ids,scratch);
}
#endif

// A group owns eight token rows. Each SIMD group owns eight of the thirty-two
// accumulation streams in one CPU reduction panel. This keeps the blended
// FP32 values in registers until their row's shared RMS factor is ready.
kernel void normalized_prompt_controller(device const bfloat *z, device const bfloat4 *weights,
    device const float *hidden, device const float *message, device const bfloat *norm,
    device bfloat *out, constant uint &rows, uint group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    #pragma clang fp contract(off)
    uint panel=sg/4, stripe=sg%4;
    uint r=((lane/4)&4)+(lane/2)%4, c=((lane/4)&2)*2+(lane%2)*2;
    uint row=group*8+r, src=min(row,rows-1), count=panel==4?8:16;
    simdgroup_float8x8 inputs[8];
    for (ushort k=0; k<8; ++k) {
        float2 a=float2(*(device const bfloat2*)(z+src*64+k*8+c));
        inputs[k].thread_elements()[0]=a.x; inputs[k].thread_elements()[1]=a.y;
    }
    float2 values[16], sum=0;
    for (ushort j=0; j<count; ++j) {
        uint tile=panel*64+j*4+stripe, d=tile*8+c;
        simdgroup_float8x8 f=make_filled_simdgroup_matrix<float,8,8>(0.f);
        simdgroup_float8x8 g=make_filled_simdgroup_matrix<float,8,8>(0.f);
        for (ushort k=0; k<8; ++k) {
            bfloat4 w=weights[(tile*8+k)*32+lane];
            simdgroup_float8x8 b,h;
            b.thread_elements()[0]=float(w.x); b.thread_elements()[1]=float(w.y);
            h.thread_elements()[0]=float(w.z); h.thread_elements()[1]=float(w.w);
            simdgroup_multiply_accumulate(f,inputs[k],b,f);
            simdgroup_multiply_accumulate(g,inputs[k],h,g);
        }
        float2 v;
        for (ushort e=0; e<2; ++e) {
            float kept=(1.f-reference_tanh(f.thread_elements()[e]))*hidden[src*2304+d+e];
            v[e]=kept+reference_tanh(g.thread_elements()[e])*message[src*2304+d+e];
        }
        values[j]=v; sum+=v*v;
    }
    threadgroup float parts[8][160], inv[8];
    parts[r][sg*8+c]=sum.x; parts[r][sg*8+c+1]=sum.y;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg<8) {
        threadgroup float *s=parts[sg];
        float v=(((s[lane]+s[lane+32])+s[lane+64])+s[lane+96])+s[lane+128];
        float a=simd_shuffle(v,lane%8+8), b=simd_shuffle(v,lane%8+16), h=simd_shuffle(v,lane%8+24);
        v=((v+a)+b)+h;
        float total=0; for(ushort i=0; i<8; ++i) total+=simd_shuffle(v,i);
        if(!lane) inv[sg]=1.f/precise::sqrt(total/2304.f+1e-6f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (row<rows) {
        for (ushort j=0; j<count; ++j) {
            uint d=panel*512+j*32+stripe*8+c;
            *(device bfloat2*)(out+row*2304+d)=bfloat2(values[j]*inv[r]*float2(*(device const bfloat2*)(norm+d)));
        }
    }
}

// Adapted from the MLX custom prompt controller's 8-row by 32-channel tile.
kernel void prompt_controller_blend(device const bfloat *x,device const bfloat4 *weights,device const float *hidden,
                                    device const float *msg,device float *out,constant uint &rows,
                                    uint2 group [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]]){

        #pragma clang fp contract(off)
        uint bank=group.x*4+sg;
        uint fm=((lane/4)&4)+(lane/2)%4,fn=((lane/4)&2)*2+(lane%2)*2;
        uint row=group.y*8+fm;
        uint valid=min(row,rows-1);
        simdgroup_float8x8 f[4],g[4];
        for (short j=0;j<4;++j) {
            f[j]=make_filled_simdgroup_matrix<float,8,8>(0.f);
            g[j]=make_filled_simdgroup_matrix<float,8,8>(0.f);
        }
        for (short k=0;k<8;++k) {
            simdgroup_float8x8 a;
            float2 v=float2(*(device const bfloat2*)(x+valid*64+k*8+fn));
            a.thread_elements()[0]=v.x; a.thread_elements()[1]=v.y;
            for (short j=0;j<4;++j) {
                bfloat4 w=weights[((bank*4+j)*8+k)*32+lane];
                simdgroup_float8x8 b,h;
                b.thread_elements()[0]=float(w.x);b.thread_elements()[1]=float(w.y);
                h.thread_elements()[0]=float(w.z);h.thread_elements()[1]=float(w.w);
                simdgroup_multiply_accumulate(f[j],a,b,f[j]);
                simdgroup_multiply_accumulate(g[j],a,h,g[j]);
            }
        }
        if (row<rows) {
            for (short j=0;j<4;++j) {
                uint col=(bank*4+j)*8+fn;
                for (ushort v=0;v<2;++v) {
                    float retained=(1.f-reference_tanh(f[j].thread_elements()[v]))*hidden[row*2304+(col+v)];
                    float added=reference_tanh(g[j].thread_elements()[v])*msg[row*2304+(col+v)];
                    out[row*2304+col+v]=retained+added;
                }
            }
        }
    }
