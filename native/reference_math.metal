#include <metal_stdlib>
using namespace metal;
// FP32 tanh with compensated arithmetic, adapted from SLEEF 3.9.0.
// Copyright Naoki Shibata and contributors 2010–2025. See SLEEF-LICENSE.txt.
inline float2 df_add(float2 a,float2 b){
    #pragma clang fp contract(off)
    float h=a.x+b.x,v=h-a.x;
    return float2(h,((a.x-(h-v))+(b.x-v))+(a.y+b.y));
}
inline float2 df_quick_add(float2 a,float2 b){
    #pragma clang fp contract(off)
    float h=a.x+b.x;
    return float2(h,(((a.x-h)+b.x)+a.y)+b.y);
}
inline float2 df_product(float2 a,float2 b){
    float h=a.x*b.x;
    return float2(h,fma(a.x,b.y,fma(a.y,b.x,fma(a.x,b.x,-h))));
}
inline float reference_tanh(float x){
    #pragma clang fp contract(off)
    float a=abs(x);
    if(a>8.664339742f)return copysign(1.f,x);
    float q=rint(a*1.4426950408889634f);
    float2 s=df_add(float2(a,0),float2(q*(-0.693145751953125f),0));
    s=df_add(s,float2(q*(-1.428606765330187045e-6f),0));
    float u=fma(0.1980960224e-3f,s.x,0.1394256484e-2f);
    u=fma(u,s.x,0.8333456703e-2f);u=fma(u,s.x,0.4166637361e-1f);
    float2 t=df_add(df_product(s,float2(u,0)),float2(0.16666665941423424f,0));
    t=df_add(df_product(s,t),float2(.5f,0));
    float sq=s.x*s.x;
    float2 square=float2(sq,fma(s.x+s.x,s.y,fma(s.x,s.x,-sq)));
    t=df_add(s,df_product(square,t));
    t=df_quick_add(float2(1,0),t);
    float2 d=ldexp(t,int2(int(q)));
    float inv=1.f/d.x;
    float2 rec=float2(inv,inv*fma(-d.y,inv,fma(-d.x,inv,1.f)));
    float2 n=df_quick_add(d,-rec),den=df_quick_add(d,rec);
    inv=1.f/den.x;
    float h=n.x*inv,residual=fma(-den.y,inv,fma(-den.x,inv,1.f));
    float l=fma(h,residual,fma(n.y,inv,fma(inv,n.x,-h)));
    return copysign(h+l,x);
}
