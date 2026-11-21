import mlx.core as mx

_kernel = mx.fast.metal_kernel(
    name="inkling_tiled_gemv",input_names=["query","key","value","w"],output_names=["out"],
    source="""
        uint lane=thread_index_in_simdgroup, group=thread_position_in_grid.x/32;
        uint first=group*ROWS;
        auto x = first < K0 ? query : (first < V0 ? key : value);
        float accum[ROWS]={};
        for (uint tile=0; tile<K/128; ++tile) {
            float vec[4];
            for (uint j=0; j<4; ++j) vec[j]=float(x[tile*128+lane*4+j]);
            uint base=(first/HEIGHT*K + tile*128)*HEIGHT + (first%HEIGHT)*128 + lane*4;
            for (uint r=0; r<ROWS; ++r)
                for (uint j=0; j<4; ++j)
                    accum[r]=fma(vec[j],float(w[base+r*128+j]),accum[r]);
        }
        for (uint r=0; r<ROWS; ++r) {
            float dot=accum[r];
            for (ushort n=16; n; n>>=1) dot+=simd_shuffle_down(dot,n);
            if (!lane) out[first+r]=T(dot);
        }
    """,
)


def pack(w, height):
    n,k=w.shape
    return mx.contiguous(w.reshape(n//height,height,k//128,128).transpose(0,2,1,3))


def project(layer, query, key, value, common, kmods, vmods):
    weights = tuple(m.weight for m in (*common, *kmods, *vmods))
    saved = getattr(layer, "_tiled_projections", None)
    if saved is None or not all(a is b for a,b in zip(weights, saved[0])):
        ends = []
        n = 0
        for w in weights:
            n += w.shape[0]
            ends.append(n)
        height = 8
        combined = mx.concatenate(weights)
        if n % height:
            combined = mx.pad(combined, [(0,height-n%height),(0,0)])
        saved = (weights, pack(combined,height), ends)
        object.__setattr__(layer, "_tiled_projections", saved)
    _, packed, ends = saved
    nb,kb,h,width = packed.shape
    out = _kernel(inputs=[query,key,value,packed],
                  template=[('ROWS',2),('HEIGHT',h),('K',kb*width),('T',query.dtype),
                            ('K0',ends[len(common)-1]),('V0',ends[len(common)+len(kmods)-1])],
                  grid=(nb*h//2*32,1,1),threadgroup=(128,1,1),
                  output_shapes=[(1,1,nb*h)],output_dtypes=[query.dtype])[0]
    return mx.split(out[...,:ends[-1]],ends[:-1],axis=-1)
