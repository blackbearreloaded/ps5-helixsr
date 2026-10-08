// Compact baseline for the three standalone NCHW8 3x3 convolution/pool stages.
// Arithmetic identity: orig lane-rotated dot16, column-major taps, half bias/ReLU.
#define ASUINT asuint
#define ASFLOAT asfloat
#define PRECISE precise
#include "common/fp16.h"
#ifndef KERNEL_SIZE
#define KERNEL_SIZE 3
#endif
[[vk::binding(0,0)]] ByteAddressBuffer source;
[[vk::binding(1,0)]] RWByteAddressBuffer target;
#define ACTIVATION_WORD(a) source.Load(a)
#define WEIGHT_WORD(a) source.Load(a)
#include "convolution_core.hlsli"
[numthreads(64,1,1)]
void main(uint3 id:SV_DispatchThreadID) {
    uint width=source.Load(0),height=source.Load(4),channels=source.Load(8),outputs=source.Load(12);
    uint e5=source.Load(16),input_offset=source.Load(20),weight_offset=source.Load(24),bias_offset=source.Load(28);
    uint pw=width/2,ph=height/2,words=outputs*pw*ph/2;
    if(id.x>=words) return;
    uint element=id.x*2;
    uint c=(element/(pw*ph*8))*8+element%8;
    uint pixel=(element/8)%(pw*ph),x=pixel%pw,y=pixel/pw;
    uint pooled0=0,pooled1=0;
    [loop] for(uint q=0;q<4;++q) {
        uint a=convolution(c,int(2*y+q/2),int(2*x+q%2),width,height,channels,outputs,e5,input_offset,weight_offset,bias_offset);
        uint b=convolution(c+1,int(2*y+q/2),int(2*x+q%2),width,height,channels,outputs,e5,input_offset,weight_offset,bias_offset);
        target.Store(4*((q+1)*words+id.x),a|(b<<16));
        pooled0=q==0?a:max_half(pooled0,a); pooled1=q==0?b:max_half(pooled1,b);
    }
    target.Store(id.x*4,pooled0|(pooled1<<16));
}
