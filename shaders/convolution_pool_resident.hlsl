// Resident graph ABI for one compact convolution/pool stage.
#define ASUINT asuint
#define ASFLOAT asfloat
#define PRECISE precise
#include "common/fp16.h"
#ifndef CONV_CHANNELS
#error CONV_CHANNELS is required
#endif
#ifndef CONV_OUTPUTS
#error CONV_OUTPUTS is required
#endif
#ifndef INPUT_E5
#define INPUT_E5 0
#endif
struct Params {
    uint width,height,input_offset,weight_offset;
    uint bias_offset,pool_offset,skip_offset,reserved;
};
[[vk::binding(0,0)]] ConstantBuffer<Params> params;
[[vk::binding(1,0)]] ByteAddressBuffer activation_source;
[[vk::binding(2,0)]] ByteAddressBuffer weight_source;
[[vk::binding(3,0)]] RWByteAddressBuffer pool_target;
[[vk::binding(4,0)]] RWByteAddressBuffer skip_target;
#define ACTIVATION_WORD(a) activation_source.Load(a)
#define WEIGHT_WORD(a) weight_source.Load(a)
#include "convolution_core.hlsli"
[numthreads(64,1,1)]
void main(uint3 id:SV_DispatchThreadID) {
    uint activation_bytes,weight_bytes,pool_bytes,skip_bytes;
    activation_source.GetDimensions(activation_bytes);
    weight_source.GetDimensions(weight_bytes);
    pool_target.GetDimensions(pool_bytes);
    skip_target.GetDimensions(skip_bytes);
    if(params.width<2 || params.height<2 || params.width>1024 || params.height>1024 ||
       (params.width&1) || (params.height&1)) return;
    uint pw=params.width/2,ph=params.height/2,words=CONV_OUTPUTS*pw*ph/2;
    uint input_size=CONV_CHANNELS*params.width*params.height*(INPUT_E5?1:2);
    uint kernel_weights=CONV_CHANNELS*CONV_OUTPUTS*KERNEL_SIZE*KERNEL_SIZE*2;
    uint output_size=words*4;
    if(params.input_offset>activation_bytes || input_size>activation_bytes-params.input_offset ||
       params.weight_offset>weight_bytes || kernel_weights>weight_bytes-params.weight_offset ||
       params.bias_offset>weight_bytes || CONV_OUTPUTS*2>weight_bytes-params.bias_offset ||
       params.pool_offset>pool_bytes || output_size>pool_bytes-params.pool_offset ||
       params.skip_offset>skip_bytes || output_size*4>skip_bytes-params.skip_offset || id.x>=words) return;
    uint element=id.x*2;
    uint c=(element/(pw*ph*8))*8+element%8;
    uint pixel=(element/8)%(pw*ph),x=pixel%pw,y=pixel/pw;
    uint pooled0=0,pooled1=0;
    [unroll] for(uint q=0;q<4;++q) {
        uint a=convolution(c,int(2*y+q/2),int(2*x+q%2),params.width,params.height,
                           CONV_CHANNELS,CONV_OUTPUTS,INPUT_E5,params.input_offset,params.weight_offset,params.bias_offset);
        uint b=convolution(c+1,int(2*y+q/2),int(2*x+q%2),params.width,params.height,
                           CONV_CHANNELS,CONV_OUTPUTS,INPUT_E5,params.input_offset,params.weight_offset,params.bias_offset);
        skip_target.Store(params.skip_offset+4*(q*words+id.x),a|(b<<16));
        pooled0=q==0?a:max_half(pooled0,a); pooled1=q==0?b:max_half(pooled1,b);
    }
    pool_target.Store(params.pool_offset+id.x*4,pooled0|(pooled1<<16));
}
