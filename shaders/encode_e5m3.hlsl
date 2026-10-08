#define ASUINT asuint
#define ASFLOAT asfloat
#define PRECISE precise
#include "common/fp16.h"
[[vk::binding(0,0)]] ByteAddressBuffer source;
[[vk::binding(1,0)]] RWByteAddressBuffer target;
[numthreads(64,1,1)]
void main(uint3 id:SV_DispatchThreadID) {
    uint bytes; source.GetDimensions(bytes);
    if(id.x>=bytes/8) return;
    uint a=source.Load(id.x*8),b=source.Load(id.x*8+4);
    uint packed=fp_to_e5m3(a&65535u)|(fp_to_e5m3(a>>16)<<8)|
                (fp_to_e5m3(b&65535u)<<16)|(fp_to_e5m3(b>>16)<<24);
    target.Store(id.x*4,packed);
}
