// Offset-aware resident FP16 to E5M3 conversion.
#define ASUINT asuint
#define ASFLOAT asfloat
#define PRECISE precise
#include "common/fp16.h"
struct Params { uint source_offset,target_offset,scalar_count,reserved; };
[[vk::binding(0,0)]] ConstantBuffer<Params> params;
[[vk::binding(1,0)]] ByteAddressBuffer source;
[[vk::binding(2,0)]] RWByteAddressBuffer target;
[numthreads(64,1,1)]
void main(uint3 id:SV_DispatchThreadID) {
    uint source_bytes,target_bytes; source.GetDimensions(source_bytes); target.GetDimensions(target_bytes);
    if(!params.scalar_count || (params.scalar_count&3) || params.source_offset>source_bytes ||
       params.scalar_count>(source_bytes-params.source_offset)/2 || params.target_offset>target_bytes ||
       params.scalar_count>target_bytes-params.target_offset) return;
    uint words=params.scalar_count/4;
    if(id.x>=words) return;
    uint a=source.Load(params.source_offset+id.x*8),b=source.Load(params.source_offset+id.x*8+4);
    uint packed=fp_to_e5m3(a&65535u)|(fp_to_e5m3(a>>16)<<8)|
                (fp_to_e5m3(b&65535u)<<16)|(fp_to_e5m3(b>>16)<<24);
    target.Store(params.target_offset+id.x*4,packed);
}
