#define ASUINT asuint
#define ASFLOAT asfloat
#define PRECISE precise
#include "common/fp16.h"
[[vk::binding(0,0)]] StructuredBuffer<uint> input_words;
[[vk::binding(1,0)]] RWStructuredBuffer<uint> output_words;
[numthreads(64,1,1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint count,stride; input_words.GetDimensions(count,stride);
    if(id.x>=count) return;
    uint h=input_words[id.x]&65535u;
    output_words[4*id.x+0]=asuint(fp_h2f(h));
    output_words[4*id.x+1]=fp_rne16(fp_h2f(h));
    output_words[4*id.x+2]=fp_to_e5m3(h);
    output_words[4*id.x+3]=asuint(fp_e5m3(h&255u));
}
