// Exhaustive bit-pattern probe of published portable arithmetic helpers.
#include <cstdint>
#include <cstring>
#include <iostream>
using uint = uint32_t;
static uint bits(float f) { uint u; std::memcpy(&u,&f,4); return u; }
static float value(uint u) { float f; std::memcpy(&f,&u,4); return f; }
#define ASUINT bits
#define ASFLOAT value
#define PRECISE
#include "common/fp16.h"
static void emit(uint v) {
    for(unsigned shift=0;shift<32;shift+=8) std::cout.put(char((v>>shift)&255));
}
int main() {
    for(uint h=0;h<65536;++h) {
        emit(bits(fp_h2f(h)));
        emit(fp_rne16(fp_h2f(h)));
        emit(fp_to_e5m3(h));
    }
    for(uint b=0;b<256;++b) emit(bits(fp_e5m3(b)));
    // Every positive finite adjacent-half midpoint and both adjacent float32 values.
    for(uint h=0;h<0x7bff;++h) {
        const float midpoint=(fp_h2f(h)+fp_h2f(h+1))*0.5f;
        for(int delta : {-1,0,1}) {
            float f=value(bits(midpoint)+delta);
            emit(fp_rne16(f)); emit(fp_rne16(-f));
        }
    }
}
