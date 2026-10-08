// Feed synthetic canonical bytes on stdin; emit tables and packed bytes for an independent oracle.
#include "model.h"
#include <iostream>
#include <iterator>
static void emit(uint32_t value) {
    for(unsigned shift=0;shift<32;shift+=8) std::cout.put(char((value>>shift)&255));
}
int main() {
    const std::vector<uint8_t> canonical((std::istreambuf_iterator<char>(std::cin)), {});
    if(canonical.size()!=bcm::canonical_bytes()) return 1;
    for(auto net : {bcm::NetNCHW8,bcm::NetNHWC}) {
        std::vector<uint8_t> packed;
        if(!bcm::build_weight_blob(net,canonical.data(),canonical.size(),packed)) return 2;
        const auto& t=bcm::net_tables(net);
        emit(t.weightCount); emit(t.weightBlobBytes);
        for(unsigned i=0;i<t.weightCount;++i) {
            const auto& e=t.weights[i];
            for(auto v : {e.blobOffset,e.size,e.canonOffset,e.O,e.I,e.k,e.layout}) emit(v);
        }
        std::cout.write(reinterpret_cast<const char*>(packed.data()), packed.size());
    }
}
