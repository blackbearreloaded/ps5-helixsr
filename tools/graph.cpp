// Offline graph contract exporter. No GPU or model weights required.
#include "model.h"
#include <charconv>
#include <iostream>
#include <stdexcept>
#include <string>
#include <set>
using namespace bcm;
static void require(bool ok, const char* reason) { if (!ok) throw std::runtime_error(reason); }
static uint32_t number(const char* text) {
    uint32_t value = 0;
    const auto end = text + std::strlen(text);
    const auto result = std::from_chars(text, end, value);
    require(result.ec == std::errc{} && result.ptr == end, "invalid unsigned integer");
    return value;
}
static double real(const char* text) {
    double value = 0;
    const auto end = text + std::strlen(text);
    const auto result = std::from_chars(text, end, value);
    require(result.ec == std::errc{} && result.ptr == end, "invalid floating-point value");
    return value;
}
static void quote(const char* text) {
    std::cout << '"';
    for (; *text; ++text) {
        require(static_cast<unsigned char>(*text) >= 32, "control byte in name");
        if (*text == '"' || *text == '\\') std::cout << '\\';
        std::cout << *text;
    }
    std::cout << '"';
}
static void xyz(const uint32_t* v) { std::cout << '[' << v[0] << ',' << v[1] << ',' << v[2] << ']'; }
int main(int argc, char** argv) {
    try {
        require(argc == 7 || argc == 14, "usage: helixsr_graph output_width output_height render_width render_height frame auto_exposure(0|1) [jx jy pjx pjy pre_exposure reset has_previous]");
        Params p;
        p.Wo = number(argv[1]); p.Ho = number(argv[2]); p.Wr = number(argv[3]); p.Hr = number(argv[4]);
        for (auto d : {p.Wo,p.Ho,p.Wr,p.Hr}) require(d > 0 && d <= 4096, "dimension outside offline contract [1,4096]");
        require(p.Wr <= p.Wo && p.Hr <= p.Ho, "downsampling unsupported");
        const auto frame = number(argv[5]), exposure = number(argv[6]);
        require(exposure <= 1, "auto exposure must be 0 or 1");
        p.hasPrev = frame != 0; p.reset = frame == 0;
        if (argc == 14) {
            p.jx = static_cast<float>(real(argv[7])); p.jy = static_cast<float>(real(argv[8]));
            p.pjx = static_cast<float>(real(argv[9])); p.pjy = static_cast<float>(real(argv[10]));
            p.pre = real(argv[11]); p.reset = number(argv[12]); p.hasPrev = number(argv[13]) != 0;
            require(p.pre > 0 && p.reset <= 1 && number(argv[13]) <= 1, "invalid dynamic frame parameters");
        }
        finalize(p);
        PlanOptions options;
        options.autoExposure = exposure; options.displayResMv = false;
        auto launches = plan(p, frame, options);
        const auto& t = net_tables(p.net);
        unsigned outputs = 0;
        for (const auto& l : launches) {
            require(l.spec && l.name, "missing kernel specification");
            require(l.grid[0] && l.grid[1] && l.grid[2], "empty dispatch");
            require(uint64_t(l.block[0])*l.block[1]*l.block[2] <= 1024, "oversize workgroup");
            outputs += std::strstr(l.name, "aniso_gaussian") != nullptr;
            std::set<unsigned> args;
            for (const auto& b : l.bindings) {
                require(size_t(b.argOffset)+8 <= l.args.size(), "binding exceeds arguments");
                require(args.insert(b.argOffset).second, "duplicate argument binding");
                // The published output-head table marks its legacy argument 256 unused.
                // Preserve this evidence; a backend must verify it before pruning a binding.
                require(b.access != AccessNone || (b.kind == BindBuffer && b.role == 4 &&
                    b.argOffset == 256 && std::strstr(l.name, "aniso_gaussian")), "unknown access intent");
                if (b.kind == BindBuffer) require(b.role < t.bufferCount && b.offset < t.buffers[b.role].bytes(p), "buffer offset outside resource");
                else if (b.kind == BindWeight) require(b.offset <= t.weightBlobBytes && b.weightSize <= t.weightBlobBytes-b.offset, "weight range outside resource");
                else require((b.kind == BindTexture || b.kind == BindSurface) && b.role < t.imageCount, "invalid image binding");
            }
        }
        require(outputs == 1, "missing or duplicate output stage");
        std::cout << "{\"schema\":1,\"network\":\"NCHW8\",\"frame\":" << frame << ",\"reset\":" << p.reset
          << ",\"output\":[" << p.Wo << ',' << p.Ho << "],\"render\":[" << p.Wr << ',' << p.Hr
          << "],\"padded\":[" << p.Wp << ',' << p.Hp << "],\"weight_bytes\":" << t.weightBlobBytes << ",\"buffers\":[";
        for (unsigned i=0;i<t.bufferCount;++i) {
            if(i) std::cout << ',';
            std::cout << "{\"name\":"; quote(t.buffers[i].name);
            std::cout << ",\"bytes\":" << t.buffers[i].bytes(p) << '}';
        }
        std::cout << "],\"images\":[";
        for (unsigned i=0;i<t.imageCount;++i) {
            if(i) std::cout << ',';
            const auto& r=t.images[i];
            unsigned w=1,h=1;
            if(r.size==SizeRender) {w=p.Wr;h=p.Hr;}
            if(r.size==SizeOutput || r.size==SizeOutputEvenH) {w=p.Wo;h=p.Ho;}
            if(r.size==SizeOutputEvenH) h=(h+1)&~1u;
            // Upstream image tables describe the captured mvhi configuration.
            if(!options.displayResMv && std::strcmp(r.name,"GAME_MV")==0) {w=p.Wr;h=p.Hr;}
            std::cout << "{\"name\":"; quote(r.name);
            std::cout << ",\"dxgi_format\":" << r.format << ",\"size\":[" << w << ',' << h
              << "],\"game\":" << (r.game?"true":"false") << ",\"persistent\":" << (r.persistent?"true":"false") << '}';
        }
        std::cout << "],\"launches\":[";
        bool first=true;
        for(const auto& l:launches) {
            if(!first) std::cout << ','; first=false;
            std::cout << "{\"kernel\":"; quote(l.name);
            std::cout << ",\"occurrence\":" << l.occurrence << ",\"grid\":"; xyz(l.grid);
            std::cout << ",\"block\":"; xyz(l.block);
            std::cout << ",\"args_hex\":\"";
            constexpr char hex[]="0123456789abcdef";
            for(auto b:l.args) std::cout << hex[b>>4] << hex[b&15];
            std::cout << "\",\"bindings\":[";
            bool firstBinding=true;
            for(const auto& b:l.bindings) {
                if(!firstBinding) std::cout << ','; firstBinding=false;
                std::cout << "{\"arg_offset\":" << b.argOffset << ",\"kind\":" << unsigned(b.kind)
                  << ",\"role\":" << unsigned(b.role) << ",\"access\":" << unsigned(b.access)
                  << ",\"offset\":" << b.offset << ",\"weight_size\":" << b.weightSize << '}';
            }
            std::cout << "]}";
        }
        std::cout << "]}\n";
        return 0;
    } catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
