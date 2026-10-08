// Copyright 2026 lonewolf0622. Licensed under the Apache License, Version 2.0.
#include "model.h"
#include <cstring>

namespace bcm
{
namespace
{
inline u64 cdiv(u64 a, u64 k) { return (a + k - 1) / k; }
inline uint32_t u32(u64 v) { return (uint32_t)v; }
inline float f32d(double x) { return (float)x; }
inline uint32_t fbits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
inline void put32(uint8_t* a, uint32_t off, uint32_t v) { memcpy(a + off, &v, 4); }

// NGX's unsigned division magic (host_model.py magic_of): s = max(0, ceil(log2 d) - 1), m = ceil(2^(32+s) / d).
inline uint32_t ceil_log2(u64 d)
{
    uint32_t r = 0;
    while ((1ull << r) < d) ++r;
    return r;
}
inline uint32_t magic_s(u64 d)
{
    const uint32_t c = ceil_log2(d);
    return c > 0 ? c - 1 : 0;
}
inline uint32_t magic_m(u64 d)
{
    const uint32_t s = magic_s(d);
    const unsigned __int128 num = (unsigned __int128)1 << (32 + s);
    return (uint32_t)((num + d - 1) / d);
}

// Luma (auto-exposure) image, per axis: half the output size when that is at most the render size (scale ratio up to
// 2), else half the render size; at least 256 (NGX: 640x360 render -> 320x256, 1542x868 -> 3840x2160 -> 771x434).
inline uint32_t luma_dim(uint32_t out, uint32_t render)
{
    const uint32_t d = out / 2 <= render ? out / 2 : render / 2;
    return d > 256 ? d : 256;
}
inline uint32_t luma_w(const Params& P) { return luma_dim(P.Wo, P.Wr); }
inline uint32_t luma_h(const Params& P) { return luma_dim(P.Ho, P.Hr); }
inline u64 luma_n(const Params& P) { return (u64)luma_w(P) * luma_h(P); }
inline u64 reduce_n(const Params& P, int k)
{
    u64 n = luma_n(P);
    for (int i = 0; i < k; ++i)
        n = cdiv(n, 512);
    return n;
}
// One float per luma pixel; the reduce passes ping-pong between two of these.
// Allocation: the largest luma image any render size can give at this output (half the output, at least 256 per axis),
// so dynamic resolution never outgrows the buffers created with the first frame's render size.
inline u64 luma_bytes(const Params& P)
{
    const u64 w = P.Wo / 2 > 256 ? P.Wo / 2 : 256, h = P.Ho / 2 > 256 ? P.Ho / 2 : 256;
    return 4ull * w * h;
}

#include "model_gen.inc"

int reduce_count(const Params& P)
{
    u64 n = luma_n(P);
    int k = 0;
    for (;;)
    {
        ++k;
        n = cdiv(n, 512);
        if (n <= 1)
            return k;
    }
}

double output_ratio_min(const Params& P)
{
    const double rw = P.Wr ? (double)P.Wo / P.Wr : 1.0, rh = P.Hr ? (double)P.Ho / P.Hr : 1.0;
    return rw < rh ? rw : rh;
}

LaunchBinding to_launch_binding(const Binding& b, const Params& P)
{
    LaunchBinding l{};
    l.argOffset = b.argOffset;
    l.kind = b.kind;
    l.role = b.role;
    l.access = b.access;
    if (b.kind == BindWeight)
    {
        l.offset = b.weightOffset;
        l.weightSize = b.weightSize;
    }
    else
        l.offset = b.offset ? b.offset(P) : 0;
    return l;
}
}  // namespace

void finalize(Params& P)
{
    P.Wp = (P.Wo + 127) / 128 * 128;
    P.Hp = (P.Ho + 127) / 128 * 128;
}

Net network_for(uint32_t Wo, uint32_t Ho, uint32_t Wr, uint32_t Hr)
{
    // NGX selects the Ultra Performance (NHWC) network for a width scale ratio above 2.5 (NGX argument oracle: 2.5
    // exactly and 2.4x3.0 stay on the main network, 3.0x2.4 switches; independent of the quality mode).
    return Wr && (double)Wo / Wr > 2.5 ? NetNHWC : NetNCHW8;
}

const NetTables& net_tables(Net n) { return k_nets[n]; }
uint32_t spec_count() { return (uint32_t)(sizeof(k_specs) / sizeof(k_specs[0])); }
const KernelSpec& spec_at(uint32_t i) { return k_specs[i]; }
uint64_t canonical_bytes() { return BCM_CANON_BYTES; }

const KernelSpec* find_spec(Net n, const char* name, uint32_t occurrence)
{
    for (const KernelSpec& s : k_specs)
        if (s.net == n && s.occurrence == occurrence && !strcmp(s.name, name))
            return &s;
    return nullptr;
}

int buffer_index(Net n, const char* name)
{
    const NetTables& t = k_nets[n];
    for (uint32_t i = 0; i < t.bufferCount; ++i)
        if (!strcmp(t.buffers[i].name, name))
            return (int)i;
    return -1;
}

int image_index(Net n, const char* name)
{
    const NetTables& t = k_nets[n];
    for (uint32_t i = 0; i < t.imageCount; ++i)
        if (!strcmp(t.images[i].name, name))
            return (int)i;
    return -1;
}

const char* engine_kernel_name(bool hdr, bool displayResMv)
{
    if (hdr)
        return displayResMv ? "cuda_engine_input_kernel_rel_hdr_mvdiff_mvhi" : "cuda_engine_input_kernel_rel_hdr_mvdiff_mvlo";
    return displayResMv ? "cuda_engine_input_kernel_rel_ldr_mvdiff_mvhi" : "cuda_engine_input_kernel_rel_ldr_mvdiff_mvlo";
}

std::vector<Launch> plan(const Params& P, uint32_t n, const PlanOptions& opt)
{
    std::vector<Launch> out;
    auto push = [&](const char* name, uint32_t occ, const char* actual) -> Launch* {
        const KernelSpec* s = find_spec(P.net, name, occ);
        if (!s)
            return nullptr;
        Launch l;
        l.spec = s;
        l.name = actual ? actual : s->name;
        l.occurrence = occ;
        s->grid(P, l.grid);
        memcpy(l.block, s->block, sizeof(l.block));
        l.args.resize(s->argsSize);
        s->args(P, l.args.data());
        for (uint32_t i = 0; i < s->bindingCount; ++i)
            if (s->bindings[i].kind != BindNone)
                l.bindings.push_back(to_launch_binding(s->bindings[i], P));
        out.push_back(std::move(l));
        return &out.back();
    };

    // Auto-exposure refresh (HDR with NGX's AutoExposure only; NGX argument oracle: LDR, or an exposure texture from
    // the game, skips the chain and the network reads that texture as EXPOSURE_CUR).
    if (n % 2 == 0 && opt.hdr && opt.autoExposure)
    {
        // Auto-exposure refresh: luma -> reduce (until one block) -> auto_exposure_copy.
        push("cuda_luma_convert_kernel", 0, nullptr);
        const int k = reduce_count(P);
        const int l0 = buffer_index(P.net, "LUMA0"), l1 = buffer_index(P.net, "LUMA1");
        const KernelSpec* r0 = find_spec(P.net, "cuda_reduce_sum_kernel", 0);
        for (int i = 0; i < k && r0; ++i)
        {
            // Captured reduce launches have their own spec (#0..#2); deeper ones reuse #0 with their own count/grid.
            const KernelSpec* ri = find_spec(P.net, "cuda_reduce_sum_kernel", (uint32_t)i);
            Launch l;
            l.spec = ri ? ri : r0;
            l.name = r0->name;
            l.occurrence = (uint32_t)i;
            memcpy(l.block, l.spec->block, sizeof(l.block));
            l.args.resize(l.spec->argsSize);
            l.spec->args(P, l.args.data());
            // one block per 512 inputs: derived for every pass (the captured grids saw one size per network)
            l.grid[0] = (uint32_t)reduce_n(P, i + 1);
            l.grid[1] = l.grid[2] = 1;
            put32(l.args.data(), 0, (uint32_t)reduce_n(P, i));
            l.bindings.push_back({8, BindBuffer, (uint8_t)(i % 2 ? l1 : l0), AccessRead, 0, 0});
            l.bindings.push_back({16, BindBuffer, (uint8_t)(i % 2 ? l0 : l1), AccessWrite, 0, 0});
            out.push_back(std::move(l));
        }
        if (Launch* a = push("cuda_auto_exposure_copy_kernel", 0, nullptr))
            for (LaunchBinding& b : a->bindings)
                if (b.kind == BindBuffer)
                    b.role = (uint8_t)(k % 2 ? l1 : l0);   // reads the last reduce output
    }
    if (n == 0)
        push("cuda_clear_buffer_kernel", 0, nullptr);
    // input kernel: the motion-vector resolution selects the argument layout (mvhi / mvlo, NGX argument oracle); the
    // LDR variants take the HDR variants' arguments
    if (Launch* e = push(opt.displayResMv ? "cuda_engine_input_kernel_rel_hdr_mvdiff_mvhi"
                                          : "cuda_engine_input_kernel_rel_hdr_mvdiff_mvlo",
                         0, engine_kernel_name(opt.hdr, opt.displayResMv)))
    {
        put32(e->args.data(), 108, opt.hdr ? 1u : 0u);              // NGX argument oracle: IsHDR
        put32(e->args.data(), 184, opt.depthInverted ? 1u : 0u);    // NGX argument oracle: DepthInverted
        // bias-current-colour (reactive) mask: the kernel reads texture argument 336 only when it is non-null, at render
        // resolution from the subrect origin in words 236/240 (0, 0)
        put32(e->args.data(), 336, opt.reactiveMask ? 1u : 0u);
        put32(e->args.data(), 340, 0u);
    }

    // Network layers in the captured order, then the output kernel, then copy_exposure.
    // Main network output kernel by the smaller of the two scale ratios (NGX argument oracle, both axes, any output
    // size): >= 1.7 dfn_fastpath, >= 1.5 dfn_mid, below (incl. native / DLAA) dfn_3x3_256.
    const char* outName = nullptr;
    const double rmin = output_ratio_min(P);
    if (P.net == NetNHWC)
        outName = "dltss_nhwc_bilinear_upsample_conv_1x1_conv_1x1_aniso_gaussian_dfn_fastpath_5x5_128_032_032_048_032_e5m3_fp16_kernel";
    else if (rmin >= 1.7)
        outName = "dltss_nchw8_bilinear_upsample_conv_1x1_conv_1x1_aniso_gaussian_dfn_fastpath_3x3_128_032_032_048_032_e5m3_fp16_kernel";
    else if (rmin >= 1.5)
        outName = "dltss_nchw8_bilinear_upsample_conv_1x1_conv_1x1_aniso_gaussian_dfn_mid_3x3_128_032_032_048_032_e5m3_fp16_kernel";
    else
        outName = "dltss_nchw8_bilinear_upsample_conv_1x1_conv_1x1_aniso_gaussian_dfn_3x3_256_032_032_048_032_e5m3_fp16_kernel";
    for (uint32_t i = 0; i < spec_count(); ++i)
    {
        const KernelSpec& s = k_specs[i];
        if (s.net != P.net || strncmp(s.name, "dltss_", 6) || strstr(s.name, "aniso_gaussian"))
            continue;
        push(s.name, s.occurrence, nullptr);
    }
    if (Launch* o = push(outName, 0, nullptr))
        put32(o->args.data(), 512, opt.hdr ? 1u : 0u);              // NGX argument oracle: IsHDR
    push("cuda_copy_exposure_kernel", 0, nullptr);
    return out;
}

bool build_weight_blob(Net n, const uint8_t* canon, size_t canonBytes, std::vector<uint8_t>& out)
{
    const NetTables& t = k_nets[n];
    out.assign(t.weightBlobBytes, 0);
    for (uint32_t i = 0; i < t.weightCount; ++i)
    {
        const WeightEntry& e = t.weights[i];
        if ((u64)e.canonOffset + e.size > canonBytes || (u64)e.blobOffset + e.size > out.size())
            return false;
        const uint16_t* src = reinterpret_cast<const uint16_t*>(canon + e.canonOffset);   // OHWI fp16
        uint16_t* dst = reinterpret_cast<uint16_t*>(out.data() + e.blobOffset);
        const uint32_t O = e.O, I = e.I, K = e.k;
        switch (e.layout)
        {
        case 0:
        case 3:
            memcpy(dst, src, e.size);
            break;
        case 1:   // [I/8][H][W][O][i8]
            for (uint32_t o = 0; o < O; ++o)
                for (uint32_t h = 0; h < K; ++h)
                    for (uint32_t w = 0; w < K; ++w)
                        for (uint32_t c = 0; c < I; ++c)
                            dst[((((c / 8) * K + h) * K + w) * O + o) * 8 + (c % 8)] = src[((o * K + h) * K + w) * I + c];
            break;
        case 2:   // [H][W][I/8][O][i8]
            for (uint32_t o = 0; o < O; ++o)
                for (uint32_t h = 0; h < K; ++h)
                    for (uint32_t w = 0; w < K; ++w)
                        for (uint32_t c = 0; c < I; ++c)
                            dst[(((h * K + w) * (I / 8) + c / 8) * O + o) * 8 + (c % 8)] = src[((o * K + h) * K + w) * I + c];
            break;
        default:
            return false;
        }
    }
    return true;
}
}  // namespace bcm
