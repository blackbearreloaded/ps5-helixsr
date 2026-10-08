// Copyright 2026 lonewolf0622. Licensed under the Apache License, Version 2.0.
//
// Model E host logic: what NGX's DLSS feature launches every frame, with which arguments, buffers and images.
// The tables come from tools/gen_model.py (generated from the host-model reconstruction, which regenerates every
// captured argument block byte for byte); the planner here follows its launch rules. Platform independent: used by
// the DLL (D3D12) and by the native plan tests.
#pragma once
#include <cstdint>
#include <cstring>
#include <vector>

#ifndef DXGI_FORMAT_DEFINED
// Minimal subset so the model compiles without the Windows headers (native tests).
enum DXGI_FORMAT
{
    DXGI_FORMAT_UNKNOWN = 0,
    DXGI_FORMAT_R16G16B16A16_FLOAT = 10,
    DXGI_FORMAT_R16G16_FLOAT = 34,
    DXGI_FORMAT_R32_FLOAT = 41,
    DXGI_FORMAT_R16_FLOAT = 54,
};
#define DXGI_FORMAT_DEFINED 1
#endif

namespace bcm
{
using u64 = uint64_t;

enum Net : uint8_t { NetNCHW8 = 0, NetNHWC = 1 };

// Frame parameters (design A, host_plan.py P).
struct Params
{
    Net      net = NetNCHW8;
    uint32_t Wo = 0, Ho = 0;        // output size
    uint32_t Wr = 0, Hr = 0;        // render subrect size
    uint32_t Wp = 0, Hp = 0;        // output size padded to 128
    float    jx = 0, jy = 0;        // jitter (render pixels, as the game passed it)
    float    pjx = 0, pjy = 0;      // previous frame's jitter
    bool     hasPrev = false;       // pjx/pjy valid
    double   pre = 1.0;             // pre-exposure
    uint32_t reset = 0;             // game reset flag
};

void finalize(Params& P);                       // computes Wp/Hp
Net  network_for(uint32_t Wo, uint32_t Ho, uint32_t Wr, uint32_t Hr);   // ratio >= 2.5 -> NHWC

// ---------------------------------------------------------------------------------------------------- tables
struct BufferRole
{
    const char* name;
    u64 (*bytes)(const Params&);
};

enum ImageSize : uint8_t { SizeOne, SizeRender, SizeOutput, SizeOutputEvenH };
struct ImageRole
{
    const char* name;
    DXGI_FORMAT format;
    ImageSize   size;
    bool        game;        // supplied by the game (color, motion vectors, depth)
    bool        persistent;  // carried to the next frame
};

// One weight pointer of the NGX upload layout, regenerated from the canonical blob.
// layout: 0 = OHWI copy, 1 = [I/8][H][W][O][i8], 2 = [H][W][I/8][O][i8], 3 = bias copy
struct WeightEntry
{
    uint32_t blobOffset, size, canonOffset, O, I, k, layout;
};

enum BindKind : uint8_t { BindNone, BindBuffer, BindTexture, BindSurface, BindWeight };
enum Access : uint8_t { AccessNone = 0, AccessRead = 1, AccessWrite = 2, AccessReadWrite = 3 };   // as observed per launch
struct Binding
{
    uint16_t argOffset;
    BindKind kind;
    uint8_t  role;                         // buffer role (BindBuffer) or image role (BindTexture/BindSurface)
    Access   access;
    u64    (*offset)(const Params&);       // byte offset into the buffer role (nullptr = 0)
    uint32_t weightOffset = 0;             // BindWeight: byte offset into the network's NGX-layout weight blob
    uint32_t weightSize = 0;
};

struct KernelSpec
{
    Net         net;
    const char* name;
    uint32_t    occurrence;                // n-th launch of this kernel name within one frame
    uint32_t    argsSize;
    uint32_t    block[3];
    void      (*args)(const Params&, uint8_t*);
    void      (*grid)(const Params&, uint32_t[3]);
    const Binding* bindings;
    uint32_t    bindingCount;
};

struct NetTables
{
    const BufferRole*  buffers;
    uint32_t           bufferCount;
    const ImageRole*   images;
    uint32_t           imageCount;
    const WeightEntry* weights;
    uint32_t           weightCount;
    uint32_t           weightBlobBytes;
};

const NetTables&  net_tables(Net n);
const KernelSpec* find_spec(Net n, const char* name, uint32_t occurrence);
int               buffer_index(Net n, const char* name);     // -1 if missing
int               image_index(Net n, const char* name);      // -1 if missing
uint32_t          spec_count();
const KernelSpec& spec_at(uint32_t i);

// ---------------------------------------------------------------------------------------------------- planner
struct LaunchBinding
{
    uint16_t argOffset;
    BindKind kind;
    uint8_t  role;
    Access   access;
    u64      offset;          // buffer byte offset / weight blob offset
    uint32_t weightSize;
};

struct Launch
{
    const KernelSpec*          spec = nullptr;
    const char*                name = nullptr;      // actual kernel (engine variants may differ from the spec name)
    uint32_t                   occurrence = 0;
    uint32_t                   grid[3] = {};
    uint32_t                   block[3] = {};
    std::vector<uint8_t>       args;                // NGX argument block (pointers/handles left as template bytes)
    std::vector<LaunchBinding> bindings;
};

struct PlanOptions
{
    bool     hdr = true;                  // HDR (linear) color input -> rel_hdr engine variant, else rel_ldr
    bool     displayResMv = true;         // display-resolution motion vectors -> *_mvhi, else *_mvlo
    bool     depthInverted = true;        // input kernel word 184 (NGX: DepthInverted flag)
    bool     autoExposure = true;         // NGX AutoExposure: the luma/reduce chain computes EXPOSURE_CUR (HDR only)
    bool     reactiveMask = false;        // GAME_REACTIVE bound: input kernel argument 336 non-null (the kernel reads it)
};

// Launch list of one evaluate: n = evaluate index since feature creation (0 = creation frame).
std::vector<Launch> plan(const Params& P, uint32_t n, const PlanOptions& opt = {});

// Regenerates the network's NGX-layout weight blob (k_weight_blob_bytes) from the canonical blob.
bool build_weight_blob(Net n, const uint8_t* canon, size_t canonBytes, std::vector<uint8_t>& out);
uint64_t canonical_bytes();

// Engine-input kernel name for the input flags.
const char* engine_kernel_name(bool hdr, bool displayResMv);
}  // namespace bcm
