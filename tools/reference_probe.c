/* SPDX-License-Identifier: MIT
 * Headless D3D12 runner for a FidelityFX 3.1 upscaler DLL (the original HelixSR).
 * Reads per-frame colour, depth and motion files, writes every output frame.
 * Derived from the BC250 provider probe (MIT). Runs on WARP unless HXP_HARDWARE=1. */
#define COBJMACROS
#define INITGUID
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <ffx_upscale.h>

__declspec(dllexport) const UINT D3D12SDKVersion = 619;
__declspec(dllexport) const char *D3D12SDKPath = ".\\D3D12\\";

int _fltused;
static HANDLE output;
static UINT render_w, render_h, output_w, output_h, frame_count;
static wchar_t directory[2048];

struct FrameInfo { float jitter_x, jitter_y; UINT reset; float pre_exposure; };

static void log_line(const char *text)
{
    DWORD n = 0;
    WriteFile(output, text, (DWORD)strlen(text), &n, NULL);
}

static void fail(const char *what, unsigned long code)
{
    char line[512];
    snprintf(line, sizeof(line), "FAILED %s: %08lx\n", what, code);
    log_line(line);
    ExitProcess(10);
}

#define HR(call) do { HRESULT hr_ = (call); if (FAILED(hr_)) fail(#call, (unsigned long)hr_); } while (0)

static void message(uint32_t kind, const wchar_t *text)
{
    char line[1900], msg[1500];
    WideCharToMultiByte(CP_UTF8, 0, text, -1, msg, sizeof(msg), NULL, NULL);
    snprintf(line, sizeof(line), "FFX message %u: %s\n", kind, msg);
    log_line(line);
}

static UINT get_uint(const char *key, UINT fallback, UINT maximum)
{
    char value[32];
    DWORD n = GetEnvironmentVariableA(key, value, sizeof(value));
    if (!n) return fallback;
    if (n >= sizeof(value)) ExitProcess(20);
    char *end = NULL;
    unsigned long parsed = strtoul(value, &end, 10);
    if (!end || *end || parsed > maximum) ExitProcess(20);
    return (UINT)parsed;
}

static HANDLE open_file(const wchar_t *format, UINT frame, bool write)
{
    static wchar_t path[4096];
    _snwprintf(path, 4096, format, directory, frame);
    HANDLE file = CreateFileW(path, write ? GENERIC_WRITE : GENERIC_READ, FILE_SHARE_READ, NULL,
                              write ? CREATE_ALWAYS : OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (file == INVALID_HANDLE_VALUE) {
        char name[600];
        WideCharToMultiByte(CP_UTF8, 0, path, -1, name, sizeof(name), NULL, NULL);
        log_line(name);
        fail(" open", GetLastError());
    }
    return file;
}

/* ---- Stage dumps: every resource the DLL creates, copied out after each of its dispatches. ---- */
#define MAX_TRACKED 128
#define MAX_STEPS 64
struct Tracked {
    ID3D12Resource *resource;
    D3D12_RESOURCE_DESC desc;
    D3D12_HEAP_TYPE heap;
    D3D12_RESOURCE_STATES state;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint;
    UINT rows;
    UINT64 row_bytes, bytes, offset;
};
static struct Tracked tracked[MAX_TRACKED];
static UINT tracked_count, current_frame, current_step, dump_frame = ~0u, dump_steps;
static bool in_dll;
static UINT64 dump_bytes;
static ID3D12Device *the_device;
static ID3D12Resource *dump_buffers[MAX_STEPS];
static HRESULT (STDMETHODCALLTYPE *original_create)(ID3D12Device *, const D3D12_HEAP_PROPERTIES *, D3D12_HEAP_FLAGS,
    const D3D12_RESOURCE_DESC *, D3D12_RESOURCE_STATES, const D3D12_CLEAR_VALUE *, REFIID, void **);
static void (STDMETHODCALLTYPE *original_dispatch)(ID3D12GraphicsCommandList *, UINT, UINT, UINT);
static void (STDMETHODCALLTYPE *original_barrier)(ID3D12GraphicsCommandList *, UINT, const D3D12_RESOURCE_BARRIER *);

static void replace_pointer(void *slot, void *pointer)
{
    DWORD old = 0, unused = 0;
    if (!VirtualProtect(slot, sizeof(pointer), PAGE_READWRITE, &old)) ExitProcess(22);
    memcpy(slot, &pointer, sizeof(pointer));
    if (!VirtualProtect(slot, sizeof(pointer), old, &unused)) ExitProcess(22);
}

static HRESULT STDMETHODCALLTYPE hooked_create(ID3D12Device *dev, const D3D12_HEAP_PROPERTIES *heap, D3D12_HEAP_FLAGS flags,
    const D3D12_RESOURCE_DESC *desc, D3D12_RESOURCE_STATES state, const D3D12_CLEAR_VALUE *clear, REFIID iid, void **out)
{
    HRESULT hr = original_create(dev, heap, flags, desc, state, clear, iid, out);
    if (in_dll && SUCCEEDED(hr) && out && *out && tracked_count < MAX_TRACKED) {
        struct Tracked *t = &tracked[tracked_count];
        t->resource = *out;
        t->desc = *desc;
        t->heap = heap->Type;
        t->state = state;
        UINT64 total = 0;
        ID3D12Device_GetCopyableFootprints(dev, desc, 0, 1, 0, &t->footprint, &t->rows, &t->row_bytes, &total);
        t->bytes = desc->Dimension == D3D12_RESOURCE_DIMENSION_BUFFER ? desc->Width : total;
        char line[256];
        snprintf(line, sizeof(line), "resource: index=%u dimension=%u width=%llu height=%u format=%u heap=%u state=0x%x flags=0x%x\n",
                 tracked_count, desc->Dimension, (unsigned long long)desc->Width, desc->Height, desc->Format,
                 heap->Type, state, desc->Flags);
        log_line(line);
        tracked_count++;
    }
    return hr;
}

static void STDMETHODCALLTYPE hooked_barrier(ID3D12GraphicsCommandList *list, UINT count, const D3D12_RESOURCE_BARRIER *barriers)
{
    for (UINT i = 0; in_dll && i < count; i++)
        if (barriers[i].Type == D3D12_RESOURCE_BARRIER_TYPE_TRANSITION)
            for (UINT k = 0; k < tracked_count; k++)
                if (tracked[k].resource == barriers[i].Transition.pResource)
                    tracked[k].state = barriers[i].Transition.StateAfter;
    original_barrier(list, count, barriers);
}

static void transition(ID3D12GraphicsCommandList *list, ID3D12Resource *resource,
                       D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after)
{
    D3D12_RESOURCE_BARRIER b = {.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        .Transition = {.pResource = resource, .Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                       .StateBefore = before, .StateAfter = after}};
    original_barrier(list, 1, &b);
}

static void STDMETHODCALLTYPE hooked_dispatch(ID3D12GraphicsCommandList *list, UINT x, UINT y, UINT z)
{
    original_dispatch(list, x, y, z);
    if (!in_dll) return;
    char line[160];
    snprintf(line, sizeof(line), "launch: frame=%u step=%u groups=%u,%u,%u\n", current_frame, current_step, x, y, z);
    log_line(line);
    if (current_frame == dump_frame && current_step < MAX_STEPS) {
        if (!dump_bytes)
            for (UINT k = 0; k < tracked_count; k++) {
                tracked[k].offset = dump_bytes;
                dump_bytes = (dump_bytes + tracked[k].bytes + 511) & ~(UINT64)511;
            }
        D3D12_HEAP_PROPERTIES hp = {.Type = D3D12_HEAP_TYPE_READBACK, .CreationNodeMask = 1, .VisibleNodeMask = 1};
        D3D12_RESOURCE_DESC rd = {.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER, .Width = dump_bytes, .Height = 1,
            .DepthOrArraySize = 1, .MipLevels = 1, .SampleDesc = {1, 0}, .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR};
        ID3D12Resource *target = NULL;
        if (FAILED(original_create(the_device, &hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COPY_DEST, NULL,
                                   &IID_ID3D12Resource, (void **)&target)))
            ExitProcess(23);
        dump_buffers[current_step] = target;
        D3D12_RESOURCE_BARRIER uav = {.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV};
        original_barrier(list, 1, &uav);
        for (UINT k = 0; k < tracked_count; k++) {
            struct Tracked *t = &tracked[k];
            if (t->heap == D3D12_HEAP_TYPE_READBACK) continue;
            bool move = t->heap == D3D12_HEAP_TYPE_DEFAULT && !(t->state & D3D12_RESOURCE_STATE_COPY_SOURCE);
            if (move) transition(list, t->resource, t->state, D3D12_RESOURCE_STATE_COPY_SOURCE);
            if (t->desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) {
                ID3D12GraphicsCommandList_CopyBufferRegion(list, target, t->offset, t->resource, 0, t->bytes);
            } else {
                D3D12_TEXTURE_COPY_LOCATION src = {.pResource = t->resource, .Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
                D3D12_TEXTURE_COPY_LOCATION dst = {.pResource = target, .Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                                                   .PlacedFootprint = t->footprint};
                dst.PlacedFootprint.Offset = t->offset;
                ID3D12GraphicsCommandList_CopyTextureRegion(list, &dst, 0, 0, 0, &src, NULL);
            }
            if (move) transition(list, t->resource, D3D12_RESOURCE_STATE_COPY_SOURCE, t->state);
        }
        dump_steps = current_step + 1;
    }
    current_step++;
}

static void write_dumps(void)
{
    static wchar_t path[4096];
    _snwprintf(path, 4096, L"%ls\\dump", directory);
    CreateDirectoryW(path, NULL);
    for (UINT step = 0; step < dump_steps; step++) {
        _snwprintf(path, 4096, L"%ls\\dump\\%02u", directory, step);
        CreateDirectoryW(path, NULL);
        char *mapped = NULL;
        D3D12_RANGE read = {0, (SIZE_T)dump_bytes}, none = {0, 0};
        if (FAILED(ID3D12Resource_Map(dump_buffers[step], 0, &read, (void **)&mapped))) ExitProcess(24);
        for (UINT k = 0; k < tracked_count; k++) {
            struct Tracked *t = &tracked[k];
            if (t->heap == D3D12_HEAP_TYPE_READBACK) continue;
            _snwprintf(path, 4096, L"%ls\\dump\\%02u\\r%02u.bin", directory, step, k);
            HANDLE file = CreateFileW(path, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
            if (file == INVALID_HANDLE_VALUE) ExitProcess(24);
            DWORD wrote = 0;
            if (t->desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER)
                WriteFile(file, mapped + t->offset, (DWORD)t->bytes, &wrote, NULL);
            else
                for (UINT y = 0; y < t->rows; y++)
                    WriteFile(file, mapped + t->offset + (UINT64)y * t->footprint.Footprint.RowPitch,
                              (DWORD)t->row_bytes, &wrote, NULL);
            CloseHandle(file);
        }
        ID3D12Resource_Unmap(dump_buffers[step], 0, &none);
        ID3D12Resource_Release(dump_buffers[step]);
        dump_buffers[step] = NULL;
    }
    dump_steps = 0;
}

struct Texture {
    ID3D12Resource *resource, *staging;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint;
    UINT64 bytes;
    UINT w, h, texel, ffx_format;
};

static ID3D12Resource *buffer(ID3D12Device *dev, UINT64 size, D3D12_HEAP_TYPE heap)
{
    D3D12_HEAP_PROPERTIES hp = {.Type = heap, .CreationNodeMask = 1, .VisibleNodeMask = 1};
    D3D12_RESOURCE_DESC rd = {.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER, .Width = size, .Height = 1,
        .DepthOrArraySize = 1, .MipLevels = 1, .SampleDesc = {1, 0}, .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR};
    ID3D12Resource *r = NULL;
    HR(ID3D12Device_CreateCommittedResource(dev, &hp, D3D12_HEAP_FLAG_NONE, &rd,
        heap == D3D12_HEAP_TYPE_UPLOAD ? D3D12_RESOURCE_STATE_GENERIC_READ : D3D12_RESOURCE_STATE_COPY_DEST,
        NULL, &IID_ID3D12Resource, (void **)&r));
    return r;
}

static void barrier(ID3D12GraphicsCommandList *list, ID3D12Resource *resource,
                    D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after)
{
    D3D12_RESOURCE_BARRIER b = {.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        .Transition = {.pResource = resource, .Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                       .StateBefore = before, .StateAfter = after}};
    ID3D12GraphicsCommandList_ResourceBarrier(list, 1, &b);
}

static struct Texture texture(ID3D12Device *dev, UINT w, UINT h, DXGI_FORMAT format, UINT texel,
                              UINT ffx_format, bool writable)
{
    struct Texture t = {.w = w, .h = h, .texel = texel, .ffx_format = ffx_format};
    D3D12_HEAP_PROPERTIES hp = {.Type = D3D12_HEAP_TYPE_DEFAULT, .CreationNodeMask = 1, .VisibleNodeMask = 1};
    D3D12_RESOURCE_DESC rd = {.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D, .Width = w, .Height = h,
        .DepthOrArraySize = 1, .MipLevels = 1, .Format = format, .SampleDesc = {1, 0},
        .Flags = writable ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE};
    HR(ID3D12Device_CreateCommittedResource(dev, &hp, D3D12_HEAP_FLAG_NONE, &rd,
        writable ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS : D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        NULL, &IID_ID3D12Resource, (void **)&t.resource));
    ID3D12Device_GetCopyableFootprints(dev, &rd, 0, 1, 0, &t.footprint, NULL, NULL, &t.bytes);
    t.staging = buffer(dev, t.bytes, writable ? D3D12_HEAP_TYPE_READBACK : D3D12_HEAP_TYPE_UPLOAD);
    return t;
}

static void upload(ID3D12GraphicsCommandList *list, struct Texture *t, const wchar_t *format, UINT frame)
{
    HANDLE file = open_file(format, frame, false);
    char *mapped = NULL;
    D3D12_RANGE none = {0, 0};
    HR(ID3D12Resource_Map(t->staging, 0, &none, (void **)&mapped));
    for (UINT y = 0; y < t->h; y++) {
        DWORD done = 0, want = t->w * t->texel;
        if (!ReadFile(file, mapped + t->footprint.Offset + (UINT64)y * t->footprint.Footprint.RowPitch,
                      want, &done, NULL) || done != want)
            fail("input read", GetLastError());
    }
    ID3D12Resource_Unmap(t->staging, 0, NULL);
    CloseHandle(file);
    D3D12_TEXTURE_COPY_LOCATION src = {.pResource = t->staging,
        .Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, .PlacedFootprint = t->footprint};
    D3D12_TEXTURE_COPY_LOCATION dst = {.pResource = t->resource,
        .Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
    barrier(list, t->resource, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
    ID3D12GraphicsCommandList_CopyTextureRegion(list, &dst, 0, 0, 0, &src, NULL);
    barrier(list, t->resource, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
}

static struct FfxApiResource api_resource(struct Texture t, bool writable)
{
    return (struct FfxApiResource){.resource = t.resource, .description = {.type = FFX_API_RESOURCE_TYPE_TEXTURE2D,
        .format = t.ffx_format, .width = t.w, .height = t.h, .depth = 1, .mipCount = 1,
        .usage = writable ? FFX_API_RESOURCE_USAGE_UAV : FFX_API_RESOURCE_USAGE_READ_ONLY},
        .state = writable ? FFX_API_RESOURCE_STATE_UNORDERED_ACCESS : FFX_API_RESOURCE_STATE_COMPUTE_READ};
}

static void render(ID3D12Device *dev, ffxContext *context, PfnFfxDispatch dispatch)
{
    ID3D12CommandQueue *queue = NULL;
    ID3D12CommandAllocator *allocator = NULL;
    ID3D12GraphicsCommandList *list = NULL;
    ID3D12Fence *fence = NULL;
    D3D12_COMMAND_QUEUE_DESC qd = {.Type = D3D12_COMMAND_LIST_TYPE_DIRECT};
    HR(ID3D12Device_CreateCommandQueue(dev, &qd, &IID_ID3D12CommandQueue, (void **)&queue));
    HR(ID3D12Device_CreateCommandAllocator(dev, D3D12_COMMAND_LIST_TYPE_DIRECT, &IID_ID3D12CommandAllocator, (void **)&allocator));
    HR(ID3D12Device_CreateCommandList(dev, 0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, NULL, &IID_ID3D12GraphicsCommandList, (void **)&list));
    HR(ID3D12Device_CreateFence(dev, 0, D3D12_FENCE_FLAG_NONE, &IID_ID3D12Fence, (void **)&fence));
    HANDLE event = CreateEventW(NULL, FALSE, FALSE, NULL);
    if (!event) ExitProcess(11);
    original_dispatch = list->lpVtbl->Dispatch;
    original_barrier = list->lpVtbl->ResourceBarrier;
    replace_pointer((void *)&list->lpVtbl->Dispatch, (void *)hooked_dispatch);
    replace_pointer((void *)&list->lpVtbl->ResourceBarrier, (void *)hooked_barrier);
    struct Texture color = texture(dev, render_w, render_h, DXGI_FORMAT_R16G16B16A16_FLOAT, 8, FFX_API_SURFACE_FORMAT_R16G16B16A16_FLOAT, false);
    struct Texture depth = texture(dev, render_w, render_h, DXGI_FORMAT_R32_FLOAT, 4, FFX_API_SURFACE_FORMAT_R32_FLOAT, false);
    struct Texture motion = texture(dev, render_w, render_h, DXGI_FORMAT_R16G16_FLOAT, 4, FFX_API_SURFACE_FORMAT_R16G16_FLOAT, false);
    struct Texture target = texture(dev, output_w, output_h, DXGI_FORMAT_R16G16B16A16_FLOAT, 8, FFX_API_SURFACE_FORMAT_R16G16B16A16_FLOAT, true);
    HANDLE info_file = open_file(L"%ls\\frames.bin", 0, false);
    float mv_scale_x = (float)get_uint("HXP_MV_SCALE_X1000", 1000, 100000000) / 1000.f;
    float mv_scale_y = (float)get_uint("HXP_MV_SCALE_Y1000", 1000, 100000000) / 1000.f;
    for (UINT frame = 0; frame < frame_count; frame++) {
        struct FrameInfo info;
        DWORD done = 0;
        if (!ReadFile(info_file, &info, sizeof(info), &done, NULL) || done != sizeof(info))
            fail("frames.bin", GetLastError());
        if (frame) {
            HR(ID3D12CommandAllocator_Reset(allocator));
            HR(ID3D12GraphicsCommandList_Reset(list, allocator, NULL));
        }
        upload(list, &color, L"%ls\\in\\%02u\\color.rgba16f", frame);
        upload(list, &depth, L"%ls\\in\\%02u\\depth.r32f", frame);
        upload(list, &motion, L"%ls\\in\\%02u\\motion.rg16f", frame);
        struct ffxDispatchDescUpscale d = {.header = {FFX_API_DISPATCH_DESC_TYPE_UPSCALE, NULL}, .commandList = list,
            .color = api_resource(color, false), .depth = api_resource(depth, false),
            .motionVectors = api_resource(motion, false), .output = api_resource(target, true),
            .jitterOffset = {info.jitter_x, info.jitter_y}, .motionVectorScale = {mv_scale_x, mv_scale_y},
            .renderSize = {render_w, render_h}, .upscaleSize = {output_w, output_h},
            .frameTimeDelta = 16.666667f, .preExposure = info.pre_exposure, .reset = info.reset != 0,
            .cameraNear = .1f, .cameraFar = 100.f, .cameraFovAngleVertical = 1.04719755f,
            .viewSpaceToMetersFactor = 1.f, .flags = 0};
        current_frame = frame;
        current_step = 0;
        in_dll = true;
        ffxReturnCode_t rc = dispatch(context, &d.header);
        in_dll = false;
        char line[256];
        snprintf(line, sizeof(line), "dispatch: frame=%u status=%u jitter=%g,%g reset=%u\n", frame, rc,
                 (double)info.jitter_x, (double)info.jitter_y, info.reset);
        log_line(line);
        if (rc) ExitProcess(12);
        barrier(list, target.resource, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        D3D12_TEXTURE_COPY_LOCATION src = {.pResource = target.resource, .Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
        D3D12_TEXTURE_COPY_LOCATION dst = {.pResource = target.staging, .Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, .PlacedFootprint = target.footprint};
        ID3D12GraphicsCommandList_CopyTextureRegion(list, &dst, 0, 0, 0, &src, NULL);
        barrier(list, target.resource, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        HR(ID3D12GraphicsCommandList_Close(list));
        ID3D12CommandList *lists[] = {(ID3D12CommandList *)list};
        ID3D12CommandQueue_ExecuteCommandLists(queue, 1, lists);
        HR(ID3D12CommandQueue_Signal(queue, fence, frame + 1));
        HR(ID3D12Fence_SetEventOnCompletion(fence, frame + 1, event));
        if (WaitForSingleObject(event, 1800000) != WAIT_OBJECT_0) {
            log_line("FENCE TIMEOUT: STOP\n");
            ExitProcess(13);
        }
        HR(ID3D12Device_GetDeviceRemovedReason(dev));
        char *mapped = NULL;
        D3D12_RANGE read = {0, (SIZE_T)target.bytes}, none = {0, 0};
        HR(ID3D12Resource_Map(target.staging, 0, &read, (void **)&mapped));
        HANDLE file = open_file(L"%ls\\out\\%02u.rgba16f", frame, true);
        for (UINT y = 0; y < output_h; y++) {
            DWORD wrote = 0;
            if (!WriteFile(file, mapped + target.footprint.Offset + (UINT64)y * target.footprint.Footprint.RowPitch,
                           output_w * 8, &wrote, NULL) || wrote != output_w * 8)
                fail("output write", GetLastError());
        }
        CloseHandle(file);
        ID3D12Resource_Unmap(target.staging, 0, &none);
        write_dumps();
        snprintf(line, sizeof(line), "completed: frame=%u\n", frame);
        log_line(line);
    }
    CloseHandle(info_file);
}

void mainCRTStartup(void)
{
    static wchar_t path[4096];
    static char line[4096];
    DWORD n = GetEnvironmentVariableW(L"HXP_LOG", path, 4096);
    if (!n || n >= 4096) ExitProcess(2);
    output = CreateFileW(path, GENERIC_WRITE, FILE_SHARE_READ, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    if (output == INVALID_HANDLE_VALUE) ExitProcess(2);
    n = GetEnvironmentVariableW(L"HXP_DIR", directory, 2048);
    if (!n || n >= 2048) ExitProcess(2);
    render_w = get_uint("HXP_RENDER_W", 0, 3840);
    render_h = get_uint("HXP_RENDER_H", 0, 2160);
    output_w = get_uint("HXP_OUTPUT_W", 0, 3840);
    output_h = get_uint("HXP_OUTPUT_H", 0, 2160);
    frame_count = get_uint("HXP_FRAMES", 1, 600);
    UINT hardware = get_uint("HXP_HARDWARE", 0, 1);
    UINT auto_exposure = get_uint("HXP_AUTO_EXPOSURE", 0, 1);
    if (!render_w || !render_h || output_w < render_w || output_h < render_h) ExitProcess(20);
    snprintf(line, sizeof(line), "workload: render=%ux%u output=%ux%u frames=%u hardware=%u auto_exposure=%u\n",
             render_w, render_h, output_w, output_h, frame_count, hardware, auto_exposure);
    log_line(line);
    if (!hardware && !LoadLibraryW(L"d3d10warp.dll")) ExitProcess(31);
    n = GetEnvironmentVariableW(L"HXP_DLL", path, 4096);
    if (!n || n >= 4096) ExitProcess(2);
    HMODULE dll = LoadLibraryW(path);
    if (!dll) fail("LoadLibrary", GetLastError());
    PfnFfxQuery query = (PfnFfxQuery)GetProcAddress(dll, "ffxQuery");
    PfnFfxCreateContext create = (PfnFfxCreateContext)GetProcAddress(dll, "ffxCreateContext");
    PfnFfxDestroyContext destroy = (PfnFfxDestroyContext)GetProcAddress(dll, "ffxDestroyContext");
    PfnFfxDispatch dispatch = (PfnFfxDispatch)GetProcAddress(dll, "ffxDispatch");
    if (!query || !create || !destroy || !dispatch) ExitProcess(4);

    IDXGIFactory4 *factory = NULL;
    IDXGIAdapter1 *adapter = NULL;
    HR(CreateDXGIFactory1(&IID_IDXGIFactory4, (void **)&factory));
    if (hardware) {
        SIZE_T best = 0;
        for (UINT i = 0;; i++) {
            IDXGIAdapter1 *candidate = NULL;
            DXGI_ADAPTER_DESC1 desc;
            if (FAILED(IDXGIFactory4_EnumAdapters1(factory, i, &candidate))) break;
            if (SUCCEEDED(IDXGIAdapter1_GetDesc1(candidate, &desc)) &&
                !(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) && desc.DedicatedVideoMemory > best) {
                best = desc.DedicatedVideoMemory;
                if (adapter) IDXGIAdapter1_Release(adapter);
                adapter = candidate;
                candidate = NULL;
            }
            if (candidate) IDXGIAdapter1_Release(candidate);
        }
        if (!adapter) ExitProcess(5);
    } else {
        HR(IDXGIFactory4_EnumWarpAdapter(factory, &IID_IDXGIAdapter1, (void **)&adapter));
    }
    DXGI_ADAPTER_DESC1 desc1;
    HR(IDXGIAdapter1_GetDesc1(adapter, &desc1));
    char name[300];
    WideCharToMultiByte(CP_UTF8, 0, desc1.Description, -1, name, sizeof(name), NULL, NULL);
    snprintf(line, sizeof(line), "adapter: %s\n", name);
    log_line(line);
    ID3D12Device *device = NULL;
    HR(D3D12CreateDevice((IUnknown *)adapter, D3D_FEATURE_LEVEL_12_0, &IID_ID3D12Device, (void **)&device));
    D3D12_FEATURE_DATA_D3D12_OPTIONS1 options1 = {0};
    D3D12_FEATURE_DATA_D3D12_OPTIONS4 options4 = {0};
    ID3D12Device_CheckFeatureSupport(device, D3D12_FEATURE_D3D12_OPTIONS1, &options1, sizeof(options1));
    ID3D12Device_CheckFeatureSupport(device, D3D12_FEATURE_D3D12_OPTIONS4, &options4, sizeof(options4));
    snprintf(line, sizeof(line), "features: wave_ops=%d lanes=%u..%u native16=%d\n", options1.WaveOps,
             options1.WaveLaneCountMin, options1.WaveLaneCountMax, options4.Native16BitShaderOpsSupported);
    log_line(line);

    struct Backend { ffxCreateContextDescHeader header; ID3D12Device *device; };
    struct ffxCreateContextDescUpscaleVersion api_version = {
        .header = {FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE_VERSION, NULL}, .version = FFX_UPSCALER_VERSION};
    struct Backend backend = {.header = {2, &api_version.header}, .device = device};
    struct ffxCreateContextDescUpscale desc = {
        .header = {FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE, &backend.header},
        .flags = FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE | FFX_UPSCALE_ENABLE_DEPTH_INVERTED |
                 (auto_exposure ? FFX_UPSCALE_ENABLE_AUTO_EXPOSURE : 0),
        .maxRenderSize = {render_w, render_h}, .maxUpscaleSize = {output_w, output_h}, .fpMessage = message};
    the_device = device;
    dump_frame = get_uint("HXP_DUMP_FRAME", ~0u, ~0u);
    original_create = device->lpVtbl->CreateCommittedResource;
    replace_pointer((void *)&device->lpVtbl->CreateCommittedResource, (void *)hooked_create);
    ffxContext context = NULL;
    in_dll = true;
    ffxReturnCode_t rc = create(&context, &desc.header, NULL);
    in_dll = false;
    snprintf(line, sizeof(line), "create: status=%u context=%p\n", rc, context);
    log_line(line);
    if (rc || !context) ExitProcess(8);
    struct ffxQueryGetProviderVersion version = {.header = {FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION, NULL}};
    rc = query(&context, &version.header);
    snprintf(line, sizeof(line), "provider: status=%u id=%llu name=%s\n", rc,
             (unsigned long long)version.versionId, version.versionName ? version.versionName : "null");
    log_line(line);
    render(device, &context, dispatch);
    rc = destroy(&context, NULL);
    snprintf(line, sizeof(line), "destroy: status=%u\n", rc);
    log_line(line);
    ExitProcess(rc ? 9 : 0);
}
