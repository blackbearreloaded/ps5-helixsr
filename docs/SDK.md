# The ps5_helixsr runtime

`include/ps5helixsr/ps5_helixsr.h` is the whole interface: a C header with a
C++17 implementation behind it (`src/ps5_helixsr.cpp`, `src/runtime_plan.cpp`,
HelixSR's host model in `third_party/helixsr/model` and the generated shader
table). An application compiles those sources with its own; the frame runner
and the showcase show how (`tools/build_runner.py`, `build_title`).

The application owns the Vulkan device, the command buffer, the input and
output images, submission and presentation. A context owns the network: its
weights, pipelines, intermediate tensors and the history of earlier frames.

## What it runs

NVIDIA's DLSS Model E network as HelixSR 1.2.0 runs it by default: the main
network at every ratio from 1× (native anti-aliasing) to 3× (Ultra
Performance), with NVIDIA's choice of output kernel at each ratio. NVIDIA's own
choice above 2.5×, a second network, is not carried; HelixSR does not use it by
default either.

A context is made for one render size and one output size. A game that changes
its render size makes a context per size, which takes about 50 ms once the
shaders are in the pipeline cache.

| | |
| --- | --- |
| Output size | up to 4096×4096 |
| Ratio | render ≤ output ≤ 3 × (render + 1) on each axis: 3× with a truncated render size, as 853×480 for 2560×1440 |
| Exposure | 1.0, or computed from the frame (`PS5HELIXSR_FLAG_AUTO_EXPOSURE`); `pre_exposure` as in FidelityFX |
| Not there | sharpening, a reactive mask, a render size that changes within a context, SDR input |

## Images

One mip level, one array layer:

| Image | Size | Format | Use |
| --- | --- | --- | --- |
| color | render | `VK_FORMAT_R16G16B16A16_SFLOAT` | sampled |
| depth | render | `VK_FORMAT_R32_SFLOAT` | sampled |
| motion vectors | render | `VK_FORMAT_R16G16_SFLOAT` | sampled |
| output | output | `VK_FORMAT_R16G16B16A16_SFLOAT` | storage |

The conventions are FidelityFX's, which HelixSR takes:

- **Color** is linear HDR.
- **Depth** is inverted: 1 at the near plane.
- **Motion vectors** point from a pixel to where it was: previous position
  minus current position, in render pixels, without the jitter.
- **Jitter** (`jitter_x`, `jitter_y`) is the sub-pixel offset of the projection
  in render pixels: a pixel samples its centre minus the jitter. Use the usual
  Halton(2, 3) sequence of 8 × (output ÷ render)² phases, minus one half
  (`examples/helixsr_showcase/main.c`).

Inputs are in `GENERAL` or `SHADER_READ_ONLY_OPTIMAL` layout (the dispatch
description says which), the output in `GENERAL`, where the runtime leaves it.
The application makes its own writes to the inputs visible to compute before
the dispatch.

## Device

On the PS5 the driver SDK must be staged with the HelixSR profile
(`make driver-sdk`, [BUILDING.md](../BUILDING.md)). Create the device with:

- `shaderInt16`;
- `storageBuffer16BitAccess`, **with `VK_KHR_16bit_storage` named among the
  enabled extensions**: ps5vk takes the feature only with its extension, and
  `vkCreateDevice` returns `VK_ERROR_FEATURE_NOT_PRESENT` otherwise. The
  network keeps half floats in buffers, and two lanes may write the two halves
  of one word; a 32-bit store would lose one of them.

A desktop driver also needs `shaderFloat16`, `shaderInt64`,
`shaderStorageImageExtendedFormats` and `computeDerivativeGroupLinear` of
`VK_KHR_compute_shader_derivatives`, and 32-lane subgroups
([BUILDING.md](../BUILDING.md#running-on-the-host)).
`PS5HELIXSR_FLAG_SUBGROUP_SIZE_CONTROL` asks for 32 lanes through
`subgroupSizeControl` on every pipeline, for a driver that offers several
sizes; the PS5's waves are 32 lanes without it.

## Use

```c
ps5helixsr_context_desc cd = {.struct_size = sizeof(cd), .physical_device = physical, .device = device,
    .output_width = 1920, .output_height = 1080, .render_width = 1280, .render_height = 720,
    .flags = PS5HELIXSR_FLAG_AUTO_EXPOSURE, .canonical_weights = model, .canonical_weights_bytes = model_bytes,
    .pipeline_cache = cache};
ps5helixsr_context *context;
if (ps5helixsr_context_create(&cd, &context) != PS5HELIXSR_OK) ...

/* every frame, into a command buffer the application has begun */
ps5helixsr_dispatch_desc dd = {.struct_size = sizeof(dd), .command_buffer = cmd,
    .color = color_view, .depth = depth_view, .motion_vectors = motion_view, .output = output_view,
    .color_layout = VK_IMAGE_LAYOUT_GENERAL, .depth_layout = VK_IMAGE_LAYOUT_GENERAL,
    .motion_vectors_layout = VK_IMAGE_LAYOUT_GENERAL,
    .jitter_x = jx, .jitter_y = jy, .pre_exposure = 1.0f, .reset = camera_cut};
ps5helixsr_dispatch(context, &dd);
/* ... the application's own commands, vkQueueSubmit, and once the fence has signaled: */
ps5helixsr_context_notify_completed(context);
```

- **The kernels** are compiled into the runtime by default. A runtime built
  without them (`tools/build_runtime_assets.py --external-kernels`, as the
  showcase is) holds only their SHA-256: the application then passes
  `kernels.bin` (`make network-files`) in `kernels` and `kernels_bytes`, and
  creation returns `PS5HELIXSR_ERROR_INVALID_ARGUMENT` when a kernel is
  missing or is not the pinned one. That is what lets an application be built
  and passed on with nothing of NVIDIA's inside.
- **The weights** are the 1,903,872 bytes of `helixsr_weights.bin`, which
  HelixSR's setup writes (`make model`). Context creation checks their size,
  reorders them for the kernels and copies them to the GPU; the application may
  free its copy afterwards. They are NVIDIA's: an application reads them from a
  file its user provides and does not ship them.
- **`ps5helixsr_dispatch` only records.** It updates the context's constants
  and descriptor sets and records the network's dispatches (18 without
  automatic exposure, 22 with it) and their barriers. It allocates nothing,
  submits nothing and waits for nothing.
- **One dispatch at a time.** A context refuses a second dispatch
  (`PS5HELIXSR_ERROR_IN_FLIGHT`) until `ps5helixsr_context_notify_completed`
  says the first has finished on the GPU. Wait for it before destroying the
  context too.
- **History.** The first dispatch of a context starts without history. Set
  `reset` at a camera cut. The jitter of one dispatch is the next one's
  previous jitter.
- `ps5helixsr_get_memory_requirements` gives the GPU memory a context will
  take, without creating it.

## Pipeline cache

Compiling the network's shaders for the PS5 takes about 36 seconds, once.
Create a `VkPipelineCache`, pass it in the context description, and save its
data after the first context (`vkGetPipelineCacheData`, about 8 MB); with that
data loaded, a context is made in 50 ms. The cache belongs to one build of
`libps5vk.a` and of the shaders, and it holds the network's kernels compiled:
keep it on the console that made it. The showcase does all of this
(`save_pipeline_cache` in `examples/helixsr_showcase/main.c`).

## Time and memory

Measured on a PS5 by the showcase's benchmark, with automatic exposure
([README](../README.md#performance)): 2.1 ms for a 1920×1080 output, 3.0 ms for
2560×1440 and 5.6 ms for 3840×2160. The time follows the output size and
hardly the render size.

The context keeps its constants in mapped memory. The ps5vk driver flushes
mapped coherent memory at every submission, at about a millisecond per
gigabyte: an application that keeps large coherent buffers mapped pays that on
each of its submissions, HelixSR's included. Map non-coherent memory, as the
showcase does, or unmap what a frame does not write.
