# Third-party notices

## HelixSR and DLSS credits

**HelixSR is the work of [lonewolf0622](https://github.com/lonewolf0622/HelixSR).**
It runs NVIDIA's DLSS Model E network as ordinary compute shaders behind an
FSR 3.1 interface: the PTX-to-HLSL translation of NVIDIA's kernels, the host
model that plans every launch, the scheduling recipes and the packed FP16
arithmetic are HelixSR's. This project is built on the setup components that
HelixSR 1.2.0 publishes and would not exist without them.

**NVIDIA created DLSS, the Model E network and its trained weights.** They are
NVIDIA's property.

BlackBearReloaded's work in this repository is the native PS5 adaptation: the
Vulkan runtime, the early convolution stages written for the console GPU, the
driver profile, the reference and acceptance tools and the showcase port. It
claims authorship of neither HelixSR nor DLSS, and it is affiliated with
neither project.

## This repository

Copyright (C) 2026 BlackBearReloaded. The runtime, tools, tests, examples and
documents in this repository are licensed under the GNU General Public License,
version 3 or (at your option) any later version (`GPL-3.0-or-later`). The
complete license text is in [`LICENSE`](LICENSE).

## HelixSR components in this repository

`third_party/helixsr` holds unmodified files of the HelixSR 1.2.0 setup
archive, Copyright 2026 lonewolf0622, under the Apache License 2.0
([`third_party/helixsr/LICENSE`](third_party/helixsr/LICENSE)): the host model
(`model/`), the kernel and sampler tables, the PTX simulator and the FP16
helper. [`sources.json`](sources.json) pins the release archive and every file.
The build fetches the same release (`tools/build_inputs.json`) and runs its
setup and its translator from there.

## NVIDIA DLSS material

No NVIDIA DLL, weight, kernel or shader made from them is stored in this
repository, and this project publishes no binary that contains any.

The build makes them on your machine, as HelixSR's setup does: from NVIDIA's
`nvngx_dlss.dll` 310.7.0, which NVIDIA distributes at
[github.com/NVIDIA/DLSS](https://github.com/NVIDIA/DLSS) under
[its own license](https://github.com/NVIDIA/DLSS/blob/v310.7.0/LICENSE.txt).
What comes out of it stays NVIDIA's material under that license:

- `helixsr_weights.bin`, the weights (the showcase's `model.bin`);
- the PTX of the network's kernels, the HLSL translated from it, the SPIR-V
  compiled from that, and the pipeline cache a console saves;
- every executable that embeds those shaders: the frame runner and the
  showcase app.

Build them for your own use. Do not commit them, and do not pass them on.

## The Vulkan driver

`external/ps5-vulkan` is a git submodule:
[BlackBearReloaded's fork](https://github.com/blackbearreloaded/ps5-vulkan) of
[Manuel Pereira's ps5-vulkan](https://github.com/mpereiraesaa/ps5-vulkan),
licensed `GPL-3.0-or-later`. Its own `LICENSING.md` covers its dependencies and
derived sources. Programs linked with the staged SDK (`libps5vk.a`,
`libpsbc.a`) are distributed under those terms as well.

## The showcase

`examples/helixsr_showcase` is the PS5 FSR4 Showcase of
[ps5-fsr4](https://github.com/blackbearreloaded/ps5-fsr4) (BlackBearReloaded,
`GPL-3.0-or-later`) with the upscaler replaced; its music is that app's. The
HUD and the signs use DejaVu Sans, rasterized at build time from the system
font under the Bitstream Vera license.

## Build and test tools

Fetched or installed, never stored here:

- [PS5 Payload SDK](https://github.com/ps5-payload-dev/sdk) by John Törnblom
  (ps5-payload-dev), through the
  [native app template](https://github.com/blackbearreloaded/ps5-native-app-boilerplate);
- Microsoft's [DirectX Shader Compiler](https://github.com/microsoft/DirectXShaderCompiler)
  (University of Illinois/NCSA and MIT licenses), glslang and SPIRV-Tools
  (Khronos, Apache 2.0);
- [Mesa](https://www.mesa3d.org) lavapipe (MIT) for host runs;
  `patches/mesa-26.2.1-lavapipe-32-lanes.patch` is a change to Mesa under
  Mesa's license;
- for the reference runs only: the HelixSR release DLL, AMD's FidelityFX API
  headers (MIT) and Microsoft's WARP and Agility SDK redistributables, prepared
  as [ps5-fsr4 describes](https://github.com/blackbearreloaded/ps5-fsr4/blob/main/docs/FSR4_REFERENCE_RUNTIME.md).
