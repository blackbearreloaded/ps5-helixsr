# Validation

Run the host tests (`make check`) and the acceptance below for a change to the
runtime, the shaders, the compiler or the driver. Store captures, logs, frames
and detailed comparisons locally under `build/` or outside the tree. Do not
publish test receipts or run diaries in this repository, and nothing made from
NVIDIA's DLL.

## Acceptance

**Reference.** The original: the HelixSR 1.2.0 DLL itself. It is a drop-in
FidelityFX 3.1 upscaler, so a small Direct3D 12 program
(`tools/reference_probe.c`, run by `tools/reference_probe.py` through Windows
interop from WSL) creates an upscaler context on WARP, Windows' software
renderer, feeds it a sequence of frames and saves what it returns. WARP offers
the 32-lane waves and native 16-bit types the kernels need.

One correction is made to the reference, and only for the reference:
`tools/reference_setup.py` builds its network files with lanes read through a
uniform lane index. The kernels read other lanes with a per-lane index
(`WaveReadLaneAt(v, lane + n)`), and WARP executes that wrongly: in lanes 4 to 7
of every 8 the read lands 8 lanes further on (12 wrong reads of 32, measured
with an instrumented kernel). With the uniform index WARP returns what the
kernels are written to compute on hardware.

**Inputs.** `tools/reference_inputs.py` writes analytic scenes as frame-input
directories: jittered color, inverted depth and motion vectors at render size,
with FidelityFX's conventions, and placed so that no sample falls exactly
between two texels. Reference, host and PS5 read the same files.

**Control.** The same runtime, shaders and inputs on the host
(`build/helixsr_host_frames`), on lavapipe with 32-lane subgroups
([BUILDING.md](BUILDING.md#running-on-the-host)).

**Rule** (`tools/check_acceptance.py`). A PS5 run of a scenario is accepted when
all of these hold:

- the run completed and saved every frame;
- no frame holds a non-finite value;
- on every frame, its PSNR against the original is at least the control's PSNR
  minus 1 dB.

The PSNR is that of the color channels, with the original frame's largest value
(at least 1) as the peak, so that an HDR frame is not judged by absolute error.

Neither the control nor the PS5 can match the original bit for bit. The early
convolutions here accumulate FP16 products in another order than the DLL's
kernels (as HelixSR itself differs from NVIDIA's arithmetic since 1.1.0), and
the network quantizes activations to 8-bit E5M3 codes between stages, which
turns a last-bit difference into a code step. The rule therefore asks that the
PS5 be as close to the original as the same runtime on an independent,
conformant Vulkan driver. It is the rule of
[ps5-fsr4](https://github.com/blackbearreloaded/ps5-fsr4/blob/main/VALIDATION.md#fsr4-acceptance),
margin included.

**Scenarios and results.** The PSNR columns give the lowest and the highest
frame of each sequence against the original; the last column but one is the
lowest PSNR between the PS5's and the control's own outputs.

| Scenario | Render → output | What it covers | Frames | PS5 (dB) | Host (dB) | PS5 against host (dB) | |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| s-static | 640×360 → 960×540 | nothing moves; steady history | 8 | 56.8 – 66.2 | 56.7 – 66.3 | 67.0 | accepted |
| s-pan | 640×360 → 960×540 | the camera pans | 8 | 58.1 – 66.2 | 58.0 – 66.3 | 67.8 | accepted |
| s-object | 640×360 → 960×540 | an object crosses a still background | 8 | 57.1 – 66.2 | 57.0 – 66.3 | 67.3 | accepted |
| s-motion | 640×360 → 960×540 | camera and object move | 8 | 58.1 – 66.2 | 58.1 – 66.3 | 68.0 | accepted |
| s-cut | 640×360 → 960×540 | 12 frames, a camera cut at frame 8 | 12 | 56.8 – 66.4 | 56.7 – 66.4 | 67.0 | accepted |
| s-bright | 640×360 → 960×540 | HDR values up to 9 | 8 | 50.3 – 60.3 | 50.2 – 60.2 | 65.7 | accepted |
| s-auto | 640×360 → 960×540 | the bright scene with automatic exposure | 8 | 58.9 – 67.0 | 58.5 – 67.1 | 66.4 | accepted |
| s-native | 960×540 → 960×540 | native anti-aliasing (1×) | 8 | 62.5 – 67.2 | 62.3 – 67.2 | 68.9 | accepted |
| s-balanced | 564×317 → 960×540 | Balanced (1.7×) | 8 | 62.3 – 66.0 | 62.0 – 65.9 | 67.5 | accepted |
| s-performance | 480×270 → 960×540 | Performance (2×) | 8 | 61.4 – 65.6 | 61.2 – 65.6 | 66.7 | accepted |
| s-extra | 384×216 → 960×540 | 2.5× | 8 | 61.3 – 65.3 | 61.2 – 65.3 | 66.9 | accepted |
| s-ultra | 320×180 → 960×540 | Ultra Performance (3×) | 8 | 55.8 – 64.4 | 55.8 – 64.3 | 65.5 | accepted |
| s-odd | 638×358 → 958×538 | sizes that are not multiples of 8 | 8 | 62.4 – 66.4 | 62.1 – 66.3 | 68.7 | accepted |
| t-quality | 1280×720 → 1920×1080 | Quality (1.5×) at the target size | 4 | 64.9 – 65.8 | 64.9 – 65.8 | 72.2 | accepted |
| t-performance | 960×540 → 1920×1080 | Performance (2×) | 4 | 64.5 – 66.9 | 64.3 – 66.8 | 70.8 | accepted |
| t-extra | 768×432 → 1920×1080 | 2.5× | 4 | 63.9 – 66.3 | 63.7 – 66.3 | 69.9 | accepted |
| t-ultra | 640×360 → 1920×1080 | Ultra Performance (3×) | 4 | 54.5 – 65.5 | 54.5 – 65.5 | 68.2 | accepted |
| t-odd | 1278×718 → 1918×1078 | odd sizes | 4 | 65.3 – 66.0 | 65.1 – 65.9 | 72.2 | accepted |
| t-auto | 1280×720 → 1920×1080 | bright scene, automatic exposure | 4 | 64.4 – 68.3 | 64.2 – 68.3 | 71.1 | accepted |
| k2-quality | 1706×960 → 2560×1440 | 1440p, Quality | 4 | 64.9 – 67.2 | 64.7 – 67.2 | 72.8 | accepted |
| k2-performance | 1280×720 → 2560×1440 | 1440p, Performance | 4 | 64.7 – 67.1 | 64.5 – 67.1 | 71.4 | accepted |
| k4-performance | 1920×1080 → 3840×2160 | 4K, Performance | 4 | 65.5 – 67.4 | 65.4 – 67.4 | 73.5 | accepted |
| k4-extra | 1536×864 → 3840×2160 | 4K, 2.5× | 4 | 64.8 – 67.0 | 64.8 – 67.0 | 71.2 | accepted |
| k4-ultra | 1280×720 → 3840×2160 | 4K, Ultra Performance | 4 | 62.3 – 64.5 | 62.2 – 64.4 | 71.5 | accepted |

The PS5 and the control are closer to each other (about 70 dB at the target
sizes) than either is to the original: what separates them from the original
is the convolutions' arithmetic, which they share.

**Ultra Performance.** At 3×, to 960×540 and to 1920×1080, the second frame of
a sequence is 54 to 56 dB from the original, on the PS5 and on the control
alike, where 2.5× gives 64 dB; the frames after it climb back to 60 dB. The
first frame matches like any other (65 dB), and at 3840×2160 the dip is 2 dB.
The cause is not known. It is common to both sides and inside the rule, and it
is the one place where this runtime is measurably further from the original.

**Automatic exposure and the original.** With automatic exposure the original
does not start the same way in every run. A run whose first frame is held back
by the stage dump (`--dump-frame 0`), and every run of a single frame, give one
result, which the runtime matches to 64 to 68 dB. A plain run of several frames
at 1280×720 → 1920×1080 gave a first frame 41 dB away from that, with the same
inputs, and one run of the small scene returned two black frames first. The
kernels and their arguments are the same in all of them, so the reference for
`t-auto` is the run with the dump; `tools/reference_probe.py` refuses a run
whose first frame is black.

The scenes are synthetic. `object`, `motion` and `cut` exercise disocclusion
and a history reset; a real game's content, with its own motion vectors and
exposure, has no numeric reference here, and the
[showcase](examples/helixsr_showcase/README.md) is judged by eye.

**Small frames are not used.** At 96×60 a frame has 5,760 pixels and a single
pixel that keeps or rejects its history decides the frame's PSNR: the control
itself scatters between 41 and 59 dB against the original there, and the PS5
and the control by 2 dB against each other, in both directions. The scenario
coverage therefore runs at 960×540 and the target sizes.

Run one scenario on each side, then check it:

    python3 tools/reference_setup.py build/inputs/helixsr build/reference-helixsr --dlss /path/to/nvngx_dlss.dll --dxc build/inputs/dxc/bin/dxc
    python3 tools/reference_inputs.py build/cases/pan --scenario pan --render 1280 720 --output 1920 1080 --frames 4
    python3 tools/reference_probe.py build/cases/pan --helixsr build/reference-helixsr --runtime <reference runtime> \
        --render 1280 720 --output 1920 1080 --frames 4
    build/helixsr_host_frames build/cases/pan build/helixsr-setup/helixsr_weights.bin 1280 720 1920 1080 4 build/host/pan
    make runner RUNNER_CASES=build/cases/pan
    python3 tools/check_acceptance.py build/cases/pan build/host/pan <ps5-results>

`<reference runtime>` is the folder with WARP, the Agility SDK and the Wine
development files that
[ps5-fsr4's reference procedure](https://github.com/blackbearreloaded/ps5-fsr4/blob/main/docs/FSR4_REFERENCE_RUNTIME.md)
prepares. Deploy the runner's `PPSA89012` folder to the locally configured
console; it runs every case, writes `NAME-NN.rgba16f` into its `results`
folder and closes itself. A case takes `DIR::::1` for automatic exposure, and
`--auto-exposure` on the other two sides.

`tools/compare_frames.py` compares any two sets of frames and writes them side
by side with their difference. `--dump-frame 0` on the reference and the host
side saves every tensor after each launch of the first frame, and
`tools/compare_stages.py` then names the first launch whose output differs:
that is how the defects of the first runs were found.

**Status.** Every scenario above is accepted on the PS5; the largest shortfall
against the control on any frame is 0.11 dB. Rerun the matrix after any change
to the compiler, the driver or the runtime.

## What the comparison found

The reference found four defects that tests without it had passed, each a way
in which the same shaders behave differently from one implementation to the
next:

- **Half-word stores.** DXC lowers a 16-bit store into a byte-address buffer to
  a 32-bit read-modify-write. Two lanes writing the two halves of one word
  lost one of the writes, on the PS5 and on lavapipe alike. The generator now
  stores through a 16-bit view of the same buffer, which needs
  `storageBuffer16BitAccess` ([the runtime](docs/SDK.md#device)).
- **A reduction under a lane mask.** The kernel that sums the frame's luminance
  for automatic exposure shuffles its last pass under a mask; the translated
  code let the lanes outside the mask skip the pass, and the lanes inside then
  read lanes that were not executing. Direct3D drivers return those lanes'
  operand; SPIR-V leaves the value undefined, and lavapipe returned their
  partial sums. The lanes outside the mask now take part with zero
  (`MASKED_SHUFFLES` in `tools/build_generated.py`).
- **Exposure without automatic exposure.** The DLL feeds the input kernel an
  exposure of 1.0 from a 1×1 texture; the runtime's own texture held 0.
- **A fused multiply-add.** The early convolutions asked for a fused FP16
  multiply-add. The console fuses it; lavapipe, and WARP under the original,
  multiply, round and add. The first matrix had one frame of 152 miss the rule
  for it, by 0.5 dB (automatic exposure at 1080p, 66.8 dB against the
  control's 68.3 dB). The frame runner's stage dump (`stages 1`, the fifth
  field of a case: `DIR::::1:1`) showed the network's input identical on both
  sides and 35% of the first convolution's values more than one step apart; a
  build with the two operations apart made that stage the same bits on both
  and the frame 68.3 dB. The convolutions now keep them apart
  ([the early stages](docs/EARLY_STAGES.md)).

## Host tests

`make check` runs 23 tests: the planner's graph against its contract, the FP16
and E5M3 numerics over every bit pattern, the weight layouts, the exact
convolution stages against a simulation of the PTX, the binding and sequence
contracts, the asset table, the plan of 52 configurations byte for byte, the
recorded command buffers, the public API on a real device, and the showcase's
tables against the documents. They establish structure and arithmetic; the
acceptance above is what establishes the picture.
