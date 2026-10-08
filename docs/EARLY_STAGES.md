# The early convolution stages

Four launches at the start of the main network hold nearly all of its
multiply-adds: five convolutions, each followed by bias, ReLU and a 2×2 max
pool. The kernels translated from NVIDIA's PTX spend most of a frame there on
the console, so the runtime replaces these four launches with its own shaders.
Everything after them runs the translated kernels.

## Tensors

`Wp` and `Hp` are the output size padded to multiples of 128. An NCHW8 element
`(c, y, x)` has scalar index `(((c / 8) * H + y) * W + x) * 8 + c % 8`. Half
tensors take two bytes per scalar, E5M3 tensors one: an E5M3 byte `b` is the
half float `b << 7`. The unpooled ("skip") output of a stage is four whole
NCHW8 planes at the pooled size, one per pixel of the 2×2 pooling cell, in the
order (0,0), (1,0), (0,1), (1,1) of (x, y).

| Stage | Input | Pooled output | Unpooled output |
| --- | --- | --- | --- |
| 3×3, 16→32 | NET_IN, half, Wp/4 × Hp/4 | half scratch, Wp/8 × Hp/8 | E5M3, B3 offset 0 |
| 1×1, 32→64 | the first stage's half pool | E5M3, Wp/16 × Hp/16, B4 offset Wp×Hp | E5M3, B4 offset 0 |
| 3×3, 64→128 | B4 offset Wp×Hp, E5M3 | half, Wp/32 × Hp/32, B5 offset 0 | half, B6 offset 0 |
| 3×3, 128→128 | B5 offset 0, half | half, Wp/64 × Hp/64, B4 offset Wp×Hp | half, B7 offset 0 |
| 3×3, 128→256 | B4 offset Wp×Hp, half | half, Wp/128 × Hp/128, B5 offset 0 | half, B8 offset 0 |

The first two stages are one launch in the original. Its outputs are stored as
E5M3, but the second layer consumes the first layer's pool in half precision,
so the runtime computes both in half scratch buffers and encodes what the graph
keeps with three small dispatches (`shaders/encode_e5m3_resident.hlsl`). B4 and
B5 are reused by later stages; the plan orders the dispatches accordingly
(`tools/compact_runtime.py`, mirrored by `src/runtime_plan.cpp`).

## The kernels

`tools/conv_kernels.py` generates one GLSL compute shader per stage.

- **Lanes are pixels.** One invocation computes one pooling cell (its four
  pixels side by side) for a block of 32 output channels; the 64 invocations of
  a group are neighbouring cells, and the block is chosen by the group's `y`.
  A weight is therefore the same for a whole wave.
- **Weights come through the scalar path.** The context reorders each stage's
  weights once, so that the 32 outputs of one (tap, input channel) are 16
  adjacent words, and the compiler loads them with one 16-word scalar load.
  It merges scalar loads only from a constant base, so each kernel carries its
  weight offset as a literal; the plan checks it against the planner's.
- **Arithmetic is packed FP16.** Activations are pairs of halves in vector
  registers. Each product is a packed FP16 multiply and a packed FP16 add on a
  pair of output channels, with the activation's half picked by the operand
  swizzle: 512 of each and 8 weight loads per loop iteration, in 96 vector
  registers.
- **The multiply and the add stay apart.** The source marks the accumulators
  `precise`, so the product is rounded to half before the add. That is what a
  software renderer computes, the original on WARP and this runtime on
  lavapipe alike, and with it the console's convolutions are the same bits as
  the host's. A fused multiply-add rounds once: it is a fifth faster over the
  whole frame (1.69 ms against 2.14 ms at 1080p), and on the console it moved
  35% of the first stage's values by more than one step, the output by 71 dB.
  `HELIXSR_CONV_FUSED=1` generates that form.
- **Accumulation is FP16 in tap order.** That is not the rounding sequence of
  NVIDIA's kernels, and not the DLL's either; HelixSR makes the same choice
  since 1.1.0. It is what separates this runtime's output from the original's
  by about 60 dB ([VALIDATION.md](../VALIDATION.md)).

`shaders/convolution_pool.hlsl` and `shaders/convolution_pool_resident.hlsl`
keep NVIDIA's accumulation instead: groups of 16 input channels, column-major
taps, lane-rotated pairs, one half rounding per group. They are not used by the
runtime; the tests compare them with a simulation of the PTX
(`tools/ptx_reference.py`) on dense weights, every isolated tap and NaNs, which
pins the layouts above.

## What they need from the device

Half-float arithmetic, nothing else: no subgroup operation, no image, and
buffers read and written as 32-bit words. `make check-shaders` compiles them
with glslang and `spirv-val`; `make shaders` also compiles each with the PS5
shader compiler.
