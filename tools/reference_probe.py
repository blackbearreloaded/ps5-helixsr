#!/usr/bin/env python3
"""Run the original HelixSR DLL on Windows (WARP by default) over a frame-input
directory and keep every output frame. The reference for the PS5 runtime.

Input directory layout (see tools/reference_inputs.py):
  frames.bin                16 bytes per frame: jitter x, y (float), reset (u32), pre-exposure (float)
  in/NN/color.rgba16f       render size, linear scene colour
  in/NN/depth.r32f          render size, reversed device depth
  in/NN/motion.rg16f        render size, current-to-previous motion in render pixels
Output: out/NN.rgba16f at output size, probe.txt, helixsr.log, run.json.

The HelixSR folder must hold amd_fidelityfx_dx12.dll and the two network files
made by its own setup from the user's DLSS DLL (never committed, never shared)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def win(path):
    return subprocess.check_output(["wslpath", "-w", str(path)], text=True).strip()


def build(runtime, out):
    """runtime: the FSR4 project's prepared build/reference-runtime (Wine headers, FFX headers, WARP, Agility)."""
    wine = runtime / "prefix/usr"
    ffx = runtime / "sdk/Kits/FidelityFX/upscalers/include"
    out.mkdir(parents=True, exist_ok=True)
    (out / "D3D12").mkdir(exist_ok=True)
    shutil.copyfile(runtime / "native-windows/d3d10warp.dll", out / "d3d10warp.dll")
    for path in sorted((runtime / "native-windows/D3D12").glob("*.dll")):
        shutil.copyfile(path, out / "D3D12" / path.name)
    exe = out / "helixsr-reference-probe.exe"
    libs = wine / "lib/x86_64-linux-gnu/wine/x86_64-windows"
    subprocess.run(["clang-18", "--target=x86_64-w64-windows-gnu", "-fms-extensions",
                    "-D__WINE_USE_MSVCRT", "-isystem", str(wine / "include/wine/wine/msvcrt"),
                    "-isystem", str(wine / "include/wine/wine/windows"), "-I" + str(ffx),
                    "-O2", "-Wall", "-Wextra", "-Werror", "-nostdlib", "--ld-path=/usr/bin/ld.lld-18",
                    "-Wl,--entry,mainCRTStartup", "-Wl,--subsystem,console", "-Wl,--no-insert-timestamp",
                    "-L" + str(libs), str(ROOT / "tools/reference_probe.c"),
                    "-lkernel32", "-lucrtbase", "-ld3d12", "-ldxgi", "-o", str(exe)], check=True)
    return exe


def run(exe, helixsr, frames_dir, render, output, frames, hardware=False, auto_exposure=False,
        mv_scale=(1.0, 1.0), timeout=7200, dump_frame=None):
    frames_dir = Path(frames_dir).resolve()
    (frames_dir / "out").mkdir(exist_ok=True)
    dll = Path(helixsr) / "amd_fidelityfx_dx12.dll"
    for name in ("helixsr_weights.bin", "helixsr_kernels.pak"):
        if not (Path(helixsr) / name).is_file():
            raise SystemExit(f"{name} missing beside the HelixSR DLL: run its setup first")
    log = Path(helixsr) / "helixsr.log"
    log.unlink(missing_ok=True)
    values = dict(HXP_DLL=win(dll), HXP_LOG=win(frames_dir / "probe.txt"), HXP_DIR=win(frames_dir),
                  HXP_RENDER_W=render[0], HXP_RENDER_H=render[1], HXP_OUTPUT_W=output[0],
                  HXP_OUTPUT_H=output[1], HXP_FRAMES=frames, HXP_HARDWARE=int(hardware),
                  HXP_AUTO_EXPOSURE=int(auto_exposure), HXP_MV_SCALE_X1000=round(mv_scale[0] * 1000),
                  HXP_MV_SCALE_Y1000=round(mv_scale[1] * 1000))
    if dump_frame is not None:
        values["HXP_DUMP_FRAME"] = dump_frame
    env = dict(os.environ, **{k: str(v) for k, v in values.items()})
    env["WSLENV"] = ":".join(filter(None, [env.get("WSLENV", ""), *values]))
    result = subprocess.run([str(exe)], cwd=exe.parent, env=env, timeout=timeout,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if log.is_file():
        shutil.copyfile(log, frames_dir / "helixsr.log")
    probe = (frames_dir / "probe.txt").read_text() if (frames_dir / "probe.txt").is_file() else ""
    receipt = dict(exit=result.returncode, probe_sha256=sha(exe), dll_sha256=sha(dll),
                   weights_sha256=sha(Path(helixsr) / "helixsr_weights.bin"),
                   kernels_sha256=sha(Path(helixsr) / "helixsr_kernels.pak"),
                   backend="hardware adapter" if hardware else "WARP", render=list(render),
                   output=list(output), frames=frames, auto_exposure=auto_exposure,
                   outputs={p.name: sha(p) for p in sorted((frames_dir / "out").glob("*.rgba16f"))})
    (frames_dir / "run.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if result.returncode or f"completed: frame={frames - 1}" not in probe:
        raise SystemExit(f"reference run failed ({result.returncode}):\n{probe}\n{result.stdout}")
    # With automatic exposure the original does not always start the same way: now and then its
    # first frames come back black, and the frames after them differ from a good run's.
    if not any((frames_dir / "out/00.rgba16f").read_bytes()):
        raise SystemExit("the original returned a black first frame: run the reference again")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("frames_dir", type=Path)
    parser.add_argument("--helixsr", type=Path, required=True, help="folder with the HelixSR DLL and its network files")
    parser.add_argument("--runtime", type=Path, required=True, help="ps5-fsr4 build/reference-runtime")
    parser.add_argument("--build", type=Path, default=ROOT / "build/reference-probe")
    parser.add_argument("--render", type=int, nargs=2, required=True)
    parser.add_argument("--output", type=int, nargs=2, required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--hardware", action="store_true", help="largest hardware adapter instead of WARP")
    parser.add_argument("--auto-exposure", action="store_true")
    parser.add_argument("--mv-scale", type=float, nargs=2, default=(1.0, 1.0))
    parser.add_argument("--dump-frame", type=int, help="write every DLL resource after each dispatch of this frame")
    args = parser.parse_args()
    exe = build(args.runtime.resolve(), args.build.resolve())
    receipt = run(exe, args.helixsr.resolve(), args.frames_dir, args.render, args.output, args.frames,
                  args.hardware, args.auto_exposure, args.mv_scale, dump_frame=args.dump_frame)
    print(json.dumps(dict(exit=receipt["exit"], frames=len(receipt["outputs"]), backend=receipt["backend"])))


if __name__ == "__main__":
    main()
