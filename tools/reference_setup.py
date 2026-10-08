#!/usr/bin/env python3
"""Prepare a HelixSR folder that gives exact results on Windows WARP.

The original HelixSR 1.2.0 release builds its network files on the user's machine from
NVIDIA's DLSS DLL. This tool runs that same setup into a working copy of the release,
with one change to its shader runtime header: lanes are read with a uniform lane index.

Why: the kernels read other lanes with a per-lane index (WaveReadLaneAt(v, lane + n)).
WARP executes that wrongly: in lanes 4-7 of every 8 the read lands 8 lanes further on
(measured with an instrumented input kernel; 12 wrong reads of 32). A loop over the 32
lanes with a uniform index reads the same values correctly, so the reference pictures
and per-stage data are those the kernels are written to produce on hardware.

Nothing made here may be committed or shared: the network files hold NVIDIA's model."""
import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys

RELEASE_ZIP_SHA256 = "7c9aeac2e3dcd73f6e9b2dd8b04c41a828fd8e1a8fe30a2ac77787097ddbeb0e"
DLSS_SHA256 = "be6e434a94ca32499515eb62ca0e6c274526055d568d0426e4c652dcdfb6ee6e"
PER_LANE = "uint px_rd(uint v, uint l) { return WaveReadLaneAt(v, (WaveGetLaneIndex() & ~31u) | l); }"
UNIFORM = """uint px_rd(uint v, uint l) {
    // reference runs on WARP: a per-lane index is mis-executed there, a uniform one is not
    uint r = 0u;
    [loop] for (uint k = 0u; k < 32u; ++k) { uint x = WaveReadLaneAt(v, k); r = (l == k) ? x : r; }
    return r;
}"""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("release", type=Path, help="extracted HelixSR 1.2.0 release folder")
    parser.add_argument("out", type=Path, help="working copy to create")
    parser.add_argument("--dlss", type=Path, required=True, help="nvngx_dlss.dll 310.7.0")
    parser.add_argument("--dxc", type=Path, required=True, help="DXC 1.9 (Linux build)")
    parser.add_argument("--exact-lane-reads", action="store_true",
                        help="keep the release's per-lane reads (shows what WARP does with the unchanged kernels)")
    args = parser.parse_args()
    if hashlib.sha256(args.dlss.read_bytes()).hexdigest() != DLSS_SHA256:
        raise SystemExit("not the DLSS 310.7.0 DLL the setup is pinned to")
    if args.out.exists():
        raise SystemExit(f"{args.out} exists")
    shutil.copytree(args.release, args.out)
    header = args.out / "setup/kernels/rt/px_hlsl.h"
    text = header.read_text()
    if PER_LANE not in text:
        raise SystemExit("unexpected shader runtime header: not the 1.2.0 release?")
    if not args.exact_lane_reads:
        header.write_text(text.replace(PER_LANE, UNIFORM))
    subprocess.run([sys.executable, "setup/helixsr_setup.py", str(args.out.resolve()), "--dlss", str(args.dlss.resolve()),
                    "--yes", "--dxc", str(args.dxc.resolve())], cwd=args.out, check=True)
    for name in ("helixsr_weights.bin", "helixsr_kernels.pak"):
        print(name, hashlib.sha256((args.out / name).read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
