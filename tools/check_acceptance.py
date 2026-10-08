#!/usr/bin/env python3
# Copyright (C) 2026 BlackBearReloaded
# SPDX-License-Identifier: GPL-3.0-or-later
"""Check a PS5 run of the frame runner against the original HelixSR and the host control.

  check_acceptance.py REFERENCE HOST PS5_RESULTS [--case NAME] [--frames N] [--json OUT]

REFERENCE     a frame-input directory after tools/reference_probe.py (inputs.json, out/NN.rgba16f):
              the original HelixSR DLL on WARP
HOST          the output directory of build/helixsr_host_frames for the same inputs (out/NN.rgba16f):
              this runtime on a conformant desktop driver
PS5_RESULTS   the frame runner's result folder fetched from the console (NAME-NN.rgba16f)

A case is accepted when, on every frame, the PS5's output holds no non-finite value and its
PSNR against the original is at least the host control's minus 1 dB (VALIDATION.md).
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_frames import compare, load  # noqa: E402

MARGIN_DB = 1.0


def check(reference, host, results, name, frames=None):
    inputs = json.loads((reference / "inputs.json").read_text())
    width, height = inputs["output"]
    rows = []
    for frame in range(frames or inputs["frames"]):
        original = load(reference / "out", "%02d.rgba16f", frame, width, height)
        control = load(host / "out", "%02d.rgba16f", frame, width, height)
        console = load(results, name + "-%02d.rgba16f", frame, width, height)
        ps5, lvp, between = compare(original, console), compare(original, control), compare(control, console)
        rows.append(dict(frame=frame, ps5_db=ps5["psnr"], host_db=lvp["psnr"], ps5_host_db=between["psnr"],
                         nonfinite=ps5["nonfinite"],
                         accepted=not ps5["nonfinite"] and ps5["psnr"] >= lvp["psnr"] - MARGIN_DB))
    return dict(case=name, scenario=inputs["scenario"], render=inputs["render"], output=inputs["output"],
                frames=rows, accepted=all(row["accepted"] for row in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reference", type=Path)
    parser.add_argument("host", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("--case", help="the case's name in the runner (default: the reference directory's)")
    parser.add_argument("--frames", type=int)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    result = check(args.reference, args.host, args.results, args.case or args.reference.name, args.frames)
    if args.json:
        args.json.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{result['case']}: {result['scenario']} {result['render'][0]}x{result['render'][1]} -> "
          f"{result['output'][0]}x{result['output'][1]}: {'ACCEPTED' if result['accepted'] else 'REJECTED'}")
    for row in result["frames"]:
        print(f"  frame {row['frame']}: PS5 {row['ps5_db']:.2f} dB, host {row['host_db']:.2f} dB against the original; "
              f"PS5 against host {row['ps5_host_db']:.2f} dB{'' if row['accepted'] else '  <-- fails'}")
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    sys.exit(main())
