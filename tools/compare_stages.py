#!/usr/bin/env python3
"""Compare per-stage resource dumps of the runtime (tools/host_frames.cpp --dump-frame)
with the original HelixSR's (tools/reference_probe.py --dump-frame), launch by launch.

The original creates its network resources in the planner's table order, which is how
the two sets are paired. Prints, for every launch, each resource whose bytes differ."""
import argparse
from pathlib import Path
import re

import numpy as np


def reference_resources(probe_text):
    """Planner buffers and images among the DLL's resources, in creation order."""
    rows = [dict(zip(("index", "dimension", "width", "height", "format", "heap"), map(int, m.groups())))
            for m in re.finditer(r"resource: index=(\d+) dimension=(\d+) width=(\d+) height=(\d+) format=(\d+) heap=(\d+)",
                                 probe_text)]
    buffers = [r for r in rows if r["dimension"] == 1 and r["heap"] == 1]
    images = [r for r in rows if r["dimension"] == 3]
    return rows, buffers, images


def snr(reference, value):
    """Signal to difference ratio in dB: numerical noise sits above 40, a wrong result near or below 0."""
    noise = np.sqrt(np.mean((reference - value) ** 2))
    signal = np.sqrt(np.mean(reference ** 2))
    if not noise:
        return "exact"
    return f"{20 * np.log10(signal / noise):.1f} dB" if signal else "reference is zero"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="reference frames directory (probe.txt, dump/)")
    parser.add_argument("runtime", type=Path, help="host_frames output directory (dump/)")
    parser.add_argument("--all", action="store_true", help="also list identical resources")
    args = parser.parse_args()
    steps = [line.split() for line in (args.runtime / "dump/steps.txt").read_text().splitlines()]
    last_step = {}
    for step, launch, name, *_ in steps:
        last_step[int(launch)] = (int(step), name)
    ours = sorted(p.stem for p in (args.runtime / "dump/00").glob("*.bin"))
    rows, buffers, images = reference_resources((args.reference / "probe.txt").read_text())
    sizes = {name: (args.runtime / "dump/00" / f"{name}.bin").stat().st_size for name in ours}
    # Pair by table order: the DLL's first default-heap unordered-access buffers are the planner's buffers.
    planner_buffers = [n for n in ("LUMA0", "LUMA1", "NET_IN") if n in sizes] + \
        sorted((n for n in ours if re.match(r"B\d+_", n)), key=lambda n: int(n[1:n.index("_")]))
    pairs = {}
    candidates = [r for r in buffers]
    for name in planner_buffers:
        for row in candidates:
            if row["width"] == sizes[name]:
                pairs[name] = row["index"]
                candidates = candidates[candidates.index(row) + 1:]
                break
    image_names = [n for n in ours if n not in planner_buffers and not n.startswith("TRANSIENT")]
    reference_dir = args.reference / "dump"
    for name in image_names:
        for row in images:
            path = reference_dir / "00" / f"r{row['index']:02d}.bin"
            if row["index"] not in pairs.values() and path.stat().st_size == sizes[name] and row["index"] >= 15:
                pairs[name] = row["index"]
                break
    print("pairs:", {k: f"r{v:02d}" for k, v in pairs.items()})
    launches = sorted(int(p.name) for p in reference_dir.iterdir())
    previous = {}
    for launch in launches:
        if launch not in last_step:
            print(f"launch {launch:2d}: not in the runtime's plan")
            continue
        step, name = last_step[launch]
        lines = []
        for resource, index in pairs.items():
            a = np.fromfile(reference_dir / f"{launch:02d}" / f"r{index:02d}.bin", np.uint8)
            b = np.fromfile(args.runtime / "dump" / f"{step:02d}" / f"{resource}.bin", np.uint8)
            # what this launch wrote: the bytes either side changed since the launch before
            before = previous.get(resource)
            previous[resource] = (a, b)
            written = np.ones(a.size, bool) if before is None else (a != before[0]) | (b != before[1])
            if not written.any():
                continue
            if a.size % 2 == 0:
                written = np.repeat(written.reshape(-1, 2).any(1), 2)
            a, b = a[written], b[written]
            different = int((a != b).sum())
            if different or args.all:
                # as bytes (E5M3 codes) and as halves: how far apart, not only how many
                byte_step = np.abs(a.astype(int) - b.astype(int))
                detail = f"bytes: {int((byte_step > 1).sum())} beyond one code, max {int(byte_step.max())}"
                if a.size % 2 == 0:
                    ha, hb = a.view("<u2").astype(int), b.view("<u2").astype(int)
                    fa, fb = a.view("<f2").astype(np.float64), b.view("<f2").astype(np.float64)
                    finite = np.isfinite(fa) & np.isfinite(fb)
                    delta = np.abs(fa - fb)[finite]
                    detail += (f"; halves: {int((np.abs(ha - hb) > 1).sum())} beyond one step, "
                               f"max abs {delta.max() if delta.size else 0:.4g}, {snr(fa[finite], fb[finite])}")
                # E5M3 codes decode as half(byte << 7)
                ea = (a.astype(np.uint16) << 7).view("<f2").astype(np.float64)
                eb = (b.astype(np.uint16) << 7).view("<f2").astype(np.float64)
                both = np.isfinite(ea) & np.isfinite(eb)
                detail += f"; as E5M3: {snr(ea[both], eb[both])}"
                lines.append(f"    {resource:34s} {different:7d} of {a.size:7d} bytes differ ({detail})")
        print(f"launch {launch:2d} (step {step:2d} {name}): {'identical' if not any('differ' in l and ' 0 of' not in l for l in lines) else ''}")
        for line in lines:
            if args.all or " 0 of" not in line:
                print(line)


if __name__ == "__main__":
    main()
