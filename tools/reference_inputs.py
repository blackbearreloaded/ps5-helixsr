#!/usr/bin/env python3
"""Deterministic frame inputs for the reference probe and the runtime runners.

An analytic scene (smooth texture, hard edges, small print-like detail, a moving disc
in front) is sampled at the jittered render pixel centres, with exact depth and motion.
A supersampled picture of the same scene at output size is written as scene truth: it
measures reconstruction quality, it is not the network's expected output.

Conventions (the FidelityFX ones the original HelixSR takes):
  jitter   in render pixels; a pixel samples the scene at its centre minus the jitter
  motion   previous position minus current position, in render pixels
  depth    reversed (near = 1)
Layout: frames.bin, in/NN/{color.rgba16f,depth.r32f,motion.rg16f}, truth/NN.rgba16f, inputs.json"""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

SCENARIOS = ("static", "pan", "object", "motion", "cut", "bright", "rows", "columns")


def halton(index, base):
    result, fraction = 0.0, 1.0 / base
    while index:
        result += (index % base) * fraction
        index //= base
        fraction /= base
    return result


def jitter(frame, phases):
    index = frame % phases + 1
    return halton(index, 2) - 0.5, halton(index, 3) - 0.5


def camera(scenario, frame):
    if scenario in ("pan", "motion"):
        return 0.0031 * frame, 0.0017 * frame
    if scenario == "cut":
        return (0.0 if frame < 8 else 0.37), 0.0
    return 0.0, 0.0


def disc(scenario, frame):
    if scenario in ("object", "motion"):
        return 0.30 + 0.0113 * frame, 0.55 - 0.0041 * frame
    return 0.30, 0.55


def background(u, v):
    """Smooth, band-limited colour plus hard-edged and fine detail; u, v are world coordinates."""
    r = 0.35 + 0.25 * np.sin(7.1 * u + 1.3) * np.cos(5.3 * v)
    g = 0.30 + 0.20 * np.sin(11.7 * v + 0.4)
    b = 0.40 + 0.25 * np.cos(9.4 * (u + v))
    stripes = (np.floor(u * 37.0) + np.floor(v * 23.0)) % 2          # checker about 3-4 output pixels wide at 96x60
    fine = 0.5 + 0.5 * np.sin(2 * np.pi * 61.0 * (u * 0.8 + v * 0.6))  # diagonal lines near the render limit
    band = (np.abs(v - 0.25) < 0.07)
    slab = (np.abs(v - 0.80) < 0.06)
    r = np.where(band, 0.08 + 0.8 * stripes, r)
    g = np.where(band, 0.08 + 0.8 * stripes, g)
    b = np.where(band, 0.10 + 0.6 * stripes, b)
    r = np.where(slab, 0.1 + 0.7 * fine, r)
    g = np.where(slab, 0.1 + 0.5 * fine, g)
    b = np.where(slab, 0.2, b)
    edge = (u * 0.6 + v) % 0.5 < 0.012                                # thin slanted lines
    return np.stack([np.where(edge, 1.6, r), np.where(edge, 1.5, g), np.where(edge, 1.2, b)], -1)


def scene(scenario, frame, u, v):
    """Colour, reversed depth and foreground mask at screen coordinates u, v in [0, 1)."""
    cx, cy = camera(scenario, frame)
    dx, dy = disc(scenario, frame)
    color = background(u + cx, v + cy)
    front = (u - dx) ** 2 + ((v - dy) * 0.625) ** 2 < 0.085 ** 2
    spokes = np.sin(14.0 * np.arctan2(v - dy, u - dx)) > 0
    disc_color = np.stack([np.where(spokes, 1.0, 0.15), np.where(spokes, 0.35, 0.10), np.where(spokes, 0.05, 0.30)], -1)
    color = np.where(front[..., None], disc_color, color)
    if scenario == "bright":
        color = color * 6.0
    depth = np.where(front, 0.75, 0.25)
    return color, depth, front


def frame_data(scenario, frame, render, output, phases, supersample=4):
    rw, rh = render
    ow, oh = output
    jx, jy = jitter(frame, phases)
    y, x = np.mgrid[0:rh, 0:rw].astype(np.float64)
    u, v = (x + 0.5 - jx) / rw, (y + 0.5 - jy) / rh
    color, depth, front = scene(scenario, frame, u, v)
    if scenario in ("rows", "columns"):
        # diagnostic: the value identifies the source row (or column) and nothing else
        index = y if scenario == "rows" else x
        color = np.repeat((0.1 + 0.02 * index)[..., None], 3, -1)
        depth, front = np.full_like(depth, 0.25), np.zeros_like(front)
    cx, cy = camera(scenario, frame)
    px, py = camera(scenario, max(frame - 1, 0))
    dx, dy = disc(scenario, frame)
    qx, qy = disc(scenario, max(frame - 1, 0))
    reset = frame == 0 or (scenario == "cut" and frame == 8)
    # previous screen position minus current, render pixels
    mvx = np.where(front, (qx - dx) * rw, (cx - px) * rw)
    mvy = np.where(front, (qy - dy) * rh, (cy - py) * rh)
    if reset:
        mvx, mvy = np.zeros_like(mvx), np.zeros_like(mvy)
    s = supersample
    yy, xx = np.mgrid[0:oh * s, 0:ow * s].astype(np.float64)
    truth, _, _ = scene(scenario, frame, (xx + 0.5) / (ow * s), (yy + 0.5) / (oh * s))
    truth = truth.reshape(oh, s, ow, s, 3).mean((1, 3))
    rgba = np.concatenate([color, np.ones((rh, rw, 1))], -1)
    truth = np.concatenate([truth, np.ones((oh, ow, 1))], -1)
    return {"color.rgba16f": rgba.astype("<f2").tobytes(), "depth.r32f": depth.astype("<f4").tobytes(),
            "motion.rg16f": np.stack([mvx, mvy], -1).astype("<f2").tobytes()}, truth.astype("<f2").tobytes(), \
        (jx, jy, int(reset))


def generate(destination, scenario, render, output, frames, phases=None):
    if scenario not in SCENARIOS:
        raise ValueError("unknown scenario")
    destination = Path(destination)
    # FidelityFX's phase count: 8 * (output / render) ** 2
    phases = phases or int(round(8 * (output[0] / render[0]) ** 2))
    info, digest = b"", hashlib.sha256()
    for frame in range(frames):
        files, truth, (jx, jy, reset) = frame_data(scenario, frame, render, output, phases)
        directory = destination / "in" / f"{frame:02d}"
        directory.mkdir(parents=True, exist_ok=True)
        for name, data in files.items():
            (directory / name).write_bytes(data)
            digest.update(data)
        (destination / "truth").mkdir(exist_ok=True)
        (destination / "truth" / f"{frame:02d}.rgba16f").write_bytes(truth)
        info += struct.pack("<ffIf", jx, jy, reset, 1.0)
    (destination / "frames.bin").write_bytes(info)
    digest.update(info)
    manifest = dict(schema=1, scenario=scenario, render=list(render), output=list(output), frames=frames,
                    jitter_phases=phases, inputs_sha256=digest.hexdigest())
    (destination / "inputs.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--scenario", choices=SCENARIOS, default="motion")
    parser.add_argument("--render", type=int, nargs=2, default=(64, 40))
    parser.add_argument("--output", type=int, nargs=2, default=(96, 60))
    parser.add_argument("--frames", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(generate(args.destination, args.scenario, tuple(args.render), tuple(args.output), args.frames)))


if __name__ == "__main__":
    main()
