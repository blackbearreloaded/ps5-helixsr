#!/usr/bin/env python3
# Copyright (C) 2026 BlackBearReloaded
# SPDX-License-Identifier: GPL-3.0-or-later
"""Compare two sets of RGBA16F frames, frame by frame.

  compare_frames.py WIDTH HEIGHT DIR_A PATTERN_A DIR_B PATTERN_B [--frames N] [--png OUT.png] [--json]

A pattern is a file name with a frame number, such as %02d.rgba16f or pan-%02d.rgba16f.
The PSNR is that of the colour channels with the first set's largest value (at least 1) as
the peak, so that an HDR frame is not judged by absolute errors. Non-finite values are counted.
--png writes both sets and four times their difference side by side.
"""
import argparse
import json
from pathlib import Path
import struct
import zlib

import numpy as np


def load(directory, pattern, frame, width, height):
    data = np.fromfile(Path(directory) / (pattern % frame), "<f2")
    if data.size != width * height * 4:
        raise SystemExit(f"{Path(directory) / (pattern % frame)}: not {width}x{height} RGBA16F")
    return data.astype(np.float64).reshape(height, width, 4)


def compare(a, b):
    """PSNR in dB (inf when identical), largest difference, peak and non-finite count of b."""
    bad = int((~np.isfinite(b)).sum())
    x, y = np.nan_to_num(a[..., :3]), np.nan_to_num(b[..., :3], posinf=0.0, neginf=0.0)
    peak = max(1.0, float(x.max()))
    difference = x - y
    rmse = float(np.sqrt((difference * difference).mean()))
    return dict(psnr=float("inf") if not rmse else float(20 * np.log10(peak / rmse)),
                max=float(np.abs(difference).max()), peak=peak, nonfinite=bad)


def write_png(path, image):
    image = (np.clip(image, 0, 1) ** (1 / 2.2) * 255 + 0.5).astype(np.uint8)
    raw = b"".join(b"\0" + row.tobytes() for row in image)

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    Path(path).write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", image.shape[1], image.shape[0],
                                                                             8, 2, 0, 0, 0)) +
                           chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("width", type=int)
    parser.add_argument("height", type=int)
    parser.add_argument("dir_a")
    parser.add_argument("pattern_a")
    parser.add_argument("dir_b")
    parser.add_argument("pattern_b")
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--png", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    rows, strips = [], []
    for frame in range(args.frames):
        a = load(args.dir_a, args.pattern_a, frame, args.width, args.height)
        b = load(args.dir_b, args.pattern_b, frame, args.width, args.height)
        rows.append(dict(frame=frame, **compare(a, b)))
        if args.png:
            gap = np.ones((args.height, 1, 3))
            scale = 1.0 / rows[-1]["peak"]
            strip = np.concatenate([a[..., :3] * scale, gap, b[..., :3] * scale, gap,
                                    np.abs(a[..., :3] - b[..., :3]) * scale * 4], 1)
            strips += [strip, np.ones((1, strip.shape[1], 3))]
    if args.png:
        zoom = max(1, 384 // args.width)
        write_png(args.png, np.concatenate(strips, 0).repeat(zoom, 0).repeat(zoom, 1))
    if args.json:
        print(json.dumps(rows))
    else:
        for row in rows:
            print(f"frame {row['frame']}: psnr {row['psnr']:.2f} dB  max {row['max']:.4f}  peak {row['peak']:.2f}"
                  f"  non-finite {row['nonfinite']}")


if __name__ == "__main__":
    main()
