#!/usr/bin/env python3
# Copyright (C) 2026 BlackBearReloaded
# SPDX-License-Identifier: GPL-3.0-or-later
"""Draw the README's performance chart as SVG, in light and dark variants.

The numbers are PS5 measurements recorded below: HelixSR's come from the showcase's own
benchmark (examples/helixsr_showcase, the mean of 300 frames submitted back to back, each
timed from submission to completion on the GPU), FSR 4's from the README of ps5-fsr4 (its
headless benchmark, measured the same way). Update them here and rerun after a new measurement.
"""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (output, render, mode, HelixSR milliseconds per frame, PS5 FSR4 milliseconds per frame)
CASES = [
    ("1920×1080", "640×360", "Ultra Performance", 2.13, 2.92),
    ("1920×1080", "1280×720", "Quality", 2.14, 2.95),
    ("2560×1440", "1280×720", "Performance", 2.98, 4.92),
    ("2560×1440", "1706×960", "Quality", 3.01, 4.95),
    ("3840×2160", "1920×1080", "Performance", 5.64, 10.86),
    ("3840×2160", "1280×720", "Ultra Performance", 5.57, 10.94),
]
FRAME_60FPS = 1000 / 60

THEMES = {
    "light": dict(background="#ffffff", text="#1f2328", muted="#59636e", grid="#d1d9e0", reference="#cf222e",
                  other="#afb8c1", outputs={"1920×1080": "#0969da", "2560×1440": "#8250df", "3840×2160": "#bc4c00"}),
    "dark": dict(background="#0d1117", text="#f0f6fc", muted="#9198a1", grid="#3d444d", reference="#f85149",
                 other="#656c76", outputs={"1920×1080": "#4493f8", "2560×1440": "#ab7df8", "3840×2160": "#f0883e"}),
}
FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI','Noto Sans',Helvetica,Arial,sans-serif"


class Svg:
    def __init__(self, width, height, title):
        self.width, self.height, self.parts = width, height, []
        self.head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
                     f'viewBox="0 0 {width} {height}" role="img" aria-label="{title}" font-family="{FONT}">'
                     f'<title>{title}</title>')

    def text(self, x, y, content, size=12, fill="#000", anchor="start", weight="normal", halo=None):
        """`halo` outlines the glyphs in the page colour so lines behind them stay out of the way."""
        outline = (f' stroke="{halo}" stroke-width="4" stroke-linejoin="round" paint-order="stroke"'
                   if halo else "")
        self.parts.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
                          f'text-anchor="{anchor}" font-weight="{weight}"{outline}>{content}</text>')

    def line(self, x1, y1, x2, y2, stroke, width=1, dash=None):
        dashes = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                          f'stroke="{stroke}" stroke-width="{width}"{dashes}/>')

    def rect(self, x, y, w, h, fill):
        self.parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="3" fill="{fill}"/>')

    def source(self):
        return self.head + "".join(self.parts) + "</svg>\n"


def axis(svg, theme, left, right, top, bottom, limit, step, unit="ms"):
    """Vertical grid lines every `step` from 0 to `limit`, labelled under the chart."""
    scale = (right - left) / limit
    value = 0
    while value <= limit + 1e-9:
        x = left + value * scale
        svg.line(x, top, x, bottom, theme["grid"])
        svg.text(x, bottom + 16, f"{value:g}" + (f" {unit}" if value + step > limit else ""), 11,
                 theme["muted"], "middle")
        value += step
    return scale


def cases_chart(theme):
    width, left, right, row, gap = 820, 262, 770, 48, 12
    groups = len({output for output, *_ in CASES})
    top = 96
    bottom = top + row * len(CASES) + gap * (groups - 1)
    svg = Svg(width, bottom + 34, "HelixSR on the PS5: milliseconds per frame for each output and mode")
    svg.text(16, 26, "HelixSR on the PS5: time per frame", 17, theme["text"], weight="600")
    svg.text(16, 46, "The showcase's benchmark, milliseconds per frame (lower is better)", 12, theme["muted"])
    x = 16
    for output, color in theme["outputs"].items():
        svg.rect(x, 60, 12, 12, color)
        svg.text(x + 18, 70.5, f"{output} output", 12, theme["text"])
        x += 150
    svg.rect(x, 62, 12, 8, theme["other"])
    svg.text(x + 18, 70.5, "FSR 4 on the PS5 (ps5-fsr4)", 12, theme["muted"])
    limit = 18
    scale = axis(svg, theme, left, right, top - 8, bottom, limit, 2)
    budget = left + FRAME_60FPS * scale
    svg.line(budget, top - 8, budget, bottom, theme["reference"], 1.5, "5 4")
    svg.text(budget - 6, top - 14, "one 60 fps frame: 16.7 ms", 11, theme["reference"], "end")
    y, previous = top, CASES[0][0]
    for output, render, mode, ms, fsr4 in CASES:
        if output != previous:
            y += gap
            previous = output
        svg.text(left - 12, y + 15, f"{render} → {output}", 13, theme["text"], "end", "600")
        svg.text(left - 12, y + 30, mode, 11, theme["muted"], "end")
        svg.rect(left, y + 4, ms * scale, 20, theme["outputs"][output])
        svg.text(left + ms * scale + 6, y + 18.5, f"{ms:.2f} ms", 12, theme["text"], weight="600",
                 halo=theme["background"])
        svg.rect(left, y + 28, fsr4 * scale, 8, theme["other"])
        svg.text(left + fsr4 * scale + 6, y + 36, f"{fsr4:.2f}", 10.5, theme["muted"], halo=theme["background"])
        y += row
    return svg.source()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "docs/perf")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name, theme in THEMES.items():
        (args.out / f"cases-{name}.svg").write_text(cases_chart(theme), encoding="utf-8")
        print(args.out / f"cases-{name}.svg")


if __name__ == "__main__":
    main()
