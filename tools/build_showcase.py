#!/usr/bin/env python3
# Copyright (C) 2026 BlackBearReloaded
# SPDX-License-Identifier: GPL-3.0-or-later
"""Build the PS5 HelixSR Showcase app (PPSA99013, examples/helixsr_showcase).

The HUD font and the city's signs are rasterized here from DejaVu Sans
(fonts-dejavu-core) with Pillow. Launch assets come from the example's sce_sys
where present, else from the native app template.

Neither the network's weights nor its kernels are part of the app: the build
reads nothing made from NVIDIA's DLL, and the folder it writes can be shared.
The app looks for model.bin and kernels.bin in its assets folder (make
network-files makes both on your machine). --model and --kernels copy local
files in for your own console; a folder that holds them must not be shared.

--host builds an off-screen binary for desktop Vulkan instead, through the
CMake build tree (build/helixsr_showcase): it runs the scripted walk and saves
a frame of each step (examples/helixsr_showcase/README.md).
"""
import argparse
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from build_runner import MODEL_SHA256, build_title, runtime_assets, sha, toolchain_arguments  # noqa: E402
from showcase_signs import build_signs, font_path  # noqa: E402

TITLE_ID = "PPSA99013"
TITLE_NAME = "PS5 HelixSR Showcase"
CONTENT_ID = "UP9000-PPSA99013_00-HELIXSRSHOWCASE0"
SOURCE = ROOT / "examples/helixsr_showcase"
# (name in the header, source, extra glslang arguments)
SHADERS = (("scene", SOURCE / "city.comp", ()),
           ("native", SOURCE / "city.comp", ("-DSCENE_COLOR_ONLY",)),
           ("compose", SOURCE / "compose.comp", ()),
           ("blit_vert", SOURCE / "blit.vert", ()),
           ("blit_frag", SOURCE / "blit.frag", ()))
# HUD faces: (name, font file, pixel size); hud.c expects these three.
FACES = (("title", "DejaVuSans-Bold.ttf", 38), ("body", "DejaVuSans.ttf", 25), ("small", "DejaVuSans.ttf", 19))
ATLAS_W = 1024
LOUDNESS_SND0 = "-28.00"  # what ps5-at9-converter normalized snd0.at9 to


def compile_shaders(out, glslang):
    header = ["#include <stdint.h>"]
    for name, source, extra in SHADERS:
        spv = out / f"helixsr_showcase_{name}.spv"
        subprocess.run([glslang, "-V", "--target-env", "vulkan1.1", *extra, str(source), "-o", str(spv)],
                       check=True, stdout=subprocess.DEVNULL)
        words = struct.unpack(f"<{spv.stat().st_size // 4}I", spv.read_bytes())
        header.append(f"static const uint32_t helixsr_showcase_{name}_spv[] = {{" + ",".join(map(hex, words)) + "};")
    (out / "helixsr_showcase_shaders.h").write_text("\n".join(header) + "\n")


def build_font(out):
    """helixsr_showcase_font.h: an 8-bit coverage atlas of every character the app draws."""
    from PIL import Image, ImageDraw, ImageFont
    text = "".join(p.read_text(encoding="utf-8") for p in sorted(SOURCE.glob("*.c")))
    codepoints = sorted(set(range(32, 127)) | {ord(c) for c in text if ord(c) > 127})

    def render(font, char):
        left, top, right, bottom = font.getbbox(char)  # from the origin at the ascent line
        image = Image.new("L", (max(right - left, 0), max(bottom - top, 0)))
        if image.width and image.height:
            ImageDraw.Draw(image).text((-left, -top), char, font=font, fill=255)
        return left, top, image

    glyphs, fonts = [], []
    for face, file, size in FACES:
        font = ImageFont.truetype(str(font_path(file)), size)
        _, _, missing = render(font, "\U0010fffd")  # the .notdef box
        ascent, descent = font.getmetrics()
        fonts.append((face, ascent + descent + 2, len(glyphs)))
        for codepoint in codepoints:
            char = chr(codepoint)
            left, top, image = render(font, char)
            if codepoint > 127 and image.tobytes() == missing.tobytes():
                raise SystemExit(f"{file} has no glyph for U+{codepoint:04X}")
            glyphs.append(dict(codepoint=codepoint, image=image, xoff=left, yoff=top,
                               advance=round(font.getlength(char))))
    # Shelf packing, tallest first.
    x = y = shelf = 0
    for glyph in sorted(glyphs, key=lambda g: -g["image"].height):
        w, h = glyph["image"].size
        if x + w > ATLAS_W:
            x, y, shelf = 0, y + shelf + 1, 0
        glyph["x"], glyph["y"] = x, y
        x, shelf = x + w + 1, max(shelf, h)
    atlas = Image.new("L", (ATLAS_W, y + shelf))
    for glyph in glyphs:
        atlas.paste(glyph["image"], (glyph["x"], glyph["y"]))
    lines = ["#include <stdint.h>",
             f"enum {{ HUD_ATLAS_W = {ATLAS_W}, HUD_ATLAS_H = {atlas.height} }};",
             "struct hud_glyph { uint32_t codepoint; uint16_t x, y, w, h; int16_t xoff, yoff, advance; };",
             "struct hud_font { int line_height; int count; const struct hud_glyph *glyphs; };",
             "static const uint8_t hud_atlas[] = {" + ",".join(map(str, atlas.tobytes())) + "};"]
    for face, line_height, first in fonts:
        members = glyphs[first:first + len(codepoints)]
        lines.append(f"static const struct hud_glyph hud_glyphs_{face}[] = {{" + ",".join(
            "{%d,%d,%d,%d,%d,%d,%d,%d}" % (g["codepoint"], g["x"], g["y"], *g["image"].size, g["xoff"], g["yoff"],
                                           g["advance"]) for g in members) + "};")
        lines.append(f"static const struct hud_font hud_font_{face} = {{{line_height}, {len(members)}, "
                     f"hud_glyphs_{face}}};")
    (out / "helixsr_showcase_font.h").write_text("\n".join(lines) + "\n")
    return dict(glyphs=len(codepoints), atlas=f"{ATLAS_W}x{atlas.height}")


def content_version(version):
    """What the console reports for an installed build: a PlayStation content version
    (01.000.000) is carried as it is; any other build carries 01.000.000."""
    return version if version and re.fullmatch(r"\d\d\.\d\d\d\.\d\d\d", version) else "01.000.000"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, help="Default: build/showcase, or build/showcase-host with --host")
    parser.add_argument("--host", action="store_true", help="Build the off-screen desktop binary")
    parser.add_argument("--model", type=Path, help="Local Model E weights to put into the folder (never shared)")
    parser.add_argument("--kernels", type=Path,
                        help="Local kernels.bin (tools/pack_kernels.py) to put into the folder (never shared)")
    parser.add_argument("--selftest", action="store_true",
                        help="Replace the pad with a scripted walk that logs timings, saves frames and exits")
    parser.add_argument("--keys", metavar="LETTERS",
                        help="Play these button presses through the pad's path, save the last frame and exit "
                             "(the letters are in main.c; the host build reads SHOWCASE_KEYS instead)")
    parser.add_argument("--version", help="The version the app shows, and its content version when it is one")
    parser.add_argument("--define", action="append", default=[], metavar="NAME=VALUE", help=argparse.SUPPRESS)
    parser.add_argument("--glslang", default="glslangValidator")
    toolchain_arguments(parser)
    args = parser.parse_args()
    if args.model and sha(args.model) != MODEL_SHA256:
        raise SystemExit("canonical model identity mismatch")
    out = (args.out or ROOT / ("build/showcase-host" if args.host else "build/showcase")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    compile_shaders(out, args.glslang)
    font = build_font(out)
    package = out / TITLE_ID
    if package.exists() and not args.host:
        shutil.rmtree(package)
    assets = out / "assets" if args.host else package / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    (assets / "signs.bin").write_bytes(build_signs().tobytes())
    if args.model:
        shutil.copyfile(args.model, assets / "model.bin")
    if args.kernels and not args.host:
        shutil.copyfile(args.kernels, assets / "kernels.bin")
    defines = [*(["SHOWCASE_SELFTEST=1"] if args.selftest else []), *args.define,
               *([f'SHOWCASE_KEYS="{args.keys}"'] if args.keys and not args.host else []),
               *([f'SHOWCASE_VERSION="{args.version}"'] if args.version else [])]
    if args.host:
        build = ROOT / "build"
        subprocess.run(["cmake", "-S", str(ROOT), "-B", str(build), f"-DHELIXSR_SHOWCASE_GENERATED={out}",
                        "-DHELIXSR_SHOWCASE_DEFINES=" + ";".join(defines)], check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["cmake", "--build", str(build), "--target", "helixsr_showcase", "-j8"], check=True,
                       stdout=subprocess.DEVNULL)
        print(json.dumps(dict(binary=str(build / "helixsr_showcase"), run_in=str(out), font=font,
                              model=bool(args.model)), indent=2))
        return
    own_assets = sorted(p.name for p in (SOURCE / "sce_sys").glob("*") if p.is_file())
    args.title_id = TITLE_ID
    build_title(args, out, package, runtime_assets(args, out, external_kernels=True),
                [SOURCE / "main.c", SOURCE / "hud.c"], TITLE_NAME,
                CONTENT_ID, include_dirs=[SOURCE, out], defines=defines, sce_sys=SOURCE / "sce_sys",
                param_overrides=dict(contentVersion=content_version(args.version),
                                     pubtools=dict(loudnessSnd0=LOUDNESS_SND0) if "snd0.at9" in own_assets else {}))
    print(json.dumps(dict(package=str(package), model=bool(args.model), kernels=bool(args.kernels), selftest=args.selftest,
                          launch_assets=own_assets, font=font, eboot_sha256=sha(package / "eboot.bin")), indent=2))


if __name__ == "__main__":
    main()
