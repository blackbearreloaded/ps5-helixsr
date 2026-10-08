#!/usr/bin/env python3
# Copyright (C) 2026 BlackBearReloaded
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pack the network's kernels into the file an application gives the runtime.

  pack_kernels.py --generated build/main-inventory/generated --out kernels.bin

The kernels are the SPIR-V that tools/build_generated.py makes on your machine from
NVIDIA's DLSS DLL (make shaders). An application built without them (the showcase, a
release) reads this file beside model.bin and passes both to ps5helixsr_context_create,
which accepts a kernel only when its SHA-256 is the one this repository pins.
Like the weights, the file is NVIDIA's material: keep it on your own console.

Layout, little endian: "HXKP", version 1, count; count entries of name[48], offset, bytes;
the modules.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[1]
MAGIC, VERSION, NAME_BYTES = b"HXKP", 1, 48


def pack(inventory, generated):
    modules = []
    for item in json.loads(Path(inventory).read_text())["kernels"]:
        if item.get("implementation") == "vkCmdFillBuffer":
            continue  # the clear is a buffer fill in the runtime, not a kernel
        data = (Path(generated) / item["name"] / (item["name"] + ".spv")).read_bytes()
        if hashlib.sha256(data).hexdigest() != item["spirv_sha256"]:
            raise SystemExit(f"{item['name']}: not the module validation/generated-main.json pins; "
                             "this build of the kernels does not belong to this revision")
        modules.append((item["name"].encode(), data))
    header = 12 + len(modules) * (NAME_BYTES + 8)
    out, offset = [MAGIC + struct.pack("<II", VERSION, len(modules))], header
    for name, data in modules:
        if len(name) >= NAME_BYTES:
            raise SystemExit(f"kernel name too long: {name.decode()}")
        out.append(name.ljust(NAME_BYTES, b"\0") + struct.pack("<II", offset, len(data)))
        offset += len(data)
    return b"".join(out + [data for _, data in modules]), len(modules)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inventory", type=Path, default=ROOT / "validation/generated-main.json")
    parser.add_argument("--generated", type=Path, default=ROOT / "build/main-inventory/generated")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data, count = pack(args.inventory, args.generated)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(data)
    print(json.dumps(dict(out=str(args.out), kernels=count, bytes=len(data),
                          sha256=hashlib.sha256(data).hexdigest())))


if __name__ == "__main__":
    main()
