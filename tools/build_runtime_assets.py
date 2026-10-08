"""Verify external SPIR-V and emit an embeddable HelixSR shader table.

The generated files belong in the ignored build tree.  The repository stores
only hashes and descriptor metadata, never extracted/generated shader payloads.
"""
import argparse
import ast
import hashlib
import json
import re
import struct
from pathlib import Path


KIND = {
    "uniform-buffer": "U",
    "buffer": "B",
    "sampled-image": "I",
    "storage-image": "W",
    "sampler": "S",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def identifier(name):
    value = re.sub(r"[^A-Za-z0-9_]", "_", name)
    require(value and not value[0].isdigit(), "shader name is not a C identifier")
    return value


def load(path, expected):
    data = path.read_bytes()
    require(len(data) >= 20 and len(data) % 4 == 0, f"invalid SPIR-V size: {path}")
    require(struct.unpack_from("<I", data)[0] == 0x07230203, f"invalid SPIR-V magic: {path}")
    require(digest(data) == expected, f"SPIR-V identity mismatch: {path}")
    return data


def literal(path, name):
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(x, ast.Name) and x.id == name for x in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"missing source literal {name}")


def generated_descriptors(item, sampler_specs):
    result = []
    textures = {}
    for descriptor in item["bindings"]:
        if descriptor["kind"] == "sampled-image":
            match = re.fullmatch(r"PXT(\d+)", descriptor["name"])
            require(match, f"invalid texture name: {item['name']}")
            textures[match.group(1)] = descriptor
    raw = sampler_specs["k8"] if item["name"].startswith("k8") else sampler_specs.get(item["name"], "")
    samplers = {int(parts[0]): int(parts[1]) for parts in (x.split(":") for x in raw.split(",") if x)}
    for descriptor in item["bindings"]:
        parameter = descriptor.get("parameter_offset")
        image_binding = -1
        filtering = -1
        if descriptor["kind"] == "sampler":
            match = re.fullmatch(r"PXS(\d+)", descriptor["name"])
            require(match and match.group(1) in textures, f"sampler has no texture: {item['name']}")
            texture = textures[match.group(1)]
            image_binding = texture["binding"]
            filtering = samplers.get(texture["parameter_offset"], -1)
            require(filtering in (0, 1), f"missing sampler policy: {item['name']}")
        slot = -1
        match = re.fullmatch(r"PX[RW](\d+)", descriptor["name"])
        if descriptor["kind"] == "buffer":
            require(match, f"invalid buffer name: {item['name']}")
            slot = int(match.group(1))
        result.append({"kind": KIND[descriptor["kind"]], "parameter_offset": -1 if parameter is None else parameter,
                       "image_binding": image_binding, "linear": filtering, "resource_slot": slot})
    return result


def collect(inventory_path, generated, compact_inventory_path, compact, external=False):
    """external: the generated kernels are not read or embedded; the table carries their
    names, descriptors and SHA-256, and the application supplies them (tools/pack_kernels.py)."""
    inventory = json.loads(inventory_path.read_text())
    compact_inventory = json.loads(compact_inventory_path.read_text())
    require(inventory.get("schema") == 1, "unsupported generated inventory")
    require(compact_inventory.get("schema") == 1, "unsupported compact inventory")
    root = Path(__file__).resolve().parents[1]
    entries = literal(root / "third_party/helixsr/kernels.py", "ENTRY")
    sampler_specs = literal(root / "third_party/helixsr/samplers.py", "SAMPLERS")
    sampler_specs["k8"] = literal(root / "third_party/helixsr/samplers.py", "K8_SAMPLERS")
    assets = []
    for item in inventory["kernels"]:
        if item["name"] == "clear_buffer":
            require(item.get("implementation") == "vkCmdFillBuffer",
                    "clear_buffer replacement is not pinned")
            continue
        require(item.get("success") is True, f"generated shader did not compile: {item['name']}")
        kinds = [binding["kind"] for binding in item["bindings"]]
        assets.append((item["name"], item["spirv_sha256"], kinds,
                       None if external else generated / item["name"] / (item["name"] + ".spv"), 32,
                       entries[item["name"]], generated_descriptors(item, sampler_specs)))
    for item in compact_inventory["shaders"]:
        require(item.get("compile", {}).get("result") == "success",
                f"compact shader did not compile: {item['name']}")
        wave = item.get("adapter", {}).get("wave_size")
        descriptors = [{"kind": KIND[x], "parameter_offset": -1, "image_binding": -1,
                        "linear": -1, "resource_slot": -1}
                       for x in item["descriptors"]]
        assets.append((item["name"], item["spirv_sha256"], item["descriptors"],
                       compact / (item["name"] + ".spv"), wave, None, descriptors))
    require(len({x[0] for x in assets}) == len(assets), "duplicate shader name")
    require(len(assets) == 19, "unexpected shader inventory size")
    result = []
    for name, sha, kinds, path, wave, kernel, descriptors in sorted(assets):
        require(sha and len(sha) == 64, f"missing shader identity: {name}")
        require(all(x in KIND for x in kinds), f"unknown descriptor kind: {name}")
        require(wave == 32, f"shader is not native wave32: {name}")
        result.append({"name": name, "sha256": sha, "kinds": "".join(KIND[x] for x in kinds),
                       "data": load(path, sha) if path else b"", "path": str(path), "kernel": kernel,
                       "descriptors": descriptors})
    return result


def emit_header(path):
    path.write_text("""// Generated by build_runtime_assets.py. Do not commit.\n#pragma once\n#include <cstddef>\n#include <cstdint>\n\nnamespace ps5helixsr::assets {\nstruct Descriptor {\n  char kind;\n  int32_t parameter_offset;\n  int32_t image_binding;\n  int32_t linear_filter;\n  int32_t resource_slot;\n};\nstruct Shader {\n  const char* name;\n  const uint32_t* code;\n  size_t words;\n  const char* sha256;\n  const char* descriptor_kinds;\n  const char* kernel_name;\n  const Descriptor* descriptors;\n  size_t descriptor_count;\n  uint32_t required_subgroup_size;\n};\nconst Shader* find(const char* name);\nconst Shader* recipe_for_kernel(const char* name);\nconst Shader* data();\nsize_t size();\n} // namespace ps5helixsr::assets\n""")


def emit_source(path, header, assets):
    lines = ["// Generated by build_runtime_assets.py. Do not commit.",
             f'#include "{header.name}"', "#include <cstring>", "",
             "namespace ps5helixsr::assets {"]
    for item in assets:
        words = struct.unpack("<" + "I" * (len(item["data"]) // 4), item["data"])
        if words:
            lines.append(f"alignas(4) static const uint32_t code_{identifier(item['name'])}[] = {{")
            for offset in range(0, len(words), 8):
                lines.append("  " + ", ".join(f"0x{x:08x}u" for x in words[offset:offset + 8]) + ",")
            lines.append("};")
        lines.append(f"static const Descriptor descriptors_{identifier(item['name'])}[] = {{")
        for descriptor in item["descriptors"]:
            lines.append(f"  {{'{descriptor['kind']}', {descriptor['parameter_offset']}, "
                         f"{descriptor['image_binding']}, {descriptor['linear']}, "
                         f"{descriptor['resource_slot']}}},")
        lines.append("};")
    lines.extend(("", "static const Shader shaders[] = {"))
    for item in assets:
        ident = identifier(item["name"])
        kernel = "nullptr" if item["kernel"] is None else f'"{item["kernel"]}"'
        code = f"code_{ident}, sizeof(code_{ident}) / 4" if item["data"] else "nullptr, 0"
        lines.append(f'  {{"{item["name"]}", {code}, '
                     f'"{item["sha256"]}", "{item["kinds"]}", {kernel}, descriptors_{ident}, '
                     f'sizeof(descriptors_{ident}) / sizeof(descriptors_{ident}[0]), 32}},')
    lines.extend(("};", "", "const Shader* find(const char* name) {",
                  "  if (!name) return nullptr;",
                  "  for (const auto& shader : shaders)",
                  "    if (std::strcmp(shader.name, name) == 0) return &shader;",
                  "  return nullptr;", "}", "const Shader* recipe_for_kernel(const char* name) {",
                  "  if (!name) return nullptr;",
                  "  for (const auto& shader : shaders)",
                  "    if (shader.kernel_name && std::strcmp(shader.kernel_name, name) == 0) return &shader;",
                  "  return nullptr;", "}", "const Shader* data() { return shaders; }",
                  "size_t size() { return sizeof(shaders) / sizeof(shaders[0]); }",
                  "} // namespace ps5helixsr::assets", ""))
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--generated", type=Path)
    parser.add_argument("--external-kernels", action="store_true",
                        help="leave the generated kernels out: the application supplies them from a file")
    parser.add_argument("--compact-inventory", type=Path, required=True)
    parser.add_argument("--compact-dir", type=Path, required=True)
    parser.add_argument("--header", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if not args.external_kernels and not args.generated:
        parser.error("--generated is required unless --external-kernels")
    assets = collect(args.inventory, args.generated, args.compact_inventory, args.compact_dir, args.external_kernels)
    args.header.parent.mkdir(parents=True, exist_ok=True)
    args.source.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    emit_header(args.header)
    emit_source(args.source, args.header, assets)
    receipt = {"schema": 1, "asset_count": len(assets),
               "total_spirv_bytes": sum(len(x["data"]) for x in assets),
               "shaders": [{k: x[k] for k in ("name", "sha256", "kinds")} |
                           {"bytes": len(x["data"]), "required_subgroup_size": 32}
                           for x in assets],
               "payloads_committed": False}
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"assets": len(assets), "bytes": receipt["total_spirv_bytes"]}, sort_keys=True))


if __name__ == "__main__":
    main()
