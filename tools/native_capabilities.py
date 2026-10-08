#!/usr/bin/env python3
"""Audit the complete shipping shader set and emit its native device contract."""
import argparse
import hashlib
import json
import struct
from pathlib import Path


CAPABILITIES = {
    1: "Shader",
    9: "Float16",
    11: "Int64",
    22: "Int16",
    49: "StorageImageExtendedFormats",
    61: "GroupNonUniform",
    62: "GroupNonUniformVote",
    63: "GroupNonUniformArithmetic",
    64: "GroupNonUniformBallot",
    65: "GroupNonUniformShuffle",
    66: "GroupNonUniformShuffleRelative",
    67: "GroupNonUniformClustered",
    68: "GroupNonUniformQuad",
    4433: "StorageBuffer16BitAccess",
    5350: "ComputeDerivativeGroupLinearKHR",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capabilities(path):
    data = path.read_bytes()
    require(len(data) >= 20 and len(data) % 4 == 0, f"invalid SPIR-V: {path}")
    words = struct.unpack("<%dI" % (len(data) // 4), data)
    require(words[0] == 0x07230203, f"invalid SPIR-V magic: {path}")
    result = set()
    offset = 5
    while offset < len(words):
        count, opcode = words[offset] >> 16, words[offset] & 0xffff
        require(count and offset + count <= len(words), f"invalid instruction: {path}")
        if opcode == 17:
            require(count == 2, f"invalid OpCapability: {path}")
            result.add(words[offset + 1])
        offset += count
    require(offset == len(words), f"trailing SPIR-V words: {path}")
    return sorted(result)


def native_record(item):
    records = item.get("native")
    require(isinstance(records, list) and len(records) == 2,
            f"missing native records: {item.get('name')}")
    compiled = next((x for x in records if "result" in x), None)
    adapter = next((x for x in records if "adapter_result" in x), None)
    require(compiled and adapter, f"incomplete native records: {item['name']}")
    return compiled, adapter


def audit(generated_inventory, generated_dir, compact_inventory, compact_dir):
    generated = json.loads(generated_inventory.read_text())
    compact = json.loads(compact_inventory.read_text())
    require(generated.get("schema") == compact.get("schema") == 1,
            "unsupported inventory schema")
    shaders = []
    compiler_hashes = {generated.get("compiler_sha256"), compact.get("compiler_sha256")}
    require(len(compiler_hashes) == 1 and None not in compiler_hashes,
            "native compiler identity differs between inventories")

    for item in generated["kernels"]:
        if item["name"] == "clear_buffer":
            require(item.get("implementation") == "vkCmdFillBuffer",
                    "rejected clear shader is still in the runtime")
            continue
        require(item.get("success") is True and item.get("required_subgroup_size") == 32,
                f"generated shader is not shipping-ready: {item['name']}")
        compiled, adapter = native_record(item)
        path = generated_dir / item["name"] / (item["name"] + ".spv")
        shaders.append((item, compiled, adapter, path, len(item["bindings"]), "generated"))

    for item in compact["shaders"]:
        compiled, adapter = item.get("compile", {}), item.get("adapter", {})
        path = compact_dir / (item["name"] + ".spv")
        shaders.append((item, compiled, adapter, path, len(item["descriptors"]), "compact"))

    require(len(shaders) == 19 and len({x[0]["name"] for x in shaders}) == 19,
            "shipping inventory must contain 19 unique modules")
    rows, union = [], set()
    for item, compiled, adapter, path, descriptor_count, source in shaders:
        name = item["name"]
        require(path.is_file() and sha(path) == item.get("spirv_sha256"),
                f"SPIR-V identity mismatch: {name}")
        require(compiled.get("result") == "success" and adapter.get("adapter_result") == 0,
                f"native compilation failed: {name}")
        require(compiled.get("code_bytes") == adapter.get("adapter_code_bytes") > 0,
                f"native code size mismatch: {name}")
        require(compiled.get("descriptors") == descriptor_count and
                0 < adapter.get("adapter_descriptors", 0) <= descriptor_count,
                f"descriptor count mismatch: {name}")
        require(compiled.get("used_sets") == 1 and adapter.get("adapter_push_bytes") == 0,
                f"unexpected native ABI: {name}")
        require(adapter.get("wave_size") == 32 and adapter.get("scratch") == 0 and
                compiled.get("scratch_bytes_per_wave") == 0 and
                compiled.get("scratch_bytes_per_thread") == 0,
                f"unexpected native execution resources: {name}")
        caps = capabilities(path)
        union.update(caps)
        rows.append({
            "name": name,
            "source": source,
            "spirv_sha256": item["spirv_sha256"],
            "spirv_bytes": path.stat().st_size,
            "native_code_bytes": adapter["adapter_code_bytes"],
            "descriptors": descriptor_count,
            "native_used_descriptors": adapter["adapter_descriptors"],
            "wave_size": adapter["wave_size"],
            "vgprs": adapter["vgprs"],
            "user_sgprs": adapter["user_sgprs"],
            "lds_bytes": adapter["lds_size"],
            "scratch_bytes": adapter["scratch"],
            "capabilities": [CAPABILITIES.get(x, f"Capability{x}") for x in caps],
            "compiler_warnings": bool(item.get("capability_warnings")),
        })

    names = {CAPABILITIES.get(x, f"Capability{x}") for x in union}
    required = {
        "shaderInt16": "Int16" in names,
        "shaderInt64": "Int64" in names,
        "shaderFloat16": "Float16" in names,
        "storageBuffer16BitAccess": "StorageBuffer16BitAccess" in names,
        "shaderStorageImageExtendedFormats": "StorageImageExtendedFormats" in names,
        "computeDerivativeGroupLinear": "ComputeDerivativeGroupLinearKHR" in names,
        "subgroupSizeControl": True,
        "requiredSubgroupSizeStagesCompute": True,
        "requiredSubgroupSize": 32,
        "subgroupOperations": sorted(x for x in names if x.startswith("GroupNonUniform")),
    }
    require(all(required[x] for x in (
        "shaderInt16", "shaderInt64", "shaderFloat16",
        "shaderStorageImageExtendedFormats", "computeDerivativeGroupLinear",
        "subgroupSizeControl", "requiredSubgroupSizeStagesCompute")),
        "required Vulkan feature accounting is incomplete")
    return {
        "schema": 1,
        "compiler_sha256": next(iter(compiler_hashes)),
        "shader_count": len(rows),
        "generated_count": sum(x["source"] == "generated" for x in rows),
        "compact_count": sum(x["source"] == "compact" for x in rows),
        "total_spirv_bytes": sum(x["spirv_bytes"] for x in rows),
        "total_native_code_bytes": sum(x["native_code_bytes"] for x in rows),
        "maximums": {
            "descriptors": max(x["descriptors"] for x in rows),
            "native_used_descriptors": max(x["native_used_descriptors"] for x in rows),
            "vgprs": max(x["vgprs"] for x in rows),
            "user_sgprs": max(x["user_sgprs"] for x in rows),
            "lds_bytes": max(x["lds_bytes"] for x in rows),
            "scratch_bytes": max(x["scratch_bytes"] for x in rows),
        },
        "spirv_capabilities": sorted(names),
        "required_vulkan_features": required,
        "shaders": sorted(rows, key=lambda x: x["name"]),
        "clear_operation": "vkCmdFillBuffer",
        "scope": "offline SPIR-V identity/capability and PS5 native compiler resource audit; no device execution",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-inventory", type=Path, required=True)
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--compact-inventory", type=Path, required=True)
    parser.add_argument("--compact-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.generated_inventory, args.generated_dir,
                   args.compact_inventory, args.compact_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in (
        "shader_count", "total_spirv_bytes", "total_native_code_bytes", "maximums")},
        sort_keys=True))


if __name__ == "__main__":
    main()
