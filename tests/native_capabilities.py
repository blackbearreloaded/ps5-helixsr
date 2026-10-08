#!/usr/bin/env python3
"""Exercise the native capability audit and its drift failures."""
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path


def expect_failure(fn, text):
    try:
        fn()
    except ValueError as error:
        assert text in str(error), error
    else:
        raise AssertionError(f"expected failure containing {text!r}")


def main():
    tool, gi, gd, ci, cd, receipt = map(Path, sys.argv[1:])
    spec = importlib.util.spec_from_file_location("native_capabilities", tool)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.audit(gi, gd, ci, cd)
    assert result["shader_count"] == 19
    assert result["generated_count"] == 13 and result["compact_count"] == 6
    assert result["maximums"] == {
        "descriptors": 18, "native_used_descriptors": 18,
        "vgprs": 256, "user_sgprs": 3,
        "lds_bytes": 64, "scratch_bytes": 0,
    }
    assert result["required_vulkan_features"]["requiredSubgroupSize"] == 32
    assert result["clear_operation"] == "vkCmdFillBuffer"
    assert all(x["wave_size"] == 32 and x["scratch_bytes"] == 0
               for x in result["shaders"])

    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        bad_inventory = folder / "generated.json"
        data = json.loads(gi.read_text())
        next(x for x in data["kernels"] if x["name"] != "clear_buffer")["native"][1]["wave_size"] = 64
        bad_inventory.write_text(json.dumps(data))
        expect_failure(lambda: module.audit(bad_inventory, gd, ci, cd), "execution resources")

        copied = folder / "compact"
        shutil.copytree(cd, copied)
        victim = next(copied.glob("*.spv"))
        altered = bytearray(victim.read_bytes())
        altered[-1] ^= 1
        victim.write_bytes(altered)
        expect_failure(lambda: module.audit(gi, gd, ci, copied), "identity mismatch")

    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"shaders": result["shader_count"], "failures": 2}))


if __name__ == "__main__":
    main()
