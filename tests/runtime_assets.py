"""Check generated runtime assets and fail-closed identity behavior."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


tool, smoke, inventory, generated, compact_inventory, compact_dir, header, source, receipt = sys.argv[1:]
run = subprocess.run([sys.executable, tool, "--inventory", inventory, "--generated", generated,
    "--compact-inventory", compact_inventory, "--compact-dir", compact_dir,
    "--header", header, "--source", source, "--receipt", receipt], capture_output=True, text=True)
assert run.returncode == 0, run.stderr
made = json.loads(run.stdout)
assert made["assets"] == 19 and made["bytes"] > 0
result = subprocess.run([smoke], capture_output=True, text=True)
assert result.returncode == 0, result.stderr
checked = json.loads(result.stdout)
assert checked == {"result": "success", "assets": 19, "bytes": made["bytes"]}
report = json.loads(Path(receipt).read_text())
assert report["asset_count"] == 19 and report["total_spirv_bytes"] == made["bytes"]
assert not report["payloads_committed"] and len({x["name"] for x in report["shaders"]}) == 19

with tempfile.TemporaryDirectory() as temp:
    broken = Path(temp) / "generated"
    shutil.copytree(generated, broken)
    victim = next(broken.rglob("*.spv"))
    data = bytearray(victim.read_bytes()); data[-1] ^= 1; victim.write_bytes(data)
    failed = subprocess.run([sys.executable, tool, "--inventory", inventory, "--generated", str(broken),
        "--compact-inventory", compact_inventory, "--compact-dir", compact_dir,
        "--header", str(Path(temp) / "x.h"), "--source", str(Path(temp) / "x.cpp"),
        "--receipt", str(Path(temp) / "x.json")], capture_output=True, text=True)
    assert failed.returncode != 0 and "identity mismatch" in failed.stderr
print("19 external shaders are hash-checked, embedded and discoverable")
