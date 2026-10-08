"""Compare the C++ runtime planner with the checked Python binding oracle."""
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from prepare_bindings import prepare


planner, runtime, inventory_path, generated_path, compact_inventory_path, compact_dir, report = sys.argv[1:]
inventory = json.loads(Path(inventory_path).read_text())
compact_inventory = json.loads(Path(compact_inventory_path).read_text())
generated = Path(generated_path)
compact = Path(compact_dir)


def expected(shape, frame, exposure, dynamic=()):
    graph = json.loads(subprocess.check_output([planner, *map(str, shape), str(frame), str(exposure), *map(str, dynamic)]))
    contract = prepare(graph, inventory, generated, compact_inventory, compact)
    result = []
    for step in contract["steps"]:
        launch = step["launch_index"]
        if step["command"] == "vkCmdFillBuffer":
            result.append(("F", launch, f"buffer:{step['role']}", step["offset"], step["size"], step["data"]))
        elif step["command"] == "barrier":
            result.append(("B", launch, tuple(step["resources"])))
        else:
            descriptors = []
            uniform = step.get("uniform_hex", step["descriptors"][0].get("data_hex"))
            for descriptor in step["descriptors"]:
                kind = {"uniform-buffer": "U", "buffer": "B", "sampled-image": "I",
                        "storage-image": "W", "sampler": "S"}[descriptor["kind"]]
                if kind == "U":
                    resource, offset, span, access, filtering = "uniforms", 0, descriptor["range"], 1, -1
                elif kind == "S":
                    resource, offset, span, access = descriptor["filter"], 0, 0, 1
                    filtering = 1 if descriptor["filter"] == "linear" else 0
                else:
                    resource = descriptor["resource"]
                    offset, span = descriptor.get("offset", 0), descriptor.get("range", 0)
                    access = {"read": 1, "write": 2, "read-write": 3}.get(descriptor["access"], descriptor["access"])
                    filtering = -1
                descriptors.append((kind, resource, offset, span, access, filtering))
            result.append(("D", launch, step.get("shader", step.get("recipe")), tuple(step["grid"]),
                           uniform, tuple(descriptors)))
    transient = {x["resource"]: x["bytes"] for x in contract["transient_buffers"]}
    return (transient.get("transient:fused_fp16_a", 0), transient.get("transient:fused_fp16_b", 0)), result


def actual(shape, frame, exposure, dynamic=()):
    output = subprocess.check_output([runtime, *map(str, shape), str(frame), str(exposure), *map(str, dynamic)], text=True).splitlines()
    assert output[0] == "HELIXSR_RUNTIME_PLAN_V1" and output[-1] == "E"
    fields = output[1].split(); assert fields[0] == "T"
    transient = tuple(map(int, fields[1:]))
    result = []; index = 2
    while index < len(output) - 1:
        fields = output[index].split(); index += 1
        if fields[0] == "F":
            result.append(("F", int(fields[1]), fields[2], *map(int, fields[3:])))
        elif fields[0] == "B":
            count = int(fields[2]); assert len(fields) == count + 3
            result.append(("B", int(fields[1]), tuple(fields[3:])))
        else:
            assert fields[0] == "D"
            count = int(fields[7]); descriptors = []
            for _ in range(count):
                descriptor = output[index].split(); index += 1
                assert descriptor[0] == "d" and int(descriptor[1]) == len(descriptors)
                descriptors.append((descriptor[2], descriptor[3], *map(int, descriptor[4:])))
            result.append(("D", int(fields[1]), fields[2], tuple(map(int, fields[3:6])),
                           fields[6], tuple(descriptors)))
    return transient, result


cases = []
for shape in ((128, 96, 128, 96), (192, 144, 128, 96), (256, 192, 128, 96),
              (1919, 1079, 1279, 719), (1920, 1080, 1280, 720)):
    for frame in (0, 1, 2, 7, 31):
        for exposure in (0, 1):
            wanted = expected(shape, frame, exposure)
            got = actual(shape, frame, exposure)
            assert got == wanted, (shape, frame, exposure)
            cases.append({"shape": shape, "frame": frame, "auto_exposure": bool(exposure),
                          "steps": len(got[1]), "dispatches": sum(x[0] == "D" for x in got[1])})
for dynamic in ((0.25, -0.375, -0.125, 0.4375, 0.75, 0, 1),
                (-0.5, 0.25, 0, 0, 2.0, 1, 0)):
    shape = (192, 144, 128, 96); frame = 19; exposure = 1
    wanted = expected(shape, frame, exposure, dynamic)
    got = actual(shape, frame, exposure, dynamic)
    assert got == wanted
    cases.append({"shape": shape, "frame": frame, "auto_exposure": True,
                  "dynamic": dynamic, "steps": len(got[1]),
                  "dispatches": sum(x[0] == "D" for x in got[1])})
Path(report).write_text(json.dumps({"schema": 1, "cases": cases, "exact_matches": len(cases),
    "scope": "C++ runtime preparation versus checked Python descriptor/uniform oracle"}, indent=2) + "\n")
print(f"{len(cases)} dynamic C++ runtime plans exactly match the checked oracle")
