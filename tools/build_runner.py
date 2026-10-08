#!/usr/bin/env python3
"""Build the PS5 frame runner (samples/frame_runner.cpp) as an installable title folder.

The folder holds the signed executable, the model, a frame-input directory made by
tools/reference_inputs.py and run.txt. It is a local test title: the model is NVIDIA's,
the folder must not be shared.

Toolchain paths default to the environment the FSR4 project uses:
  PS5_PAYLOAD_SDK, PS5_NATIVE_APP_TEMPLATE, PS5VK_SDK (the driver's dist-sdk)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MODEL_SHA256 = "762adfde720035f7ac43910846e0ce6826c239bcd41f863196f77c547ae57153"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compile_source(compiler, source, target, includes, environment, cxx=True, defines=()):
    subprocess.run([str(compiler), "-std=c++17" if cxx else "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                    "-Wno-missing-field-initializers", "-Wno-unused-parameter", *["-D" + x for x in defines],
                    *["-I" + str(x) for x in includes], "-c", str(source), "-o", str(target)],
                   env=environment, check=True)


def build_title(args, out, package, asset_source, app_sources, title_name, content_id, include_dirs=(), defines=(),
                sce_sys=None, param_overrides=None):
    """A signed title folder: the application's sources (.c or .cpp) with the runtime, the planner,
    the generated shader tables and the application heap. Launch assets in sce_sys replace the
    template's; param_overrides sets fields of param.json (a dictionary is merged into the one there)."""
    sdk, template, driver = args.payload_sdk.resolve(), args.native_template.resolve(), args.driver_sdk.resolve()
    compiler, c_compiler, linker = sdk / "bin/prospero-clang++", sdk / "bin/prospero-clang", sdk / "bin/prospero-lld"
    builder = template / "build/host/ps5-native-tool"
    environment = dict(os.environ, PS5_PAYLOAD_SDK=str(sdk))
    includes = [*include_dirs, ROOT / "include", ROOT / "src", ROOT / "third_party/helixsr/model", asset_source.parent,
                ROOT / "samples", driver / "include"]
    objects = []
    for index, source in enumerate((*app_sources, ROOT / "src/ps5_helixsr.cpp", ROOT / "src/runtime_plan.cpp",
                                    ROOT / "third_party/helixsr/model/model.cpp", asset_source,
                                    ROOT / "samples/native_heap.c")):
        target, cxx = out / f"title-{index}.o", Path(source).suffix == ".cpp"
        compile_source(compiler if cxx else c_compiler, source, target, includes, environment, cxx=cxx,
                       defines=defines if index < len(app_sources) else ())
        objects.append(target)
    crt = out / "crt.o"
    subprocess.run([str(compiler), "-std=c++20", "-O2", "-fno-exceptions", "-fno-rtti", "-c",
                    str(template / "tooling/native/app_crt.cpp"), "-o", str(crt)], env=environment, check=True)
    pie, elf = out / "title-pie.elf", out / "eboot.elf"
    wraps = ("malloc", "calloc", "realloc", "free", "posix_memalign", "memalign", "aligned_alloc",
             "malloc_usable_size")
    subprocess.run([str(linker), "-L" + str(sdk / "target/lib"), "-T", str(driver / "lib/ps5-pie.ld"),
                    "--eh-frame-hdr", "--version-script", str(driver / "lib/app-symbols.map"), "-e", "_start",
                    "-o", str(pie), str(crt), *map(str, objects), *["--wrap=" + name for name in wraps],
                    str(driver / "lib/libps5vk.a"), str(driver / "lib/libpsbc.a"),
                    *[str(sdk / "target/lib" / name) for name in ("libc++.a", "libc++abi.a", "libunwind.a", "libc.a")],
                    "--as-needed", str(sdk / "target/lib/libkernel.so"),
                    *sorted(str(x) for x in (sdk / "target/lib").glob("*.so")),
                    str(driver / "lib/libSceAgc.so"), str(driver / "lib/libSceAgcDriver.so")], check=True)
    subprocess.run([str(builder), "link", "--in", str(pie), "--out", str(elf), "--stub-dir", str(sdk / "target/lib"),
                    "--module-sdk", "0x02000009", "--stub", str(driver / "lib/libSceAgc.so"), "--stub",
                    str(driver / "lib/libSceAgcDriver.so"), "--companion-sdk", "0x08050001", "--file-name",
                    "eboot.elf"], check=True)
    subprocess.run([str(builder), "self", "--sign", "--in", str(elf), "--out", str(package / "eboot.bin"),
                    "--magic", "0x1D3D154F"], check=True)
    (package / "sce_sys").mkdir(exist_ok=True)
    (package / "sce_module").mkdir(exist_ok=True)
    parameters = json.loads(args.param_template.read_text())
    parameters.update(titleId=args.title_id, conceptId=args.title_id[4:], contentId=content_id)
    parameters["downloadDataSize"] = 256
    parameters["localizedParameters"]["en-US"]["titleName"] = title_name
    for key, value in (param_overrides or {}).items():
        parameters[key] = {**parameters.get(key, {}), **value} if isinstance(value, dict) else value
    (package / "sce_sys/param.json").write_text(json.dumps(parameters, indent=2) + "\n")
    shutil.copyfile(template / "runtime/libc.prx", package / "sce_module/libc.prx")
    # The shell refuses to launch a title without the launch assets.
    for name in ("icon0.png", "pic0.dds", "pic1.dds", "snd0.at9"):
        own = Path(sce_sys) / name if sce_sys else None
        shutil.copyfile(own if own and own.is_file() else template / "sce_sys" / name, package / "sce_sys" / name)


def runtime_assets(args, out, external_kernels=False):
    """The generated shader tables as a source file; returns its path. With external_kernels the
    network's kernels are left out (the table keeps their identities) and nothing of NVIDIA's is read."""
    generated = out / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    source = generated / "helixsr_assets.cpp"
    subprocess.run([sys.executable, str(ROOT / "tools/build_runtime_assets.py"),
                    "--inventory", str(ROOT / "validation/generated-main.json"),
                    *(["--external-kernels"] if external_kernels else ["--generated", str(args.generated_dir)]),
                    "--compact-inventory", str(ROOT / "validation/compact-resident.json"),
                    "--compact-dir", str(args.compact_dir), "--header", str(generated / "helixsr_assets.h"),
                    "--source", str(source), "--receipt", str(generated / "runtime-assets.json")], check=True)
    return source


def toolchain_arguments(parser):
    env = os.environ
    template_default = env.get("PS5_NATIVE_APP_TEMPLATE", "")
    parser.add_argument("--generated-dir", type=Path, default=ROOT / "build/main-inventory/generated")
    parser.add_argument("--compact-dir", type=Path, default=ROOT / "build/compact-resident")
    parser.add_argument("--payload-sdk", type=Path, default=Path(env.get("PS5_PAYLOAD_SDK", "")))
    parser.add_argument("--native-template", type=Path, default=Path(template_default))
    parser.add_argument("--driver-sdk", type=Path, default=Path(env.get("PS5VK_SDK", "")))
    parser.add_argument("--param-template", type=Path,
                        default=Path(template_default) / "sce_sys/param.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cases", nargs="+", metavar="DIR[:FRAMES[:TIMING[:PROFILE[:AUTO[:STAGES]]]]]",
                        help="frame-input directories (tools/reference_inputs.py), each with an optional "
                             "frame count, a number of extra timed runs of the upscaler, a number of profiled "
                             "runs, 1 for automatic exposure and 1 to save the first frame's stages")
    parser.add_argument("--model", type=Path, required=True, help="canonical Model E weights (local, never shared)")
    parser.add_argument("--out", type=Path, default=ROOT / "build/runner")
    parser.add_argument("--no-save", action="store_true", help="do not write the output frames")
    parser.add_argument("--auto-exposure", action="store_true", help="automatic exposure in every case")
    parser.add_argument("--title-id", default="PPSA89012")
    toolchain_arguments(parser)
    args = parser.parse_args()
    if sha(args.model) != MODEL_SHA256:
        raise SystemExit("canonical model identity mismatch")
    out = args.out.resolve()
    package = out / args.title_id
    if package.exists():
        shutil.rmtree(package)
    assets = package / "assets"
    assets.mkdir(parents=True)
    shutil.copyfile(args.model, assets / "model.bin")
    cases = []
    for item in args.cases:
        directory, *rest = item.split(":")
        directory = Path(directory)
        inputs = json.loads((directory / "inputs.json").read_text())
        frames = int(rest[0]) if rest and rest[0] else inputs["frames"]
        timing = int(rest[1]) if len(rest) > 1 and rest[1] else 0
        profile = int(rest[2]) if len(rest) > 2 and rest[2] else 0
        automatic = int(args.auto_exposure or (len(rest) > 3 and rest[3] == "1"))
        stages = int(len(rest) > 4 and rest[4] == "1")
        name = directory.name
        (assets / name).mkdir()
        shutil.copyfile(directory / "frames.bin", assets / name / "frames.bin")
        for frame in range(frames):
            shutil.copytree(directory / "in" / f"{frame:02d}", assets / name / "in" / f"{frame:02d}")
        (assets / name / "run.txt").write_text(
            f"render {inputs['render'][0]} {inputs['render'][1]}\noutput {inputs['output'][0]} {inputs['output'][1]}\n"
            f"frames {frames}\nauto {automatic}\nsave {int(not args.no_save)}\ntiming {timing}\n"
            f"profile {profile}\nstages {stages}\n")
        cases.append(dict(name=name, inputs_sha256=inputs["inputs_sha256"], scenario=inputs["scenario"],
                          render=inputs["render"], output=inputs["output"], frames=frames, timing=timing,
                          auto_exposure=bool(automatic)))
    (assets / "cases.txt").write_text("".join(case["name"] + "\n" for case in cases))
    build_title(args, out, package, runtime_assets(args, out), [ROOT / "samples/frame_runner.cpp"],
                "PS5 HelixSR frame runner", f"UP9000-{args.title_id}_00-HELIXSRFRAMERUNS")
    receipt = dict(title_id=args.title_id, cases=cases,
                   eboot_sha256=sha(package / "eboot.bin"), eboot_bytes=(package / "eboot.bin").stat().st_size,
                   driver_libps5vk_sha256=sha(args.driver_sdk / "lib/libps5vk.a"))
    (out / "runner.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
