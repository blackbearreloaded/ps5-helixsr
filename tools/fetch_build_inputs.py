#!/usr/bin/env python3
# Copyright (C) 2026 BlackBearReloaded
# SPDX-License-Identifier: GPL-3.0-or-later
"""Fetch and verify the pinned public build inputs listed in tools/build_inputs.json.

Everything lands in the ignored build tree, where the Makefile looks for it:

- the native app template at its pinned commit, built once: its host tool, the
  payload SDK it pins and the libc.prx it generates from source;
- dxc and libdxcompiler.so from Microsoft's DXC release;
- the HelixSR 1.2.0 release, extracted: its setup builds the network on this
  machine from NVIDIA's DLSS DLL (make model).

NVIDIA's DLL is not fetched here. make model takes a local copy (DLSS_DLL=...)
or lets HelixSR's setup download it from NVIDIA's GitHub once you have accepted
NVIDIA's license (ACCEPT_NVIDIA_DLSS_LICENSE=1).

--env prints the variables the builds read (PS5VK_LAB_ROOT, PS5_PAYLOAD_SDK,
PS5_NATIVE_APP_TEMPLATE); --github-env appends them to $GITHUB_ENV.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PINS = ROOT / "tools/build_inputs.json"
DOWNLOADS = ROOT / "build/inputs/downloads"
INPUTS = ("app_template", "dxc", "helixsr")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url, digest):
    """The file at url, cached under build/inputs/downloads and checked against digest."""
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    target = DOWNLOADS / url.rsplit("/", 1)[1]
    if target.is_file() and sha256(target) == digest:
        return target
    print(f"downloading {url}", flush=True)
    partial = target.with_name(target.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "ps5-helixsr-build-inputs"})
    with urllib.request.urlopen(request, timeout=120) as response, open(partial, "wb") as out:
        shutil.copyfileobj(response, out, 1 << 20)
    actual = sha256(partial)
    if actual != digest:
        partial.unlink()
        raise SystemExit(f"{url}: SHA-256 {actual}, expected {digest}")
    partial.replace(target)
    return target


def git(*args, cwd=None):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def checkout(repository, commit, path):
    """A checkout of repository at commit; an existing one at that commit is kept as it is."""
    path = ROOT / path
    if (path / ".git").exists() and git("rev-parse", "HEAD", cwd=path) == commit:
        return path
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    print(f"fetching {repository} at {commit}", flush=True)
    git("init", "-q", cwd=path)
    git("remote", "add", "origin", repository, cwd=path)
    git("fetch", "-q", "--depth", "1", "origin", commit, cwd=path)
    git("checkout", "-q", "FETCH_HEAD", cwd=path)
    return path


def extract_tar(archive, members, path):
    """The members (paths below the archive's top directory), in one pass over the archive."""
    path = ROOT / path
    if all((path / m).is_file() for m in members):
        return path
    found = set()
    with tarfile.open(archive, "r|*") as tar:
        for info in tar:
            member = info.name.split("/", 1)[-1]
            if member not in members or not info.isfile():
                continue
            (path / member).parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(info) as source, open(path / member, "wb") as out:
                shutil.copyfileobj(source, out)
            (path / member).chmod(info.mode & 0o777)
            found.add(member)
    if found != set(members):
        raise SystemExit(f"{archive.name} lacks {sorted(set(members) - found)}")
    return path


def extract_release(archive, path):
    """The HelixSR release without its top directory; kept when its setup script is already there."""
    path = ROOT / path
    if (path / "setup/helixsr_setup.py").is_file():
        return path
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            member = info.filename.split("/", 1)[-1]
            if info.is_dir() or not member or ".." in Path(member).parts:
                continue
            (path / member).parent.mkdir(parents=True, exist_ok=True)
            (path / member).write_bytes(z.read(info))
            mode = info.external_attr >> 16 & 0o777  # the release's scripts and tools are executable
            if mode:
                (path / member).chmod(mode)
    if not (path / "setup/helixsr_setup.py").is_file():
        raise SystemExit(f"{archive.name} holds no setup/helixsr_setup.py")
    return path


def build_template(path):
    """The template's payload SDK, generated libc.prx and host tool, from its own scripts.

    Its sample app is not built: at the pinned commit, LLVM 21's lld leaves that small
    image without the .data.rel.ro section the tool requires.
    """
    tool = path / "build/host/ps5-native-tool"
    needed = [tool, path / "runtime/libc.prx", path / ".deps/native/ps5-payload-sdk/bin/prospero-clang"]
    if not all(p.is_file() for p in needed):
        print("building the native app template's SDK, runtime and tool", flush=True)
        subprocess.run(["bash", "tools/setup-native-dependencies.sh"], cwd=path, check=True)
        subprocess.run(["make", "-C", str(path), "runtime/libc.prx"], check=True)
        # rebuild-libc.sh compiles the tool as tools/build.sh does, into its work folder.
        tool.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path / "build/runtime-shim/ps5-native-tool", tool)
    missing = [str(p) for p in needed if not p.is_file()]
    if missing:
        raise SystemExit("the native app template build did not produce " + ", ".join(missing))


def environment(pins):
    """The template sits in a lab layout (lab/third_party/...), where the driver's SDK builder looks."""
    template = ROOT / pins["app_template"]["path"]
    return {"PS5VK_LAB_ROOT": str(template.parents[1]), "PS5_NATIVE_APP_TEMPLATE": str(template),
            "PS5_PAYLOAD_SDK": str(template / ".deps/native/ps5-payload-sdk")}


def fetch(pins, only):
    wanted = lambda name: not only or name in only  # noqa: E731
    if wanted("app_template"):
        pin = pins["app_template"]
        build_template(checkout(pin["repository"], pin["commit"], pin["path"]))
    if wanted("dxc"):
        pin = pins["dxc"]
        extract_tar(download(pin["url"], pin["sha256"]), pin["members"], pin["path"])
    if wanted("helixsr"):
        pin = pins["helixsr"]
        extract_release(download(pin["url"], pin["sha256"]), pin["path"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", action="append", default=[], choices=INPUTS,
                        help="fetch only these inputs (repeatable)")
    parser.add_argument("--env", action="store_true", help="print the build variables and exit")
    parser.add_argument("--github-env", action="store_true", help="also append the build variables to $GITHUB_ENV")
    args = parser.parse_args()
    pins = json.loads(PINS.read_text())
    variables = environment(pins)
    if args.env:
        print("\n".join(f"export {k}={v}" for k, v in variables.items()))
        return
    fetch(pins, set(args.only))
    if args.github_env:
        with open(os.environ["GITHUB_ENV"], "a") as f:
            f.writelines(f"{k}={v}\n" for k, v in variables.items())
    print("\n".join(f"export {k}={v}" for k, v in variables.items()))


if __name__ == "__main__":
    main()
