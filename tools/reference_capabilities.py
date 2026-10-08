#!/usr/bin/env python3
"""Audit whether the pinned published setup contains a runnable CPU reference."""
import argparse
import hashlib
import json
from pathlib import Path
import re


SOURCES = ("kernels/rt/px_cpp.h", "kernels/common/port_cpp.h")
PRIMITIVES = ("px_yield", "px_ballot", "px_tex_level", "px_tex_fetch", "px_surf_store")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def audit(setup, inventory):
    source_hashes = inventory.get("source_hashes", {})
    verified = {}
    for relative in SOURCES:
        path = setup / relative
        require(path.is_file(), f"missing published source: {relative}")
        expected = source_hashes.get(relative)
        require(expected, f"inventory has no hash for {relative}")
        actual = sha(path)
        require(actual == expected, f"published source hash mismatch: {relative}")
        verified[relative] = actual

    texts = []
    for path in setup.rglob("*"):
        if path.is_file() and path.suffix in (".h", ".hpp", ".c", ".cc", ".cpp", ".inc"):
            texts.append((path.relative_to(setup).as_posix(),
                          path.read_text(errors="replace")))
    declarations, definitions = {}, {}
    cpu_texts = [(path, text) for path, text in texts if "hlsl" not in path.lower()]
    for name in PRIMITIVES:
        declarations[name] = sorted(path for path, text in cpu_texts
                                    if re.search(r"\b" + name + r"\s*\(", text))
        definitions[name] = sorted(path for path, text in cpu_texts
                                   if re.search(r"\b" + name + r"\s*\([^;{}]*\)\s*\{", text))
        require(declarations[name], f"published CPU primitive is absent: {name}")
    missing = [name for name in PRIMITIVES if not definitions[name]]
    launchers = sorted(path for path, text in cpu_texts
                       if re.search(r"\b(main|run_kernel|launch_kernel)\s*\(", text) and
                       any(token in text for token in PRIMITIVES))
    available = not missing and bool(launchers)
    return {
        "schema": 1,
        "verified_source_hashes": verified,
        "primitive_declarations": declarations,
        "primitive_definitions": definitions,
        "missing_runtime_definitions": missing,
        "full_graph_runner_candidates": launchers,
        "full_graph_cpu_reference_available": available,
        "conclusion": ("published setup contains a runnable full-graph CPU reference"
                       if available else
                       "published setup supplies shared CPU semantic headers but no runnable full-graph CPU reference"),
        "next_gate": "run the signed console candidate and evaluate its 16 captures",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("setup", type=Path)
    parser.add_argument("inventory", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--expect-incomplete", action="store_true")
    args = parser.parse_args()
    result = audit(args.setup, json.loads(args.inventory.read_text()))
    if args.expect_incomplete:
        require(not result["full_graph_cpu_reference_available"],
                "published reference unexpectedly became runnable; qualify it before use")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"available": result["full_graph_cpu_reference_available"],
                      "missing": result["missing_runtime_definitions"]}))


if __name__ == "__main__":
    main()
