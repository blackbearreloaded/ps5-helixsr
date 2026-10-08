#!/usr/bin/env python3
"""Self-test the published CPU-reference capability audit."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    tool = Path(sys.argv[1])
    spec = importlib.util.spec_from_file_location("reference_capabilities", tool)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory() as folder:
        setup = Path(folder)
        (setup / "kernels/rt").mkdir(parents=True)
        (setup / "kernels/common").mkdir(parents=True)
        px = setup / "kernels/rt/px_cpp.h"
        port = setup / "kernels/common/port_cpp.h"
        px.write_text("void px_yield(); unsigned px_ballot(bool);\n"
                      "F px_tex_level(int,float,float,float); F px_tex_fetch(int,int,int);\n"
                      "void px_surf_store(int,int,int,F,int);\n")
        port.write_text("// shared CPU arithmetic primitives\n")
        inventory = {"source_hashes": {
            "kernels/rt/px_cpp.h": digest(px),
            "kernels/common/port_cpp.h": digest(port)}}
        result = module.audit(setup, inventory)
        assert not result["full_graph_cpu_reference_available"]
        assert set(result["missing_runtime_definitions"]) == set(module.PRIMITIVES)
        px.write_text(px.read_text() + "void px_yield(){} unsigned px_ballot(bool){return 0;}\n"
                      "F px_tex_level(int,float,float,float){return F();}\n"
                      "F px_tex_fetch(int,int,int){return F();}\n"
                      "void px_surf_store(int,int,int,F,int){}\n"
                      "int main(){px_yield(); px_ballot(false); px_tex_level(0,0,0,0);"
                      "px_tex_fetch(0,0,0); px_surf_store(0,0,0,F(),0);}\n")
        inventory["source_hashes"]["kernels/rt/px_cpp.h"] = digest(px)
        complete = module.audit(setup, inventory)
        assert complete["full_graph_cpu_reference_available"]
        px.write_text(px.read_text() + "tamper")
        try:
            module.audit(setup, inventory)
        except ValueError as error:
            assert "hash mismatch" in str(error)
        else:
            raise AssertionError("source tamper was accepted")
    print(json.dumps({"incomplete": True, "complete": True, "tamper_rejected": True}))


if __name__ == "__main__":
    main()
