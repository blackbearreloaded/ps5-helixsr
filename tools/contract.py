"""Attach published sampler evidence and audit recipe coverage, without executing setup code.

Usage: python3 tools/contract.py build/frame0.json [--require-recipes]
Recipe presence is NOT a claim that generated/native shader artifacts exist.
"""
import ast
import hashlib
import json
from pathlib import Path
import sys
from clear_operation import lower_clear

ROOT = Path(__file__).resolve().parents[1]

def verify_sources(root=ROOT):
    pins = json.loads((root / 'sources.json').read_text())['helixsr']['files']
    for relative, expected in pins.items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f'source digest mismatch: {relative}')

def literal(path, name):
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f'missing source literal {name}')

def contract(graph):
    verify_sources()
    base = ROOT / 'third_party/helixsr'
    entries = literal(base / 'kernels.py', 'ENTRY')
    aliases = {entry: alias for alias, entry in entries.items()}
    samplers = literal(base / 'samplers.py', 'SAMPLERS')
    input_samplers = literal(base / 'samplers.py', 'K8_SAMPLERS')
    missing = set()
    for launch in graph['launches']:
        name = launch['kernel']
        if name=='cuda_clear_buffer_kernel':
            launch['native_operation']=lower_clear(graph,launch)
        alias = aliases.get(name)
        launch['recipe'] = alias
        if alias is None:
            missing.add(name)
        raw = input_samplers if alias and alias.startswith('k8') else samplers.get(alias, '')
        settings = {int(p[0]): list(map(int, p[1:])) for p in (s.split(':') for s in raw.split(',') if s)}
        for binding in launch['bindings']:
            if binding['kind'] != 2:  # BindTexture
                continue
            offset = binding['arg_offset']
            if offset not in settings:
                raise ValueError(f'missing sampler: {name} argument {offset}')
            filter_mode, address_u, address_v = settings[offset]
            if filter_mode not in (0, 1) or address_u != 1 or address_v != 1:
                raise ValueError(f'unsupported sampler evidence: {name} argument {offset}')
            binding['sampler'] = {'filter': 'linear' if filter_mode else 'point', 'address_u': 'clamp', 'address_v': 'clamp',
                                  'coordinate_policy': 'pending_shader_verification'}
    graph['recipe_coverage'] = {'missing': sorted(missing), 'complete': not missing}
    graph['native_ready'] = False
    return graph

if __name__ == '__main__':
    try:
        if len(sys.argv) not in (2, 3) or (len(sys.argv) == 3 and sys.argv[2] != '--require-recipes'):
            raise ValueError(__doc__)
        result = contract(json.loads(Path(sys.argv[1]).read_text()))
        print(json.dumps(result, indent=2))
        if len(sys.argv) == 3 and not result['recipe_coverage']['complete']:
            raise ValueError('required kernel recipes absent: ' + ', '.join(result['recipe_coverage']['missing']))
    except (ValueError, KeyError, OSError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
