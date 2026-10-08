import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'tools'))
spec = importlib.util.spec_from_file_location('contract', root / 'tools/contract.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.verify_sources()
for size in (1400, 1500, 1700):
    raw = subprocess.check_output([sys.argv[1], str(size), str(size), '1000', '1000', '0', '1'], text=True)
    result = module.contract(json.loads(raw))
    missing = result['recipe_coverage']['missing']
    assert len(missing) == 4 and all('conv_3x3_pool' in name for name in missing)
    textures = [b for p in result['launches'] for b in p['bindings'] if b['kind'] == 2]
    assert textures and all('sampler' in b for b in textures)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'graph.json'
        path.write_text(raw)
        rejected = subprocess.run([sys.executable, str(root / 'tools/contract.py'), str(path), '--require-recipes'], capture_output=True, text=True)
        assert rejected.returncode == 1 and 'required kernel recipes absent' in rejected.stderr
    broken = json.loads(raw)
    next(b for p in broken['launches'] for b in p['bindings'] if b['kind'] == 2)['arg_offset'] = 65535
    try:
        module.contract(broken)
        raise AssertionError('unknown sampler accepted')
    except ValueError as error:
        assert 'missing sampler' in str(error)
print('Source hashes, sampler coverage and missing-recipe rejection passed for all three output heads')
