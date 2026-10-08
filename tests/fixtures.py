import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile

root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('fixture_generator',root/'tools/fixtures.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
with tempfile.TemporaryDirectory() as directory:
    base = Path(directory)
    for shape in ((32,24,48,36),(33,25,49,37)):
        first = module.generate(base/'first',*shape)
        second = module.generate(base/'second',*shape)
        assert first == second and len(first['frames']) == 24
        for entry in first['frames']:
            for role,info in entry['files'].items():
                data = (base/'first'/info['path']).read_bytes()
                assert len(data) == info['bytes'] and hashlib.sha256(data).hexdigest() == info['sha256']
                assert data == (base/'second'/info['path']).read_bytes()
            color = (base/'first'/entry['files']['color.rgba16f']['path']).read_bytes()
            depth = (base/'first'/entry['files']['depth.r32f']['path']).read_bytes()
            motion = (base/'first'/entry['files']['motion.rg16f']['path']).read_bytes()
            mask = (base/'first'/entry['files']['disocclusion.r8u']['path']).read_bytes()
            assert len(color)==shape[0]*shape[1]*8 and len(depth)==len(motion)==shape[0]*shape[1]*4
            assert len(mask)==shape[0]*shape[1]
            if entry['scene']=='constant':
                assert set(struct.iter_unpack('<4e',color)) == {(0.25,0.5,1.0,1.0)}
            if entry['scene']!='motion' or entry['frame']==0:
                assert not any(mask) and not any(motion)
            else:
                assert sum(mask)>0, 'moving foreground must reveal background'
                pairs = list(zip(struct.iter_unpack('<f',depth),struct.iter_unpack('<2e',motion)))
                assert all(mv==((-1.0,0.0) if z[0]==0.75 else (1.0,0.0)) for z,mv in pairs)
                assert {z[0] for z,mv in pairs} == {0.25,0.75}
        # Freeze default input bytes to detect silent convention or raster changes.
        if shape==(32,24,48,36):
            digest = hashlib.sha256((base/'first/manifest.json').read_bytes()).hexdigest()
            expected = json.loads((root/'tests/fixture_pin.json').read_text())['manifest_sha256']
            assert digest == expected, (digest,expected)
print('24-frame fixtures replay identically at even and odd sizes; motion, depth and disocclusion checks passed')
