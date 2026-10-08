import json
import subprocess
import sys

exe = sys.argv[1]
def run(*args):
    return subprocess.run([exe, *map(str, args)], capture_output=True, text=True)
def graph(*args):
    result = run(*args)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)

for invalid in [(0,1080,1280,720,0,0), (4097,1080,1280,720,0,0),
                (1920,1080,1921,720,0,0), (1920,1080,1280,720,0,2),
                ('-1',1080,1280,720,0,0), ('1920x',1080,1280,720,0,0)]:
    assert run(*invalid).returncode != 0
for width, expected in [(1499,'dfn_3x3_256'), (1500,'dfn_mid'),
                        (1699,'dfn_mid'), (1700,'dfn_fastpath')]:
    g = graph(width,width,1000,1000,0,0)
    outputs = [p for p in g['launches'] if 'aniso_gaussian' in p['kernel']]
    assert len(outputs) == 1 and expected in outputs[0]['kernel']
for shape in [(192,144,128,96),(1920,1080,1280,720),(1919,1079,1279,719),
              (3840,2160,1920,1080),(1,1,1,1)]:
    for frame in range(4):
        for exposure in range(2):
            g = graph(*shape,frame,exposure)
            assert g == graph(*shape,frame,exposure), 'nondeterministic graph'
            names = [p['kernel'] for p in g['launches']]
            mv = next(r for r in g['images'] if r['name'] == 'GAME_MV')
            assert mv['size'] == list(shape[2:])
            assert ('cuda_clear_buffer_kernel' in names) == (frame == 0)
            assert ('cuda_luma_convert_kernel' in names) == (exposure == 1 and frame % 2 == 0)
            assert names[-1] == 'cuda_copy_exposure_kernel'
            for p in g['launches']:
                args = bytes.fromhex(p['args_hex'])
                assert all(b['arg_offset'] + 8 <= len(args) for b in p['bindings'])
print('44 valid graph configurations, deterministic replay and six invalid-input cases passed')
