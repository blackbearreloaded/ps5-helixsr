"""Compile and receipt the early-stage resident shaders without model assets:
the five fast convolutions (tools/conv_kernels.py, GLSL) and the E5M3 encoder (HLSL)."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import conv_kernels

VARIANTS={name:(v[0],v[1],v[2],int(v[3])) for name,v in conv_kernels.VARIANTS.items()}
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def run(command):
    result=subprocess.run(list(map(str,command)),capture_output=True,text=True,timeout=120)
    if result.returncode: raise RuntimeError(result.stderr+result.stdout)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dxc',type=Path,required=True);p.add_argument('--compiler',type=Path,required=True)
    p.add_argument('--spirv-val',default='spirv-val');p.add_argument('--out',type=Path,required=True)
    p.add_argument('--glslang',default='glslangValidator')
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    sources=[a.root/'tools/conv_kernels.py',
             a.root/'shaders/encode_e5m3_resident.hlsl',a.root/'third_party/helixsr/common/fp16.h']
    manifest={'schema':1,'dxc_sha256':digest(a.dxc),'compiler_sha256':digest(a.compiler),
              'source_hashes':{str(x.relative_to(a.root)):digest(x) for x in sources},'shaders':[],'execution_qualified':False,
              'scope':'SPIR-V validation and native offline compilation; no device execution'}
    jobs=[(name,None,[],
           ['uniform-buffer','buffer','buffer','buffer','buffer'],
           dict(channels=v[0],outputs=v[1],kernel_size=v[2],input_e5=bool(v[3]),
                outputs_per_invocation=conv_kernels.VARIANTS[name][4],lanes=conv_kernels.LANES,
                weight_offset=conv_kernels.VARIANTS[name][5]))
          for name,v in VARIANTS.items()]
    jobs.append(('encode_e5m3_resident',a.root/'shaders/encode_e5m3_resident.hlsl',[],
                 ['uniform-buffer','buffer','buffer'],{}))
    for name,source,defines,kinds,specialization in jobs:
        output=a.out/(name+'.spv')
        if source is None:
            generated=a.out/(name+'.comp')
            generated.write_text(conv_kernels.source(name))
            run([a.glslang,'-V','--target-env','vulkan1.2',generated,'-o',output])
        else:
            run([a.dxc,'-spirv','-fspv-target-env=vulkan1.2','-T','cs_6_2','-E','main','-HV','2021','-O3',*defines,
                 '-I',a.root/'third_party/helixsr','-I',a.root/'shaders',source,'-Fo',output])
        run([a.spirv_val,'--target-env','vulkan1.2',output])
        descriptors=[f'0:{i}:{kind}:1' for i,kind in enumerate(kinds)]
        native=run([a.compiler,output,*descriptors])
        records=[json.loads(line) for line in native.stdout.splitlines() if line.startswith('{')]
        compiled=next((x for x in records if x.get('result')=='success'),None)
        adapter=next((x for x in records if x.get('adapter_result')==0),None)
        if not compiled or not adapter or adapter.get('adapter_descriptors')!=len(kinds) or adapter.get('wave_size')!=32:
            raise RuntimeError(f'unexpected native result for {name}: {records}')
        manifest['shaders'].append({'name':name,'spirv_sha256':digest(output),'descriptors':kinds,
            'specialization':specialization,'compile':compiled,'adapter':adapter,'stderr':native.stderr})
        print(f'{name}: PASS',flush=True)
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
