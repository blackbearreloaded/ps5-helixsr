"""Real-stage CPU comparison plus compact SPIR-V execution, with external PTX inputs."""
import hashlib
import json
from pathlib import Path
import subprocess
import struct
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from probe_convolution import run

graph,runner,shader,resident_runner,resident_dir,ptx_dir,output_dir=sys.argv[1:]
output=Path(output_dir)
resident_dir=Path(resident_dir)
results=[]
for short in ('064_064_008_fp16_fp16','128_064_008_fp16_fp16','128_064_008_e5m3_fp16'):
    name=f'dltss_nchw8_conv_3x3_pool_{short}_kernel'
    ptx=Path(ptx_dir)/f'{name}.sm_89.ptx'
    # Two dense signed-weight cases and all nine sparse taps independently check
    # accumulation/activation, every padding edge and both output tensor layouts.
    cases=[('dense',(1,1),seed) for seed in (1729,1730)]
    cases += [('identity',(y,x),1729) for y in range(3) for x in range(3)]
    cases += [('nan',(1,1),1729)]
    for pattern,tap,seed in cases:
        case=f'{short}-{pattern}-{tap[0]}{tap[1]}-{seed}'
        directory=output/case
        receipt=run(graph,ptx,pattern,tap,True,directory,seed)
        actual=directory/'actual.bin'
        process=subprocess.run([runner,shader,str(directory/'input.bin'),str(actual),str(receipt['output_bytes']),str(receipt['groups'])],capture_output=True,text=True,timeout=90)
        assert process.returncode==0,process.stderr
        expected=(directory/'expected.bin').read_bytes()
        data=actual.read_bytes()
        assert data==expected,(case,'compact shader differs from checked stage reference')
        header=struct.unpack_from('<8I',(directory/'input.bin').read_bytes())
        width,height,channels,outputs,e5,input_offset,weight_offset,bias_offset=header
        names={(64,128,1):'conv_64_128_k3_e5',(128,128,0):'conv_128_128_k3_half',
               (128,256,0):'conv_128_256_k3_half'}
        resident_shader=resident_dir/(names[(channels,outputs,e5)]+'.spv')
        pool_bytes=receipt['output_bytes']//5
        params=directory/'resident-params.bin'
        params.write_bytes(struct.pack('<8I',width,height,input_offset,weight_offset,bias_offset,0,0,0))
        pool=directory/'resident-pool.bin';skip=directory/'resident-skip.bin'
        resident=subprocess.run([resident_runner,str(resident_shader),str(receipt['groups']),f'u:{params}',
            f'r:{directory/"input.bin"}',f'r:{directory/"input.bin"}',f'w:{pool}:{pool_bytes}',f'w:{skip}:{pool_bytes*4}'],
            capture_output=True,text=True,timeout=90)
        assert resident.returncode==0,resident.stderr
        resident_data=pool.read_bytes()+skip.read_bytes()
        assert resident_data==expected,(case,'resident descriptor shader differs from checked stage reference')
        receipt['actual_sha256']=hashlib.sha256(data).hexdigest()
        receipt['resident_actual_sha256']=hashlib.sha256(resident_data).hexdigest()
        receipt['host']=process.stderr.strip()
        receipt['resident_host']=resident.stderr.strip()
        results.append(receipt)
        print(f'PASS {case}: {len(data)} output bytes',flush=True)
output.mkdir(parents=True,exist_ok=True)
(output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
print(f'{len(results)} complete stage comparisons passed')
