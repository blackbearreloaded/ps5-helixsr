"""Validate split GPU passes against all outputs of the original fused stage."""
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from probe_fused import run

graph,runner,resident_runner,shader_dir,ptx_dir,output_dir=sys.argv[1:]
shaders=Path(shader_dir);output=Path(output_dir)
name='dltss_nchw8_conv_3x3_pool_conv_1x1_pool_512_032_008_fp16_e5m3_kernel'
results=[]
cases=[('dense',(1,1),seed) for seed in (1729,1730)]
cases += [('identity',(y,x),1729) for y in range(3) for x in range(3)]
for pattern,tap,seed in cases:
    case=f'{pattern}-{tap[0]}{tap[1]}-{seed}'
    directory=output/case
    receipt=run(graph,Path(ptx_dir)/f'{name}.sm_89.ptx',pattern,tap,seed,directory)
    def dispatch(shader,source,destination,size,groups):
        result=subprocess.run([runner,str(shaders/shader),str(source),str(destination),str(size),str(groups)],capture_output=True,text=True,timeout=90)
        assert result.returncode==0,result.stderr
        return destination.read_bytes()
    first_reference=(directory/'first-fp16.bin').read_bytes()
    first=dispatch('convolution_pool.spv',directory/'first-input.bin',directory/'first-actual.bin',len(first_reference),receipt['first_groups'])
    assert first==first_reference,(case,'first FP16 layer differs')
    def resident_conv(shader,payload,activation,activation_offset,destination_prefix,groups):
        header=struct.unpack_from('<8I',payload.read_bytes())
        width,height,channels,outputs,e5,_,weight_offset,bias_offset=header
        pool_bytes=outputs*(width//2)*(height//2)*2
        params=directory/f'{destination_prefix}-params.bin'
        params.write_bytes(struct.pack('<8I',width,height,activation_offset,weight_offset,bias_offset,0,0,0))
        pool=directory/f'{destination_prefix}-pool.bin';skip=directory/f'{destination_prefix}-skip.bin'
        process=subprocess.run([resident_runner,str(shaders/shader),str(groups),f'u:{params}',f'r:{activation}',f'r:{payload}',
            f'w:{pool}:{pool_bytes}',f'w:{skip}:{pool_bytes*4}'],capture_output=True,text=True,timeout=90)
        assert process.returncode==0,process.stderr
        return pool.read_bytes(),skip.read_bytes(),process.stderr.strip()
    first_pool,first_skip,first_resident_host=resident_conv('conv_16_32_k3_half.spv',directory/'first-input.bin',
        directory/'first-input.bin',struct.unpack_from('<I',(directory/'first-input.bin').read_bytes(),20)[0],'first-resident',receipt['first_groups'])
    assert first_pool+first_skip==first_reference,(case,'first resident layer differs')
    second_payload=bytearray((directory/'second-input.bin').read_bytes())
    input_offset=struct.unpack_from('<I',second_payload,20)[0]
    second_payload[input_offset:input_offset+receipt['first_pool_bytes']]=first[:receipt['first_pool_bytes']]
    (directory/'second-chained.bin').write_bytes(second_payload)
    second_reference=(directory/'second-fp16.bin').read_bytes()
    second=dispatch('convolution_1x1_pool.spv',directory/'second-chained.bin',directory/'second-actual.bin',len(second_reference),receipt['second_groups'])
    assert second==second_reference,(case,'second FP16 layer differs')
    second_pool,second_skip,second_resident_host=resident_conv('conv_32_64_k1_half.spv',directory/'second-chained.bin',
        directory/'first-resident-pool.bin',0,'second-resident',receipt['second_groups'])
    assert second_pool+second_skip==second_reference,(case,'second resident layer differs')
    def resident_encode(label,source,expected):
        params=directory/f'{label}-resident-encode-params.bin'
        scalars=len(source.read_bytes())//2
        params.write_bytes(struct.pack('<4I',0,0,scalars,0))
        destination=directory/f'{label}-resident-encoded.bin'
        process=subprocess.run([resident_runner,str(shaders/'encode_e5m3_resident.spv'),str((scalars//4+63)//64),
            f'u:{params}',f'r:{source}',f'w:{destination}:{scalars}'],capture_output=True,text=True,timeout=90)
        assert process.returncode==0,process.stderr
        actual=destination.read_bytes();assert actual==expected,(case,label,'resident encoder differs')
        return hashlib.sha256(actual).hexdigest()
    receipt['first_resident_e5_sha256']=resident_encode('first-skip',directory/'first-resident-skip.bin',(directory/'first-skip-e5.bin').read_bytes())
    second_e5=(directory/'second-e5.bin').read_bytes();pool_e5_bytes=len(second_pool)//2
    receipt['second_pool_resident_e5_sha256']=resident_encode('second-pool',directory/'second-resident-pool.bin',second_e5[:pool_e5_bytes])
    receipt['second_skip_resident_e5_sha256']=resident_encode('second-skip',directory/'second-resident-skip.bin',second_e5[pool_e5_bytes:])
    receipt['first_resident_host']=first_resident_host;receipt['second_resident_host']=second_resident_host
    for label,data,reference in (('first',first[receipt['first_pool_bytes']:],'first-skip-e5.bin'),('second',second,'second-e5.bin')):
        source=directory/f'{label}-encode-input.bin';source.write_bytes(data)
        encoded=dispatch('encode_e5m3.spv',source,directory/f'{label}-encoded.bin',len(data)//2,(len(data)//8+63)//64)
        assert encoded==(directory/reference).read_bytes(),(case,label,'E5 output differs')
        receipt[f'{label}_actual_sha256']=hashlib.sha256(encoded).hexdigest()
    results.append(receipt)
    print(f'PASS fused {case}: all three E5 outputs and FP16 layer boundaries',flush=True)
(output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
print(f'{len(results)} full fused-stage comparisons passed; host harness copies intermediates between submissions')
