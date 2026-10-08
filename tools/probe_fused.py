"""Compare all three first-fused-stage outputs against a compact two-layer model."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import struct
import subprocess
import numpy as np
from convolution import convolution_pool
from probe_convolution import Memory
from ptx_reference import CTA,ptxsim

def e5(values):
    bits=values.view(np.uint16)
    power=(bits&0xfc00).view(np.float16).astype(np.float64)
    rounded=(values.astype(np.float64)+power/16).astype(np.float16)
    return ((rounded.view(np.uint16)>>7)&255).astype(np.uint8)

def run(graph_executable,ptx,pattern='identity',tap=(1,1),seed=1729,fixture_dir=None):
    graph=json.loads(subprocess.check_output([str(graph_executable),'192','144','128','96','1','0'],text=True))
    kernel=ptxsim.Kernel(str(ptx))
    launch=next(p for p in graph['launches'] if p['kernel']==kernel.name)
    args=bytearray.fromhex(launch['args_hex'])
    u32=lambda offset:struct.unpack_from('<I',args,offset)[0]
    height,width=u32(316),u32(320)
    by_arg={b['arg_offset']:b for b in launch['bindings']}
    memory=Memory()
    buffers={}
    for role,r in enumerate(graph['buffers']):
        buffers[role]=((role+1)<<32,np.zeros(r['bytes'],dtype=np.uint8))
        memory.add(*buffers[role])
    weights=np.zeros(graph['weight_bytes'],dtype=np.uint8)
    memory.add(1<<48,weights)
    for b in launch['bindings']:
        base=buffers[b['role']][0] if b['kind']==1 else 1<<48
        struct.pack_into('<Q',args,b['arg_offset'],base+b['offset'])
    values=np.empty((2,height,width,8),dtype=np.float16)
    for c in range(16):
        for y in range(height):
            for x in range(width):
                values[c//8,y,x,c%8]=(1+(c*3+x+y*5)%16)/16
    def weight_view(arg,shape):
        b=by_arg[arg]
        return weights[b['offset']:b['offset']+b['weight_size']].view(np.float16).reshape(shape)
    w1=weight_view(8,(2,3,3,32,8)); b1=weight_view(88,(32,))
    w2=weight_view(40,(4,1,1,64,8)); b2=weight_view(176,(64,))
    for o in range(32): w1[(o%16)//8,tap[0],tap[1],o,o%8]=1
    for o in range(64): w2[(o%32)//8,0,0,o,o%8]=1
    if pattern=='dense':
        rng=np.random.default_rng(seed)
        values[:]=rng.integers(0,16,size=values.shape).astype(np.float16)/16
        for w in (w1,w2): w[:]=rng.integers(-8,9,size=w.shape).astype(np.float16)/128
        for b in (b1,b2): b[:]=rng.integers(-4,5,size=b.shape).astype(np.float16)/32
    src=by_arg[0]
    packed=values.view(np.uint8).reshape(-1)
    buffers[src['role']][1][src['offset']:src['offset']+len(packed)]=packed
    pool1,skip1=convolution_pool(values,w1,b1)
    pool2,skip2=convolution_pool(pool1,w2,b2)
    expected={}
    for arg,tensor in ((80,skip1),(160,pool2),(168,skip2)):
        b=by_arg[arg]
        base=buffers[b['role']][0]+b['offset']
        expected[base]=e5(tensor).reshape(-1)
    steps=mma=0
    for ctaid in itertools.product(*(range(n) for n in launch['grid'])):
        cta=CTA(kernel,memory,args,ctaid,launch['block'])
        steps+=cta.run(); mma+=cta.mma_count
        if any(cta.copy_pending) or any(cta.copy_groups):
            raise AssertionError('fused stage returns with unfinished asynchronous groups')
    for addresses,data in memory.writes:
        for address,row in zip(addresses,data):
            for base,reference in expected.items():
                offset=int(address)-base
                if 0<=offset and offset+len(row)<=len(reference):
                    if not np.array_equal(row,reference[offset:offset+len(row)]):
                        raise AssertionError(('fused tensor mismatch',hex(int(address)),row.tolist(),reference[offset:offset+len(row)].tolist()))
                    break
            else:
                raise AssertionError(('write outside expected fused tensors',hex(int(address))))
    for base,reference in expected.items():
        actual=memory.read(np.array([base],dtype=np.uint64),len(reference))[0]
        if not np.array_equal(actual,reference):
            raise AssertionError('incomplete fused output')
    receipt={'kernel':kernel.name,'pattern':pattern,'seed':seed,'tap':list(tap),'steps':steps,'mma_instructions':mma,
            'outputs':{str(arg):hashlib.sha256(e5(tensor).tobytes()).hexdigest() for arg,tensor in ((80,skip1),(160,pool2),(168,skip2))},
            'ptx_sha256':hashlib.sha256(Path(ptx).read_bytes()).hexdigest()}
    if fixture_dir is not None:
        directory=Path(fixture_dir);directory.mkdir(parents=True,exist_ok=True)
        def payload(v,w,b):
            _,h,wide,_=v.shape
            a=v.tobytes();weights_data=w.tobytes()
            return struct.pack('<8I',wide,h,v.shape[0]*8,len(b),0,32,32+len(a),32+len(a)+len(weights_data))+a+weights_data+b.tobytes()
        for name,data in {'first-input.bin':payload(values,w1,b1),'second-input.bin':payload(pool1,w2,b2),
            'first-fp16.bin':pool1.tobytes()+skip1.tobytes(),'second-fp16.bin':pool2.tobytes()+skip2.tobytes(),
            'first-skip-e5.bin':e5(skip1).tobytes(),'second-e5.bin':e5(pool2).tobytes()+e5(skip2).tobytes()}.items():
            (directory/name).write_bytes(data)
        receipt.update(first_pool_bytes=pool1.nbytes,second_pool_bytes=pool2.nbytes,
                       first_groups=(pool1.size//2+63)//64,second_groups=(pool2.size//2+63)//64)
        (directory/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('graph');parser.add_argument('ptx')
    parser.add_argument('--pattern',choices=('identity','dense'),default='identity')
    parser.add_argument('--tap',type=int,nargs=2,choices=range(3),default=(1,1))
    parser.add_argument('--seed',type=int,default=1729)
    parser.add_argument('--fixture-dir')
    args=parser.parse_args()
    print(json.dumps(run(args.graph,args.ptx,args.pattern,args.tap,args.seed,args.fixture_dir),indent=2))
