"""Compare real main-network convolution CTAs with independent synthetic tensors.

Zero/bias bring-up, sparse tap geometry, dense arithmetic and a NaN policy witness
are available. This is not end-to-end or full numerical-range acceptance.
"""
import argparse
import json
import struct
import subprocess
import itertools
from pathlib import Path
import hashlib
import numpy as np
from ptx_reference import CTA,ptxsim
from convolution import convolution_pool

class Memory(ptxsim.Memory):
    def __init__(self):
        super().__init__()
        self.writes=[]
    def write(self,addresses,data):
        self.writes.append((addresses.copy(),data.copy()))
        super().write(addresses,data)

def run(graph_executable,ptx,pattern,tap=(1,1),all_ctas=False,fixture_dir=None,seed=1729):
    graph=json.loads(subprocess.check_output([str(graph_executable),'192','144','128','96','1','0'],text=True))
    kernel=ptxsim.Kernel(str(ptx))
    launch=next(p for p in graph['launches'] if p['kernel']==kernel.name)
    args=bytearray.fromhex(launch['args_hex'])
    memory=Memory()
    buffers={}
    for role,resource in enumerate(graph['buffers']):
        address=(role+1)<<32
        data=np.zeros(resource['bytes'],dtype=np.uint8)
        buffers[role]=(address,data)
        memory.add(address,data)
    weights=np.zeros(graph['weight_bytes'],dtype=np.uint8)
    weight_address=1<<48
    memory.add(weight_address,weights)
    expected_buffers={}
    if pattern=='bias':
        bias=next(b for b in launch['bindings'] if b['arg_offset']==56 and b['kind']==4)
        weights[bias['offset']:bias['offset']+bias['weight_size']]=np.full(bias['weight_size']//2,0.25,dtype=np.float16).view(np.uint8)
    if pattern in ('identity','dense','trace','nan'):
        u32=lambda offset:struct.unpack_from('<I',args,offset)[0]
        height,width,channels=u32(80),u32(84),u32(92)
        pool_height,pool_width,outputs=u32(100),u32(104),u32(108)
        assert (height,width)==(2*pool_height,2*pool_width)
        by_arg={b['arg_offset']:b for b in launch['bindings']}
        src=by_arg[0]
        encoded='e5m3_fp16' in kernel.name
        values=np.empty((channels//8,height,width,8),dtype=np.float16)
        for channel in range(channels):
            for y in range(height):
                for x in range(width):
                    # Positive dyadic values representable in both input encodings.
                    bits=np.uint16((64+(17*channel+3*x+5*y)%128)<<7)
                    values[channel//8,y,x,channel%8]=bits.view(np.float16)
        packed=(values.view(np.uint16)>>7).astype(np.uint8) if encoded else values.view(np.uint8)
        data=buffers[src['role']][1]
        data[src['offset']:src['offset']+packed.size]=packed.reshape(-1)
        wb=by_arg[8]
        layout=weights[wb['offset']:wb['offset']+wb['weight_size']].view(np.float16).reshape(channels//8,3,3,outputs,8)
        for output in range(outputs):
            channel=output%channels
            layout[channel//8,tap[0],tap[1],output,channel%8]=1
        if pattern=='dense':
            rng=np.random.default_rng(seed)
            values[:]=rng.integers(0,16,size=values.shape).astype(np.float16)/16
            packed=(values.view(np.uint16)>>7).astype(np.uint8) if encoded else values.view(np.uint8)
            data[src['offset']:src['offset']+packed.size]=packed.reshape(-1)
            layout[:]=rng.integers(-8,9,size=layout.shape).astype(np.float16)/128
            bias_values=rng.integers(-4,5,size=outputs).astype(np.float16)/32
            bias_binding=by_arg[56]
            weights[bias_binding['offset']:bias_binding['offset']+bias_binding['weight_size']]=bias_values.view(np.uint8)
        if pattern=='trace':
            data[:]=0
            for channel in range(channels):
                for ky in range(3):
                    for kx in range(3):
                        layout[channel//8,ky,kx,:,channel%8]=np.uint16(0x3000+channel*9+ky*3+kx).view(np.float16)
        if pattern=='nan':
            values[:]=0
            packed=(values.view(np.uint16)>>7).astype(np.uint8) if encoded else values.view(np.uint8)
            data[src['offset']:src['offset']+packed.size]=packed.reshape(-1)
            layout[:]=0
            bias_binding=by_arg[56]
            weights[bias_binding['offset']:bias_binding['offset']+bias_binding['weight_size']]=np.full(outputs,0x7e00,dtype=np.uint16).view(np.uint8)
        pooled=np.empty((outputs//8,pool_height,pool_width,8),dtype=np.float16)
        unpooled=np.empty((4,outputs//8,pool_height,pool_width,8),dtype=np.float16)
        for output in range(outputs):
            channel=output%channels
            for y in range(pool_height):
                for x in range(pool_width):
                    quad=[]
                    for dy,dx in ((0,0),(0,1),(1,0),(1,1)):
                        sy,sx=2*y+dy+tap[0]-1,2*x+dx+tap[1]-1
                        quad.append(values[channel//8,sy,sx,channel%8] if 0<=sy<height and 0<=sx<width else np.float16(0))
                    pooled[output//8,y,x,output%8]=max(quad)
                    for q,value in enumerate(quad):
                        unpooled[q,output//8,y,x,output%8]=value
        if pattern=='dense':
            pooled,unpooled=convolution_pool(values,layout,bias_values)
        if pattern=='nan':
            pooled.view(np.uint16)[:]=0x7fff
            unpooled.view(np.uint16)[:]=0x7fff
        for arg,tensor in ((24,pooled),(40,unpooled)):
            binding=by_arg[arg]
            base,data=buffers[binding['role']]
            expected_buffers[base+binding['offset']]=tensor.view(np.uint8).reshape(-1)
    for binding in launch['bindings']:
        if binding['kind']==1:
            address=buffers[binding['role']][0]+binding['offset']
        elif binding['kind']==4:
            address=weight_address+binding['offset']
        else:
            raise ValueError('convolution reference currently accepts buffers and weights only')
        struct.pack_into('<Q',args,binding['arg_offset'],address)
    steps=mma_count=ctas=0
    trace=[]
    first_destination=kernel.ins[kernel.mma_sites[0]]['ops'][0]
    def mma_trace(cta,site,idx,D,A,B,C):
        if pattern=='trace' and kernel.ins[kernel.mma_sites[site]]['ops'][0]==first_destination:
            pairs=[]
            for pair in range(8):
                word=int(cta.r[B[pair//4]][pair%4])
                pairs.extend([divmod((word&65535)-0x3000,9),divmod((word>>16)-0x3000,9)])
            trace.append({'site':site,'channel_tap_pairs':pairs})
    ids=itertools.product(*(range(n) for n in launch['grid'])) if all_ctas else [(0,0,0)]
    for ctaid in ids:
        cta=CTA(kernel,memory,args,ctaid,launch['block'],hooks={'mma':mma_trace})
        steps+=cta.run()
        mma_count+=cta.mma_count
        ctas+=1
    expected=np.float16(0.25 if pattern=='bias' else 0).view(np.uint16)
    written=0
    for addresses,data in memory.writes:
        if pattern in ('identity','dense','nan'):
            for address,row in zip(addresses,data):
                matched=False
                for base,reference in expected_buffers.items():
                    offset=int(address)-base
                    if 0<=offset and offset+len(row)<=len(reference):
                        if not np.array_equal(row,reference[offset:offset+len(row)]):
                            raise AssertionError(('stage tensor mismatch',hex(int(address)),offset,row.tolist(),reference[offset:offset+len(row)].tolist()))
                        matched=True
                        break
                if not matched:
                    raise AssertionError(('write outside expected tensor',hex(int(address))))
        elif not np.all(data.copy().view(np.uint16)==expected):
            raise AssertionError(('unexpected zero-input convolution output',pattern,data[:4].tolist()))
        written+=data.size
    if not written or not mma_count:
        raise AssertionError('convolution did not execute and write output')
    if all_ctas and pattern in ('identity','dense','nan'):
        for base,reference in expected_buffers.items():
            actual=memory.read(np.array([base],dtype=np.uint64),len(reference))[0]
            if not np.array_equal(actual,reference):
                raise AssertionError('complete output tensor mismatch')
    receipt={'kernel':kernel.name,'pattern':pattern,'seed':seed,'tap':list(tap),'cta_count':ctas, 'steps':steps,'trace':trace,
            'mma_instructions':mma_count,'write_bytes':written,'scope':'synthetic stage tensors compared against lane-rotated CPU interpreter policy; no NVIDIA or native execution oracle',
            'ptx_sha256':hashlib.sha256(Path(ptx).read_bytes()).hexdigest()}
    if fixture_dir is not None:
        if not all_ctas or pattern not in ('identity','dense','nan'):
            raise ValueError('export requires complete identity/dense/NaN stage comparison')
        input_data=packed.tobytes()
        weight_data=layout.tobytes()
        bias_binding=by_arg[56]
        bias_data=weights[bias_binding['offset']:bias_binding['offset']+bias_binding['weight_size']].tobytes()
        header=struct.pack('<8I',width,height,channels,outputs,int(encoded),32,32+len(input_data),32+len(input_data)+len(weight_data))
        payload=header+input_data+weight_data+bias_data
        golden=pooled.tobytes()+unpooled.tobytes()
        directory=Path(fixture_dir)
        directory.mkdir(parents=True,exist_ok=True)
        (directory/'input.bin').write_bytes(payload)
        (directory/'expected.bin').write_bytes(golden)
        receipt.update(input_sha256=hashlib.sha256(payload).hexdigest(),expected_sha256=hashlib.sha256(golden).hexdigest(),
                       output_bytes=len(golden),groups=(pooled.size//2+63)//64)
        (directory/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('graph')
    parser.add_argument('ptx')
    parser.add_argument('--pattern',choices=('zero','bias','identity','dense','trace','nan'),default='zero')
    parser.add_argument('--tap',type=int,nargs=2,choices=range(3),default=(1,1))
    parser.add_argument('--all-ctas',action='store_true')
    parser.add_argument('--fixture-dir')
    parser.add_argument('--seed',type=int,default=1729)
    args=parser.parse_args()
    print(json.dumps(run(args.graph,args.ptx,args.pattern,args.tap,args.all_ctas,args.fixture_dir,args.seed),indent=2))
