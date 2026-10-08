import copy
import hashlib
from pathlib import Path
import struct
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from build_generated import adapt
from prepare_bindings import prepare_launch

source='''cbuffer PxArgs : register(b0) { uint4 PXP[6]; uint4 PXGRID; };
ByteAddressBuffer PXR0 : register(t0); // param +0
ByteAddressBuffer PXR1 : register(t1); // param +8
RWByteAddressBuffer PXW2 : register(u0); // param +16
Texture2D<float4> PXT0 : register(t2); // param +24
SamplerState PXS0 : register(s0);
RWTexture2D<float4> PXU0 : register(u1); // param +32
Texture2D<float4> PXT1 : register(t3); // param +40
[WaveSize(32)]
[numthreads(64, 1, 1)]
'''
adapted,descriptors=adapt(source,{32:'rg16f'})
shader={'name':'test','success':True,'bindings':descriptors,'source_sha256':hashlib.sha256(adapted.encode()).hexdigest(),
        'spirv_sha256':'synthetic-structural-test','required_subgroup_size':32}
graph={'buffers':[{'bytes':64},{'bytes':128}],'weight_bytes':256,
       'images':[{'dxgi_format':10,'size':[8,4]},{'dxgi_format':34,'size':[8,4]}]}
sampler={'filter':'linear','address_u':'clamp','address_v':'clamp'}
launch={'recipe':'test','kernel':'test_kernel','occurrence':0,'block':[64,1,1],'grid':[7,2,1],
        'args_hex':bytes(range(96)).hex(),'bindings':[
    {'arg_offset':0,'kind':1,'role':0,'access':1,'offset':12,'weight_size':0},
    {'arg_offset':8,'kind':4,'role':0,'access':1,'offset':64,'weight_size':16},
    {'arg_offset':16,'kind':1,'role':1,'access':2,'offset':4,'weight_size':0},
    {'arg_offset':24,'kind':2,'role':0,'access':1,'offset':0,'weight_size':0,'sampler':sampler},
    {'arg_offset':32,'kind':3,'role':1,'access':2,'offset':0,'weight_size':0},
    {'arg_offset':40,'kind':3,'role':0,'access':1,'offset':0,'weight_size':0},
    {'arg_offset':48,'kind':1,'role':1,'access':0,'offset':0,'weight_size':0}]}
body='uint x=PX_PARAM32(64u); uint64_t y=PX_PARAM64(0u);'
result=prepare_launch(graph,launch,shader,source,body)
expected=bytearray(range(96))
struct.pack_into('<3Q',expected,0,0x1000000000c,0x20000000040,0x30000000004)
struct.pack_into('<Q',expected,48,0)
expected+=struct.pack('<4I',7,2,1,0)
assert bytes.fromhex(result['uniform_hex'])==expected
assert [d['binding'] for d in result['descriptors']]==list(range(8))
assert [(d['resource'],d['offset'],d['range']) for d in result['descriptors'][1:4]]==[
    ('buffer:0',0,64),('weights',0,256),('buffer:1',0,128)]
assert result['descriptors'][2]['weight_span']==16
assert result['descriptors'][4]['sampling']=='normalized-sample-level'
assert result['descriptors'][5]['normalized_coordinates'] is True
assert result['descriptors'][7]['sampling']=='integer-load'
assert result['descriptors'][7]['resource']=='image:0' # Readonly surface becomes texture.
assert [x['parameter_offset'] for x in result['omitted_parameters']]==[48]
assert result['execution_qualified'] is False

bad=[]
def reject(change):
    g,l,s=copy.deepcopy((graph,launch,shader))
    changed_source,changed_body=change(g,l,s)
    try:
        prepare_launch(g,l,s,changed_source,changed_body)
    except (ValueError,KeyError):
        bad.append(True)
    else:
        raise AssertionError('invalid runtime contract accepted')
def mutate_binding(index,field,value):
    def change(g,l,s):
        l['bindings'][index][field]=value
        return source,body
    return change
for values in ((0,'offset',64),(0,'offset',-1),(0,'role',99),(0,'access',2),
               (1,'weight_size',300),(1,'offset',-1),(2,'access',1),(3,'access',2),
               (3,'arg_offset',0),(4,'offset',4),(4,'kind',1),(5,'access',3)):
    reject(mutate_binding(*values))
reject(lambda g,l,s:(source,body+' uint z=PX_PARAM32(52u);'))
reject(lambda g,l,s:(source,body+' uint z=PX_PARAM32(dynamicOffset);'))
reject(lambda g,l,s:(source.replace('PXP[6]','PXP[7]'),body))
reject(lambda g,l,s:(source.replace('numthreads(64','numthreads(32'),body))
def bad_sampler(g,l,s):
    l['bindings'][3]['sampler']['filter']='unknown'
    return source,body
reject(bad_sampler)
def bad_descriptor(g,l,s):
    s['bindings'][1]['binding']=99
    return source,body
reject(bad_descriptor)
def bad_format(g,l,s):
    g['images'][1]['dxgi_format']=54
    return source,body
reject(bad_format)
def big_buffer(g,l,s):
    g['buffers'][0]['bytes']=2**32+4
    return source,body
reject(big_buffer)
def bad_grid(g,l,s):
    l['grid'][0]=0
    return source,body
reject(bad_grid)
def bad_wave(g,l,s):
    s['required_subgroup_size']=8
    return source,body
reject(bad_wave)
print(f'Exact uniform bytes, full-allocation ranges, image/sampler mapping and {len(bad)} rejection cases pass')
