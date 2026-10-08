"""Prepare checked descriptors and argument blocks for generated main-path stages.

This is an offline resource contract, not a command recorder or execution result.
The compact early-stage ABI is integrated when its checked inventory is supplied.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
from contract import contract
from build_generated import adapt
from compact_runtime import FUSED,STAGES,lower as lower_compact,verify_inventory as verify_compact

FORMATS={10:'rgba16f',34:'rg16f',41:'r32f',54:'r16f'}

def require(condition,message):
    if not condition:
        raise ValueError(message)

def parameter_reads(body):
    """Reject dynamic argument indexing rather than assuming a parameter is dead."""
    reads=set()
    for width,arg in re.findall(r'PX_PARAM(32|64)\s*\(([^)]*)\)',body):
        require(re.fullmatch(r'\d+u?',arg.strip()),'dynamic argument access cannot certify omitted bindings')
        start=int(arg.strip().rstrip('u'))
        reads.update(range(start,start+int(width)//8))
    return reads

def prepare_launch(graph,launch,shader,source,body):
    require(shader.get('success') and shader['name']==launch['recipe'],'missing or mismatched compiled recipe')
    descriptors=shader['bindings']
    formats={b['arg_offset']:FORMATS[graph['images'][b['role']]['dxgi_format']]
             for b in launch['bindings'] if b['kind']==3 and 0<=b['role']<len(graph['images'])}
    adapted,declared_bindings=adapt(source,formats,'PX_GST16(' in body)
    require(declared_bindings==descriptors,'descriptor receipt differs from source declarations')
    require(hashlib.sha256(adapted.encode()).hexdigest()==shader['source_sha256'],'adapted shader identity mismatch')
    require([d['binding'] for d in descriptors]==list(range(len(descriptors))),'non-dense or duplicate descriptor bindings')
    require(len({d['name'] for d in descriptors})==len(descriptors),'duplicate descriptor names')
    require(descriptors[0]['kind']=='uniform-buffer' and descriptors[0]['name']=='PxArgs','missing argument buffer')
    require(sum(d['kind']=='uniform-buffer' for d in descriptors)==1,'multiple argument buffers')
    args=bytearray.fromhex(launch['args_hex'])
    declared=re.search(r'uint4 PXP\[(\d+)\]',source)
    require(declared and int(declared.group(1))*16==(len(args)+15)//16*16,'argument block size differs from shader')
    block=re.search(r'\[numthreads\((\d+),\s*(\d+),\s*(\d+)\)\]',source)
    require(block and list(map(int,block.groups()))==launch['block'],'workgroup differs from shader')
    wave=re.search(r'\[WaveSize\((\d+)\)\]',source)
    require(wave and int(wave.group(1))==shader.get('required_subgroup_size')==32,'missing or mismatched wave32 requirement')
    by_arg={b['arg_offset']:b for b in launch['bindings']}
    require(len(by_arg)==len(launch['bindings']),'duplicate graph parameter')
    for offset in by_arg:
        require(isinstance(offset,int) and offset>=0 and offset%8==0 and offset+8<=len(args),'invalid parameter span')
    prepared=[{'binding':0,'kind':'uniform-buffer','resource':'arguments','offset':0,'range':int(declared.group(1))*16+16}]
    mapped=set()
    declared_images={}
    for d in descriptors[1:]:
        if d['kind']=='sampler':
            continue
        offset=d['parameter_offset']
        require(offset in by_arg and offset not in mapped,'missing or repeated graph binding')
        b=by_arg[offset];mapped.add(offset)
        item={'binding':d['binding'],'kind':d['kind'],'parameter_offset':offset,'access':b['access']}
        if d['kind']=='buffer':
            slot=re.fullmatch(r'PX([RW])(\d+)',d['name'])
            require(slot and b['kind'] in (1,4),'buffer descriptor kind mismatch')
            require(b['access'] in ((1,) if slot.group(1)=='R' else (2,3)),'buffer access differs from declaration')
            if b['kind']==4:
                size=graph['weight_bytes'];resource='weights'
                require(b['access']==1 and b['weight_size']>0 and b['offset']+b['weight_size']<=size,'weight span outside blob')
                item['weight_span']=b['weight_size']
            else:
                require(0<=b['role']<len(graph['buffers']),'unknown buffer role')
                size=graph['buffers'][b['role']]['bytes'];resource=f'buffer:{b["role"]}'
            require(0<=b['offset']<size<=2**32 and size%4==0,'buffer range exceeds 32-bit shader addressing')
            # Bind the complete allocation. The original offset stays in the pointer,
            # avoiding both descriptor-offset alignment assumptions and double offsets.
            tagged=((int(slot.group(2))+1)<<40)|b['offset']
            struct.pack_into('<Q',args,offset,tagged)
            item.update(resource=resource,offset=0,range=size,pointer_byte_offset=b['offset'],tagged_pointer=tagged)
        else:
            require(d['kind'] in ('sampled-image','storage-image') and b['kind'] in (2,3),'image descriptor kind mismatch')
            require(0<=b['role']<len(graph['images']) and b['offset']==0,'invalid image binding')
            image=graph['images'][b['role']]
            require(image['dxgi_format'] in FORMATS and all(x>0 for x in image['size']),'unsupported image shape or format')
            if d['kind']=='sampled-image':
                require(b['access']==1 and re.fullmatch(r'PXT\d+',d['name']),'sampled image is not readonly')
                declared_images[d['name']]=(b,item)
                item['sampling']='integer-load'  # Upgraded below when a sampler exists.
            else:
                require(b['kind']==3 and b['access'] in (1,2,3) and re.fullmatch(r'PXU\d+',d['name']),'storage image mismatch')
                require(d.get('image_format')==FORMATS[image['dxgi_format']],'typed storage-image format mismatch')
            item.update(resource=f'image:{b["role"]}',format=FORMATS[image['dxgi_format']],size=image['size'],layout='GENERAL')
            # Generated texture/surface operations select slots statically. Preserve
            # opaque template tokens for control-flow tests; never use them as a VA.
        prepared.append(item)
    for d in descriptors[1:]:
        if d['kind']!='sampler':
            continue
        slot=re.fullmatch(r'PXS(\d+)',d['name'])
        require(slot and 'PXT'+slot.group(1) in declared_images,'sampler has no matching texture slot')
        binding,image=declared_images['PXT'+slot.group(1)]
        sampler=binding.get('sampler')
        require(sampler and sampler['filter'] in ('linear','point') and
                sampler['address_u']==sampler['address_v']=='clamp','missing supported sampler policy')
        image['sampling']='normalized-sample-level'
        prepared.append({'binding':d['binding'],'kind':'sampler','image_binding':image['binding'],
                         'filter':sampler['filter'],'address_u':'clamp','address_v':'clamp','address_w':'clamp',
                         'normalized_coordinates':True,'min_lod':0,'max_lod':0,'anisotropy':False,'compare':False})
    # Some published binding tables include legacy fields absent from generated
    # resource declarations. Require no parameter read before clearing their tokens.
    reads=parameter_reads(body)
    omitted=[]
    for offset in sorted(set(by_arg)-mapped):
        require(not reads.intersection(range(offset,offset+8)),'unmapped graph binding is read by shader')
        omitted.append({'parameter_offset':offset,'original_binding':by_arg[offset],'reason':'no descriptor and no parameter read in pinned generated body'})
        struct.pack_into('<Q',args,offset,0)
    args.extend(bytes((-len(args))%16))
    require(len(launch['grid'])==3 and all(isinstance(x,int) and 0<x<2**32 for x in launch['grid']),'invalid dispatch grid')
    args.extend(struct.pack('<4I',*launch['grid'],0))
    return {'command':'dispatch','recipe':shader['name'],'kernel':launch['kernel'],'occurrence':launch['occurrence'],
            'grid':launch['grid'],'block':launch['block'],'descriptors':sorted(prepared,key=lambda x:x['binding']),
            'uniform_hex':args.hex(),'omitted_parameters':omitted,'spirv_sha256':shader['spirv_sha256'],
            'required_subgroup_size':shader.get('required_subgroup_size'),'execution_qualified':False}

def prepare(graph,inventory,generated,compact_inventory=None,compact_dir=None,root=Path(__file__).resolve().parents[1]):
    graph=contract(graph)
    records={r['name']:r for r in inventory['kernels']}
    require(len(records)==len(inventory['kernels']),'duplicate inventory recipe')
    compact_records=verify_compact(root,compact_dir,compact_inventory) if compact_inventory is not None and compact_dir is not None else None
    steps=[];missing=[];verified=set();transients={}
    for index,launch in enumerate(graph['launches']):
        if 'native_operation' in launch:
            steps.append(dict(launch['native_operation'],launch_index=index));continue
        if launch['kernel']==FUSED or launch['kernel'] in STAGES:
            if compact_records is None:
                missing.append({'launch_index':index,'kernel':launch['kernel'],'recipe':None,'reason':'missing compact resident inventory'})
            else:
                steps.extend(lower_compact(graph,launch,index,compact_records,transients))
            continue
        name=launch['recipe'];shader=records.get(name)
        if not shader or not shader.get('success'):
            missing.append({'launch_index':index,'kernel':launch['kernel'],'recipe':name,'reason':shader.get('error','missing generated runtime ABI') if shader else 'missing generated runtime ABI'})
            continue
        directory=generated/name
        if name not in verified:
            require({name+'.hlsl',name+'-vulkan.hlsl',name+'_body.h',name+'_cfg.h'}<=set(shader['generated_source_hashes']),
                    'incomplete generated source identity')
            for filename,digest in shader['generated_source_hashes'].items():
                require(Path(filename).name==filename,'invalid generated filename')
                require(hashlib.sha256((directory/filename).read_bytes()).hexdigest()==digest,'generated source digest mismatch: '+filename)
            require(hashlib.sha256((directory/(name+'.spv')).read_bytes()).hexdigest()==shader['spirv_sha256'],'SPIR-V digest mismatch')
            verified.add(name)
        source=(directory/(name+'.hlsl')).read_text()
        body=(directory/(name+'_body.h')).read_text()
        steps.append(dict(prepare_launch(graph,launch,shader,source,body),launch_index=index))
    return {'schema':1,'complete_binding_coverage':not missing,'execution_qualified':False,
            'scope':'offline descriptor and argument contract; not a recorded or executed graph',
            'transient_buffers':[{'resource':'transient:'+name,'bytes':size} for name,size in sorted(transients.items())],
            'steps':steps,'missing':missing}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('graph',type=Path);parser.add_argument('inventory',type=Path)
    parser.add_argument('generated',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--compact-inventory',type=Path);parser.add_argument('--compact-dir',type=Path)
    parser.add_argument('--require-complete',action='store_true')
    a=parser.parse_args()
    compact=json.loads(a.compact_inventory.read_text()) if a.compact_inventory else None
    if bool(a.compact_inventory)!=bool(a.compact_dir): raise SystemExit('compact inventory and directory must be provided together')
    result=prepare(json.loads(a.graph.read_text()),json.loads(a.inventory.read_text()),a.generated,compact,a.compact_dir)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    if a.require_complete and not result['complete_binding_coverage']:
        raise SystemExit('incomplete runtime binding coverage')
