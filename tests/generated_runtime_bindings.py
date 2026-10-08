"""Bind actual pinned generated artifacts to the host planner over the shape matrix."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from prepare_bindings import prepare

planner,inventory_path,generated_path,compact_inventory_path,compact_dir,report_path=sys.argv[1:]
inventory=json.loads(Path(inventory_path).read_text());generated=Path(generated_path)
compact_inventory=json.loads(Path(compact_inventory_path).read_text());compact_dir=Path(compact_dir)
records={r['name']:r for r in inventory['kernels']}
results=[]
for shape in ((192,144,128,96),(1920,1080,1280,720),(1919,1079,1279,719),(256,192,128,96),(128,96,128,96)):
    for frame in (0,1,2):
        for exposure in (0,1):
            graph=json.loads(subprocess.check_output([planner,*map(str,shape),str(frame),str(exposure)]))
            result=prepare(graph,inventory,generated,compact_inventory,compact_dir)
            assert not result['execution_qualified']
            head=next(p['recipe'] for p in graph['launches'] if 'aniso_gaussian' in p['kernel'])
            assert result['complete_binding_coverage']
            assert not result['missing']
            covered={s['launch_index'] for s in result['steps']}|{s['launch_index'] for s in result['missing']}
            assert sorted(covered)==list(range(len(graph['launches'])))
            assert result['transient_buffers']==[
                {'resource':'transient:fused_fp16_a','bytes':5*graph['padded'][0]*graph['padded'][1]},
                {'resource':'transient:fused_fp16_b','bytes':5*graph['padded'][0]*graph['padded'][1]//2}]
            compact=[s for s in result['steps'] if s.get('implementation')=='compact-resident']
            assert len(compact)==8 and [s['shader'] for s in compact]==[
                'conv_16_32_k3_half','encode_e5m3_resident','conv_32_64_k1_half','encode_e5m3_resident',
                'encode_e5m3_resident','conv_64_128_k3_e5','conv_128_128_k3_half','conv_128_256_k3_half']
            for step in compact:
                assert [d['binding'] for d in step['descriptors']]==list(range(len(step['descriptors'])))
                assert all(isinstance(d['range'],int) and d['range']>0 for d in step['descriptors'])
                assert step['grid'][0]>0 and step['grid'][1]>0 and step['grid'][2]==1 and step['block']==[64,1,1]
                assert step['descriptors'][0]['range'] in (16,32)
                assert len(bytes.fromhex(step['descriptors'][0]['data_hex']))==step['descriptors'][0]['range']
            for step in result['steps']:
                if step['command']!='dispatch' or step.get('implementation')=='compact-resident':continue
                original=graph['launches'][step['launch_index']]
                args=bytes.fromhex(original['args_hex']);uniform=bytes.fromhex(step['uniform_hex'])
                assert uniform[-16:]==struct.pack('<4I',*original['grid'],0)
                assert len(uniform)==(len(args)+15)//16*16+16
                assert not any(uniform[len(args):-16])
                changed=set()
                for d in step['descriptors']:
                    if d['kind']!='buffer':continue
                    offset=d['parameter_offset'];changed.update(range(offset,offset+8))
                    original_binding=next(b for b in original['bindings'] if b['arg_offset']==offset)
                    ptr=struct.unpack_from('<Q',uniform,offset)[0]
                    assert ptr&0xffffffff==original_binding['offset']
                    assert d['offset']==0
                    if original_binding['kind']==4:
                        assert d['range']==graph['weight_bytes'] and d['weight_span']==original_binding['weight_size']
                    else:
                        assert d['range']==graph['buffers'][original_binding['role']]['bytes']
                for omitted in step['omitted_parameters']:
                    offset=omitted['parameter_offset'];changed.update(range(offset,offset+8))
                    assert uniform[offset:offset+8]==bytes(8)
                    assert offset in (256,480) and step['recipe'] in ('k5','k5f','k5n')
                assert all(uniform[i]==args[i] for i in range(len(args)) if i not in changed)
            results.append({'shape':shape,'frame':frame,'auto_exposure':exposure,'output_head':head,
                            'mapped_launches':len(result['steps']),'missing_launches':len(result['missing']),
                            'binding_contract_sha256':hashlib.sha256(json.dumps(result,sort_keys=True).encode()).hexdigest()})

# Exercise real asset identity rejection without touching the user's generated tree.
launch=next(p for p in graph['launches'] if p['recipe']=='k4')
single=copy.deepcopy(graph);single['launches']=[launch]
with tempfile.TemporaryDirectory() as temp:
    root=Path(temp);shutil.copytree(generated/'k4',root/'k4')
    path=root/'k4/k4_body.h';path.write_bytes(path.read_bytes()+b'\n')
    try:
        prepare(single,inventory,root,compact_inventory,compact_dir)
    except ValueError as error:
        assert 'digest mismatch' in str(error)
    else:
        raise AssertionError('modified generated body accepted')
    broken_compact=copy.deepcopy(compact_inventory);broken_compact['shaders'][0]['descriptors'][0]='buffer'
    try:
        prepare(graph,inventory,generated,broken_compact,compact_dir)
    except ValueError as error:
        assert 'descriptor receipt mismatch' in str(error)
    else:
        raise AssertionError('modified compact descriptor receipt accepted')
    complete=json.loads(subprocess.check_output([planner,'192','144','128','96','0','1']))
    broken_graph=copy.deepcopy(complete)
    stage=next(x for x in broken_graph['launches'] if x['kernel'].endswith('128_064_008_e5m3_fp16_kernel'))
    next(x for x in stage['bindings'] if x['arg_offset']==264)['offset']+=4
    try:
        prepare(broken_graph,inventory,generated,compact_inventory,compact_dir)
    except ValueError as error:
        assert 'duplicate weight differs' in str(error)
    else:
        raise AssertionError('divergent duplicate weight accepted')
    graph_path=root/'graph.json';graph_path.write_text(json.dumps(complete))
    accepted=subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'tools/prepare_bindings.py'),
        str(graph_path),inventory_path,generated_path,str(root/'complete.json'),'--compact-inventory',compact_inventory_path,
        '--compact-dir',str(compact_dir),'--require-complete'],capture_output=True,text=True)
    assert accepted.returncode==0,accepted.stderr
    assert json.loads((root/'complete.json').read_text())['complete_binding_coverage']
    below=json.loads(subprocess.check_output([planner,'128','96','128','96','0','1']))
    graph_path.write_text(json.dumps(below))
    below_accepted=subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'tools/prepare_bindings.py'),
        str(graph_path),inventory_path,generated_path,str(root/'contract.json'),'--compact-inventory',compact_inventory_path,
        '--compact-dir',str(compact_dir),'--require-complete'],capture_output=True,text=True)
    assert below_accepted.returncode==0,below_accepted.stderr
    assert json.loads((root/'contract.json').read_text())['complete_binding_coverage']

report={'schema':1,'cases':results,'tampered_source_rejected':True,'tampered_compact_receipt_rejected':True,
        'divergent_duplicate_weight_rejected':True,'strict_complete_accepted':True,'strict_k5n_accepted':True,'execution_qualified':False,
        'scope':'descriptor/argument identity, compact lowering and resource-range checks only; all output heads compile'}
Path(report_path).write_text(json.dumps(report,indent=2)+'\n')
print(f'{len(results)} complete generated graph bindings pass; source and receipt tampering rejected')
