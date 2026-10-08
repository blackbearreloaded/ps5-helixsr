"""Exercise persistent resources and hazard-complete full-frame command recording."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from prepare_bindings import prepare
from runtime_sequence import RuntimeContext

planner,inventory_path,generated_path,compact_inventory_path,compact_dir,report_path=sys.argv[1:]
inventory=json.loads(Path(inventory_path).read_text());generated=Path(generated_path)
compact_inventory=json.loads(Path(compact_inventory_path).read_text());compact_dir=Path(compact_dir)

def frame(shape,index,exposure=1):
    graph=json.loads(subprocess.check_output([planner,*map(str,shape),str(index),str(exposure)]))
    binding=prepare(graph,inventory,generated,compact_inventory,compact_dir)
    return graph,binding

def inspect(context,record):
    assert record['allocations_during_record']==record['submits']==record['readbacks']==0
    assert record['allocation_count']==context.allocation_count
    assert all(r['layout']=='GENERAL' for r in context.resources.values() if r['kind']=='image')
    assert not any(c['command'] in ('allocate','create-pipeline','submit','readback') for c in record['commands'])
    dispatches=[c for c in record['commands'] if c['command']=='dispatch']
    assert len(dispatches)==len(record['uniform_writes'])
    assert len({u['slot'] for u in record['uniform_writes']})==len(record['uniform_writes'])
    for uniform in record['uniform_writes']:
        assert uniform['offset']%256==0 and uniform['offset']+uniform['bytes']<=context.uniform_bytes
    for command in dispatches:
        assert command['slot'] in context.descriptor_slots
        assert command['descriptor_set']==context.descriptor_slots[command['slot']]
        assert command['pipeline'] in context.pipelines
        assert all(x>0 for x in command['grid']) and all(x>0 for x in command['block'])
        for descriptor in command['descriptors']:
            if descriptor['kind']=='sampler':
                assert descriptor['filter'] in ('linear','point') and descriptor['max_lod']==0
                continue
            resource=context.resources[descriptor['resource']]
            if resource['kind']=='buffer':
                assert 0<=descriptor['offset'] and descriptor['range']>0
                assert descriptor['offset']+descriptor['range']<=resource['bytes']
            else:
                assert descriptor['layout']=='GENERAL' and descriptor['size']==resource['size']
    for command in record['commands']:
        if command['command']=='barrier':
            assert command['barriers']
            for barrier in command['barriers']:
                assert barrier['resource'] in context.resources
                assert barrier['src_stage'] in ('HOST','TRANSFER','COMPUTE_SHADER')
                assert barrier['dst_stage'] in ('TRANSFER','COMPUTE_SHADER')
    return len(dispatches),sum(c['command']=='barrier' for c in record['commands'])

cases=[]
for shape in ((192,144,128,96),(1920,1080,1280,720),(1919,1079,1279,719),(256,192,128,96),(128,96,128,96)):
    g0,b0=frame(shape,0);g1,b1=frame(shape,1)
    context=RuntimeContext([g0,g1],[b0,b1])
    manifest=context.manifest()
    assert manifest['recording_contract']['requires_previous_dispatch_complete']
    assert manifest['recording_contract']['records_only']
    assert manifest['descriptor_sets']>0 and manifest['pipelines']
    before=context.allocation_count
    r0=context.record(g0,b0)
    count0,barriers0=inspect(context,r0)
    assert r0['motion_history']=={'read':'history:motion:a','write':'history:motion:b'}
    assert any(c['command']=='transition-images' for c in r0['commands'])
    assert sum(c['command']=='copy-zero-buffer-to-image' for c in r0['commands'])==5
    try: context.record(g0,b0)
    except ValueError as error: assert 'in flight' in str(error)
    else: raise AssertionError('in-flight context reuse accepted')
    context.complete()
    r1=context.record(g1,b1)
    count1,barriers1=inspect(context,r1)
    assert r1['motion_history']=={'read':'history:motion:b','write':'history:motion:a'}
    assert not any(c['command'] in ('transition-images','copy-zero-buffer-to-image') for c in r1['commands'])
    assert context.allocation_count==before
    context.complete()
    cases.append({'shape':shape,'allocations':before,'pipelines':len(context.pipelines),
                  'descriptor_sets':len(context.descriptor_slots),'uniform_bytes':context.uniform_bytes,
                  'frame0_dispatches':count0,'frame1_dispatches':count1,
                  'frame0_barriers':barriers0,'frame1_barriers':barriers1,
                  'manifest_sha256':hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()})

# Long fixed-size sequence: stable allocations, alternating motion histories,
# exposure refresh cadence, and an explicit camera-cut reset at frame 32.
shape=(192,144,128,96);g0,b0=frame(shape,0);g1,b1=frame(shape,1)
context=RuntimeContext([g0,g1],[b0,b1]);allocation_count=context.allocation_count
command_counts=[];dispatch_counts=[];reset_clears=[]
for index in range(64):
    graph,binding=frame(shape,index)
    record=context.record(graph,binding,reset=index==32)
    dispatches,barriers=inspect(context,record)
    assert record['motion_history']['read']==('history:motion:a' if index%2==0 else 'history:motion:b')
    assert context.allocation_count==allocation_count
    clears=sum(c['command']=='copy-zero-buffer-to-image' for c in record['commands'])
    assert clears==(5 if index in (0,32) else 0)
    command_counts.append(len(record['commands']));dispatch_counts.append(dispatches);reset_clears.append(clears)
    context.complete()

report={'schema':1,'cases':cases,'long_sequence':{'frames':64,'allocations':allocation_count,
        'allocation_count_stable':True,'command_count_range':[min(command_counts),max(command_counts)],
        'dispatch_count_range':[min(dispatch_counts),max(dispatch_counts)],'reset_clear_frames':[i for i,x in enumerate(reset_clears) if x]},
        'in_flight_reuse_rejected':True,'records_only':True,'execution_qualified':False,
        'scope':'persistent resource ownership, descriptor slots, uniform packing, barriers and full-frame command order'}
Path(report_path).write_text(json.dumps(report,indent=2)+'\n')
print('5 persistent layouts and 64 consecutive full-frame command streams pass')
