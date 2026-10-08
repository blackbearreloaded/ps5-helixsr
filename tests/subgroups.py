"""Execute whole 32-lane permutations, quads and reductions across multiple groups."""
import json
import hashlib
import os
from pathlib import Path
import random
import struct
import subprocess
import sys
import tempfile

runner,shader,probe=sys.argv[1:4]
report=Path(sys.argv[4]) if len(sys.argv)==5 else None
receipt={'schema':1,'qualification_passed':False,'shader_sha256':hashlib.sha256(Path(shader).read_bytes()).hexdigest()}
def save():
    if report:
        report.write_text(json.dumps(receipt,indent=2)+'\n')
env=dict(os.environ)
caps=json.loads(subprocess.check_output([probe],env=env))
receipt.update(capabilities=caps,vector_width_environment=env.get('LP_NATIVE_VECTOR_WIDTH'))
save()
assert caps['devices'] and caps['devices'][0]['compute_wave32_requestable'],caps
rng=random.Random(981321)
values=[rng.randrange(1,1000000) for _ in range(192)]
with tempfile.TemporaryDirectory() as temp:
    root=Path(temp);source=root/'input.bin';output=root/'output.bin'
    source.write_bytes(struct.pack('<192I',*values))
    result=subprocess.run([runner,shader,str(source),str(output),str(192*32),'3','--wave32'],
                          env=env,capture_output=True,text=True,timeout=90)
    receipt.update(runner_returncode=result.returncode,runner_stderr=result.stderr)
    save()
    assert result.returncode==0,result.stderr
    actual=list(struct.iter_unpack('<8I',output.read_bytes()))
    assert len(actual)==len(values)
    groups={}
    for i,row in enumerate(actual):
        assert row[0]==32 and row[7]==i,(i,row)
        lanes=groups.setdefault(row[6],{})
        assert row[1] not in lanes,(i,row)
        lanes[row[1]]=i
    receipt.update(expected_subgroups=6,observed_subgroups=len(groups),
                   observed_lane_indices={str(first):sorted(lanes) for first,lanes in groups.items()},
                   input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                   output_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    save()
    assert len(groups)==6,{'expected_groups':6,'observed_groups':len(groups),'lane_counts':[len(x) for x in groups.values()]}
    for lanes in groups.values():
        assert set(lanes)==set(range(32)),lanes
        assert len({i//64 for i in lanes.values()})==1,lanes
    for i,row in enumerate(actual):
        lanes=groups[row[6]];lane=row[1]
        expected=(values[lanes[31-lane]],values[lanes[lane^1]],values[lanes[lane^2]],sum(values[j] for j in lanes.values()))
        assert row[2:6]==expected,(i,row,expected,{'shuffle_source_index':values.index(row[2]) if row[2] in values else None})
receipt['qualification_passed']=True
save()
print('192 lanes: explicit wave32, reverse shuffles, X/Y quads and reductions pass')
