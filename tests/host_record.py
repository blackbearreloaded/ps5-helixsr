"""Create and record a complete Vulkan frame for each generated output head."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

generator,runner,planner,inventory,generated,compact_inventory,compact_dir,report=sys.argv[1:]
cases=[]
for name,shape in (('below-1.5',(128,96,128,96)),('middle',(192,144,128,96)),('high',(256,192,128,96))):
    with tempfile.TemporaryDirectory() as temp:
        fixture=Path(temp)/'record.fixture'
        built=subprocess.run([sys.executable,generator,'--planner',planner,'--inventory',inventory,
            '--generated',generated,'--compact-inventory',compact_inventory,'--compact-dir',compact_dir,
            '--output',str(fixture),'--width',str(shape[0]),'--height',str(shape[1]),
            '--render-width',str(shape[2]),'--render-height',str(shape[3])],capture_output=True,text=True)
        assert built.returncode==0,built.stderr
        fixture_result=json.loads(built.stdout)
        recorded=subprocess.run([runner,str(fixture)],capture_output=True,text=True)
        assert recorded.returncode==0,recorded.stderr
        result=json.loads(recorded.stdout)
        assert result['result']=='success' and not result['submitted'] and not result['execution_qualified']
        assert result['resources']==fixture_result['resources']==25
        assert result['pipelines']==fixture_result['pipelines']==17
        assert result['descriptor_sets']==fixture_result['sets']==22
        assert result['dispatches']==22 and result['barriers']==24
        cases.append(dict(name=name,shape=shape,fixture_sha256=fixture_result['fixture_sha256'],**result))
receipt={'schema':1,'cases':cases,'actual_vulkan_objects':True,'actual_command_buffer':True,
         'submitted':False,'execution_qualified':False,
         'scope':'host Vulkan resource, pipeline, descriptor and command-buffer creation; no dispatch execution'}
Path(report).write_text(json.dumps(receipt,indent=2)+'\n')
print('all three output heads create and record one complete Vulkan command buffer')
