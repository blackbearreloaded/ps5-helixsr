import copy
import json
from pathlib import Path
import struct
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from clear_operation import lower_clear

for shape in ((192,144,128,96),(1919,1079,1279,719),(3840,2160,1920,1080)):
    graph=json.loads(subprocess.check_output([sys.argv[1],*map(str,shape),'0','1']))
    launch=next(p for p in graph['launches'] if p['kernel']=='cuda_clear_buffer_kernel')
    operation=lower_clear(graph,launch)
    assert operation['size']==2*graph['padded'][0]*graph['padded'][1]
    for offset,value in ((8,1),(16,1),(24,255),(28,0)):
        invalid=copy.deepcopy(launch)
        args=bytearray.fromhex(invalid['args_hex']);struct.pack_into('<I',args,offset,value)
        invalid['args_hex']=args.hex()
        try:
            lower_clear(graph,invalid)
            raise AssertionError('unsupported clear accepted')
        except ValueError:
            pass
print('Full-range zero clear lowering passed at three shapes; subrect and fill-value changes rejected')
