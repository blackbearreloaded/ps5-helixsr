import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

runner, shader, reference = sys.argv[1:]
raw = subprocess.check_output([reference])
expected = list(struct.iter_unpack('<3I',raw[:65536*12]))
decode_e5 = struct.unpack_from('<256I',raw,65536*12)
def float_equal_bits(actual, wanted):
    # NaN payload is not an arithmetic contract; class and all finite bits are.
    a = struct.unpack('<f',struct.pack('<I',actual))[0]
    b = struct.unpack('<f',struct.pack('<I',wanted))[0]
    return actual == wanted or (math.isnan(a) and math.isnan(b))
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    for values in (list(range(65536))+[0,0x3c00,0xffff], [0,1,0x3ff,0x400,0x7bff,0x7c00,0x7fff,0x8000,0xfc00]):
        source, destination = root/'input.bin',root/'output.bin'
        source.write_bytes(struct.pack('<'+'I'*len(values),*values))
        result = subprocess.run([runner,shader,str(source),str(destination),str(16*len(values)),str((len(values)+63)//64)],capture_output=True,text=True,timeout=90)
        assert result.returncode == 0, result.stderr
        print(result.stderr.strip())
        data=destination.read_bytes()
        assert len(data)==16*len(values)
        for h,actual in zip(values,struct.iter_unpack('<4I',data)):
            assert float_equal_bits(actual[0],expected[h][0]), (h,'half decode',actual,expected[h])
            assert actual[1:3]==expected[h][1:3], (h,'encode',actual,expected[h])
            assert float_equal_bits(actual[3],decode_e5[h&255]), (h,'E5 decode',actual)
        rejected = subprocess.run([runner,shader,str(source),str(destination),'16','0'],capture_output=True,text=True)
        assert rejected.returncode == 1 and 'positive integer' in rejected.stderr
print('Host SPIR-V arithmetic agrees with independently verified CPU helpers for all half encodings and partial workgroups')
