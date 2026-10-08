"""Check every packed word by decoding destination coordinates (inverse mapping)."""
import random
import struct
import subprocess
import sys

canonical = random.Random(1741).randbytes(1903872)
raw = subprocess.check_output([sys.argv[1]], input=canonical)
offset = 0
checked = 0
for network in range(2):
    count, size = struct.unpack_from('<II',raw,offset)
    offset += 8
    entries = []
    for _ in range(count):
        entries.append(struct.unpack_from('<7I',raw,offset))
        offset += 28
    packed = raw[offset:offset+size]
    offset += size
    written = bytearray(size)
    for destination, length, source, outputs, inputs, kernel, layout in entries:
        assert destination+length <= size and source+length <= len(canonical)
        if layout in (0,3):
            assert packed[destination:destination+length] == canonical[source:source+length]
        else:
            assert length == 2*outputs*inputs*kernel*kernel and inputs % 8 == 0
            for word in range(length//2):
                residual, lane = divmod(word,8)
                residual, output = divmod(residual,outputs)
                if layout == 1:
                    residual, x = divmod(residual,kernel)
                    group, y = divmod(residual,kernel)
                elif layout == 2:
                    residual, group = divmod(residual,inputs//8)
                    y, x = divmod(residual,kernel)
                else:
                    raise AssertionError(f'unknown layout {layout}')
                channel = group*8+lane
                source_word = ((output*kernel+y)*kernel+x)*inputs+channel
                a = destination+2*word
                b = source+2*source_word
                assert packed[a:a+2] == canonical[b:b+2], (network,layout,word)
                checked += 1
        written[destination:destination+length] = b'\1'*length
    assert all(byte == 0 for byte, used in zip(packed,written) if not used), 'padding not zero'
assert offset == len(raw)
for malformed in (b'', canonical[:-1], canonical+b'\0'):
    result = subprocess.run([sys.argv[1]], input=malformed, capture_output=True)
    assert result.returncode == 1 and not result.stdout
print(f'Both network layouts: {checked} reordered words, copy regions, padding and malformed lengths passed')
