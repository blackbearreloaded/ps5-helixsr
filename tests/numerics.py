"""Independent oracle uses Python's IEEE half packing and exact binary64 sums.

Adding a half value to a half power-of-two scale needs fewer than 53 significant
bits, so binary64 preserves the exact pre-rounding value for this operation.
This tests conversion policy, not the full network's accumulation order.
"""
import math
import struct
import subprocess
import sys

def half(h):
    return struct.unpack('<e', struct.pack('<H', h))[0]
def encode(value):
    if math.isnan(value):
        return 0x7e00  # Published helper's canonical NaN policy.
    try:
        return struct.unpack('<H', struct.pack('<e', value))[0]
    except OverflowError:
        return 0xfc00 if value < 0 else 0x7c00
def f32bits(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]
def f32value(bits):
    return struct.unpack('<f', struct.pack('<I', bits))[0]

raw = subprocess.check_output([sys.argv[1]])
words = iter(struct.unpack('<' + 'I' * (len(raw)//4), raw))
for h in range(65536):
    decoded, roundtrip, e5 = next(words), next(words), next(words)
    expected = half(h)
    if math.isnan(expected):
        assert decoded & 0x7f800000 == 0x7f800000 and decoded & 0x7fffff
    else:
        assert decoded == f32bits(expected), (h, decoded)
    assert roundtrip == encode(expected), (h, roundtrip)
    pre_round = expected + half(h & 0xfc00) / 16
    assert e5 == (encode(pre_round) >> 7) & 255, (h, e5)
for b in range(256):
    actual = f32value(next(words))
    expected = half(b << 7)
    assert (math.isnan(actual) and math.isnan(expected)) or actual == expected, b
for h in range(0x7bff):
    midpoint = (half(h)+half(h+1))/2
    for delta in (-1, 0, 1):
        value = f32value(f32bits(midpoint)+delta)
        assert next(words) == encode(value), (h,delta,'positive')
        assert next(words) == encode(-value), (h,delta,'negative')
assert next(words, None) is None
print('65,536 half patterns; 256 E5M3 decodings; 190,458 signed midpoint-neighbor conversions passed')
