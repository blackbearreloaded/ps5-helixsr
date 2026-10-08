from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from build_generated import adapt

source='''cbuffer PxArgs : register(b0) { uint4 PXP[2]; };
Texture2D<float4> PXT0 : register(t0); // param +280
SamplerState PXS0 : register(s0);
RWByteAddressBuffer PXW0 : register(u0); // param +376
RWTexture2D<float4> PXU0 : register(u1); // param +328
'''
result,bindings=adapt(source,{328:'rg16f'})
assert [b['binding'] for b in bindings]==list(range(5))
assert [b['kind'] for b in bindings]==['uniform-buffer','sampled-image','sampler','buffer','storage-image']
assert '[[vk::image_format("rg16f")]]' in result
assert bindings[4]['parameter_offset']==328
try:
    adapt(source)
    raise AssertionError('unknown storage-image format accepted')
except ValueError:
    pass
try:
    adapt(source+'StructuredBuffer<uint> Unknown : register(t9);\n',{328:'rg16f'})
    raise AssertionError('unparsed resource accepted')
except ValueError:
    pass
print('Dense descriptors, preserved argument mapping and explicit image-format rejection passed')
