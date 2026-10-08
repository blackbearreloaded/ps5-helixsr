"""Lower the planner's full-range zero clear to Vulkan's buffer fill operation."""
import struct

def lower_clear(graph,launch):
    if launch['kernel']!='cuda_clear_buffer_kernel':
        raise ValueError('not the clear stage')
    args=bytes.fromhex(launch['args_hex'])
    if len(args)!=40 or len(launch['bindings'])!=1:
        raise ValueError('unexpected clear ABI')
    width,height,x,y,span_width,span_height,value,check=struct.unpack_from('<8I',args)
    binding=launch['bindings'][0]
    if binding['kind']!=1 or binding['access']!=2 or binding['arg_offset']!=32 or binding['offset']!=0:
        raise ValueError('unsupported clear binding')
    resource=graph['buffers'][binding['role']]
    if x or y or width!=span_width or height!=span_height or value!=0 or check!=1:
        raise ValueError('clear is not the supported full-range zero operation')
    if width*height!=resource['bytes'] or resource['bytes']%4:
        raise ValueError('clear range does not exactly cover a word-aligned buffer')
    return {'command':'vkCmdFillBuffer','role':binding['role'],'offset':0,'size':resource['bytes'],
            'data':0,'producer_stage':'TRANSFER','producer_access':'TRANSFER_WRITE',
            'consumer_stage':'COMPUTE_SHADER','consumer_access':'SHADER_READ|SHADER_WRITE'}
