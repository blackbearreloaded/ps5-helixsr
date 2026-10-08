"""Lower the four compact early graph launches to resident buffer dispatches."""
import hashlib
import json
from pathlib import Path
import struct

import conv_kernels

FUSED='dltss_nchw8_conv_3x3_pool_conv_1x1_pool_512_032_008_fp16_e5m3_kernel'
STAGES={
 'dltss_nchw8_conv_3x3_pool_128_064_008_e5m3_fp16_kernel':('conv_64_128_k3_e5',64,128,16),
 'dltss_nchw8_conv_3x3_pool_128_064_008_fp16_fp16_kernel':('conv_128_128_k3_half',128,128,32),
 'dltss_nchw8_conv_3x3_pool_064_064_008_fp16_fp16_kernel':('conv_128_256_k3_half',128,256,64),
}
SHADER_KINDS={
 'conv_16_32_k3_half':['uniform-buffer','buffer','buffer','buffer','buffer'],
 'conv_32_64_k1_half':['uniform-buffer','buffer','buffer','buffer','buffer'],
 'conv_64_128_k3_e5':['uniform-buffer','buffer','buffer','buffer','buffer'],
 'conv_128_128_k3_half':['uniform-buffer','buffer','buffer','buffer','buffer'],
 'conv_128_256_k3_half':['uniform-buffer','buffer','buffer','buffer','buffer'],
 'encode_e5m3_resident':['uniform-buffer','buffer','buffer'],
}
def require(value,message):
    if not value: raise ValueError(message)
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def verify_inventory(root,shader_dir,manifest):
    require(manifest.get('schema')==1 and not manifest.get('execution_qualified'),'invalid compact inventory scope')
    records={x['name']:x for x in manifest['shaders']}
    require(set(records)==set(SHADER_KINDS) and len(records)==len(manifest['shaders']),'incomplete compact shader set')
    for relative,wanted in manifest['source_hashes'].items():
        require(Path(relative).as_posix() in ('tools/conv_kernels.py',
          'shaders/encode_e5m3_resident.hlsl','third_party/helixsr/common/fp16.h'),'unexpected compact source')
        require(digest(root/relative)==wanted,'compact source digest mismatch: '+relative)
    for name,kinds in SHADER_KINDS.items():
        record=records[name]
        require(record['descriptors']==kinds,'compact descriptor receipt mismatch: '+name)
        require(record['adapter'].get('adapter_result')==0 and record['adapter'].get('wave_size')==32 and
                record['adapter'].get('adapter_descriptors')==len(kinds),'compact native receipt mismatch: '+name)
        require(digest(shader_dir/(name+'.spv'))==record['spirv_sha256'],'compact SPIR-V digest mismatch: '+name)
    return records
def binding(launch,offset):
    values=[x for x in launch['bindings'] if x['arg_offset']==offset]
    require(len(values)==1,'missing or duplicate compact binding')
    return values[0]
def checked_buffer(graph,item,access=None):
    require(item['kind']==1 and 0<=item['role']<len(graph['buffers']) and item['offset']%4==0,'invalid compact buffer')
    if access is not None: require(item['access']==access,'invalid compact buffer access')
    size=graph['buffers'][item['role']]['bytes']
    require(0<=item['offset']<size<=2**32 and size%4==0,'compact buffer range invalid')
    return {'resource':f'buffer:{item["role"]}','range':size,'pointer_byte_offset':item['offset']}
def checked_weight(graph,item,size):
    require(item['kind']==4 and item['access']==1 and item['offset']%4==0 and item['weight_size']==size and
            item['offset']+size<=graph['weight_bytes']<=2**32,'invalid compact weight span')
    return item['offset']
def conv(records,name,width,height,input_resource,input_range,input_offset,weight_range,weight_offset,bias_offset,
         pool_resource,pool_range,pool_offset,skip_resource,skip_range,skip_offset,channels,outputs,kernel,input_e5,launch_index,kernel_name):
    require(width>=2 and height>=2 and not(width&1 or height&1),'invalid compact tensor dimensions')
    require(input_offset+channels*width*height*(1 if input_e5 else 2)<=input_range,'compact input exceeds resource')
    require(weight_range<=2**32 and weight_offset+channels*outputs*kernel*kernel*2<=weight_range and
            bias_offset+outputs*2<=weight_range,'compact weights exceed resource')
    require(weight_offset==conv_kernels.VARIANTS[name][5],'convolution weights are not where the kernel reads them')
    pw,ph=width//2,height//2;output_bytes=outputs*pw*ph*2
    require(pool_offset+output_bytes<=pool_range and skip_offset+4*output_bytes<=skip_range,'compact output exceeds resource')
    params=struct.pack('<8I',width,height,input_offset,weight_offset,bias_offset,pool_offset,skip_offset,0)
    words=output_bytes//4
    descriptors=[{'binding':0,'kind':'uniform-buffer','data_hex':params.hex(),'range':len(params)},
      {'binding':1,'kind':'buffer','resource':input_resource,'offset':0,'range':input_range,'access':'read'},
      {'binding':2,'kind':'buffer','resource':'weights','offset':0,'range':weight_range,'access':'read'},
      {'binding':3,'kind':'buffer','resource':pool_resource,'offset':0,'range':pool_range,'access':'write'},
      {'binding':4,'kind':'buffer','resource':skip_resource,'offset':0,'range':skip_range,'access':'write'}]
    return {'command':'dispatch','implementation':'compact-resident','shader':name,'spirv_sha256':records[name]['spirv_sha256'],
      'grid':conv_kernels.grid(name,width,height),'block':[64,1,1],'descriptors':descriptors,'launch_index':launch_index,
      'original_kernel':kernel_name,'arithmetic':{'channels':channels,'outputs':outputs,'kernel_size':kernel,'input_e5':input_e5}}
def encode(records,source,source_range,source_offset,target,target_range,target_offset,scalars,launch_index,kernel_name):
    require(scalars>0 and scalars%4==0 and source_offset+scalars*2<=source_range and target_offset+scalars<=target_range,
            'compact encoder exceeds resource')
    params=struct.pack('<4I',source_offset,target_offset,scalars,0);name='encode_e5m3_resident'
    return {'command':'dispatch','implementation':'compact-resident','shader':name,'spirv_sha256':records[name]['spirv_sha256'],
      'grid':[(scalars//4+63)//64,1,1],'block':[64,1,1],'launch_index':launch_index,'original_kernel':kernel_name,
      'descriptors':[{'binding':0,'kind':'uniform-buffer','data_hex':params.hex(),'range':len(params)},
       {'binding':1,'kind':'buffer','resource':source,'offset':0,'range':source_range,'access':'read'},
       {'binding':2,'kind':'buffer','resource':target,'offset':0,'range':target_range,'access':'write'}]}
def barrier(resources,launch_index,kernel_name):
    return {'command':'barrier','resources':resources,'src_stage':'COMPUTE_SHADER','src_access':'SHADER_WRITE',
            'dst_stage':'COMPUTE_SHADER','dst_access':'SHADER_READ|SHADER_WRITE','launch_index':launch_index,
            'original_kernel':kernel_name}
def lower(graph,launch,index,records,transients):
    name=launch['kernel'];wp,hp=graph['padded'];weight_bytes=graph['weight_bytes']
    if name==FUSED:
        require(launch['block']==[256,1,1],'unexpected fused workgroup')
        src=checked_buffer(graph,binding(launch,0),1)
        w1=checked_weight(graph,binding(launch,8),16*32*3*3*2);b1=checked_weight(graph,binding(launch,88),32*2)
        w2=checked_weight(graph,binding(launch,40),32*64*2);b2=checked_weight(graph,binding(launch,176),64*2)
        out1=checked_buffer(graph,binding(launch,80),2);pool2=checked_buffer(graph,binding(launch,160),2);skip2=checked_buffer(graph,binding(launch,168),2)
        require(pool2['resource']==skip2['resource'] and pool2['range']==skip2['range'],'fused outputs must share B4')
        pixels=wp*hp;a_size=5*pixels;b_size=5*pixels//2
        transients.update({'fused_fp16_a':a_size,'fused_fp16_b':b_size})
        # ponytail: two FP16 scratch buffers retain the already-proven arithmetic; fuse E5 stores only after profiling.
        commands=[conv(records,'conv_16_32_k3_half',wp//4,hp//4,src['resource'],src['range'],src['pointer_byte_offset'],weight_bytes,w1,b1,
                       'transient:fused_fp16_a',a_size,0,'transient:fused_fp16_a',a_size,pixels,16,32,3,False,index,name),
                  barrier(['transient:fused_fp16_a'],index,name)]
        commands += [encode(records,'transient:fused_fp16_a',a_size,pixels,out1['resource'],out1['range'],out1['pointer_byte_offset'],2*pixels,index,name),
                     conv(records,'conv_32_64_k1_half',wp//8,hp//8,'transient:fused_fp16_a',a_size,0,weight_bytes,w2,b2,
                          'transient:fused_fp16_b',b_size,0,'transient:fused_fp16_b',b_size,pixels//2,32,64,1,False,index,name),
                     barrier(['transient:fused_fp16_b'],index,name),
                     encode(records,'transient:fused_fp16_b',b_size,0,pool2['resource'],pool2['range'],pool2['pointer_byte_offset'],pixels//4,index,name),
                     encode(records,'transient:fused_fp16_b',b_size,pixels//2,skip2['resource'],skip2['range'],skip2['pointer_byte_offset'],pixels,index,name),
                     barrier([out1['resource'],pool2['resource']],index,name)]
        return commands
    require(name in STAGES,'not a compact early stage')
    shader,channels,outputs,divisor=STAGES[name]
    require(launch['block']==[128,1,1],'unexpected standalone workgroup')
    src=checked_buffer(graph,binding(launch,0),1);pool=checked_buffer(graph,binding(launch,24),2);skip=checked_buffer(graph,binding(launch,40),2)
    weight=checked_weight(graph,binding(launch,8),channels*outputs*3*3*2);bias=checked_weight(graph,binding(launch,56),outputs*2)
    duplicate=binding(launch,264)
    require(duplicate['kind']==4 and duplicate['offset']==binding(launch,8)['offset'] and
            duplicate['weight_size']==binding(launch,8)['weight_size'],'standalone duplicate weight differs')
    command=conv(records,shader,wp//divisor,hp//divisor,src['resource'],src['range'],src['pointer_byte_offset'],weight_bytes,weight,bias,
                 pool['resource'],pool['range'],pool['pointer_byte_offset'],skip['resource'],skip['range'],skip['pointer_byte_offset'],
                 channels,outputs,3,shader.endswith('_e5'),index,name)
    return [command,barrier([pool['resource'],skip['resource']],index,name)]
