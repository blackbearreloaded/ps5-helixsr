"""Rebuild the published main-path shaders from external setup/PTX inputs.

Generated network material remains under the selected ignored build directory.
This validates compilation and descriptor mapping, not image correctness.
"""
import argparse
import struct
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

NAMES=('k8_hdr_mvdiff_mvlo','luma_convert','reduce_sum','auto_exposure_copy','copy_exposure','clear_buffer',
       'k4','k7','k9','k10','k11','k5','k5f','k5n')
FAMILIES=('k4','k7','k9','k10','k11')
# A reduction whose last pass shuffles under a lane mask: the lanes outside the mask skip it in the
# translated kernel, and the lanes inside then read lanes that are not executing. D3D drivers return
# the skipped lanes' (zero) operand there; in SPIR-V the value is undefined, and lavapipe returns their
# partial sums. The lanes outside the mask take part with zero instead. (old, new) in NAME_body.h.
# NAME: (pattern, replacement) on NAME_body.h: after the ballot of the lanes inside the mask, the lanes
# outside it (the first predicate) zero the accumulator and enter the reduction's block instead of skipping it.
MASKED_SHUFFLES={'reduce_sum':(
    r'(= px_ballot\(x_p\d+\);\n\s+)if \((x_p\d+)\) rb\d+ = true;(\n\s+if \(\(!\2\)\) rb)(\d+)( = true;\n)',
    lambda m, accumulator: f'{m.group(1)}if ({m.group(2)}) {{ {accumulator} = 0x00000000U; rb{int(m.group(4)) + 1} = true; }}'
                           f'{m.group(3)}{m.group(4)}{m.group(5)}')}

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

HALF_VIEW_BINDING=100

def adapt(source,image_formats=None,half_stores=False):
    """Assign dense explicit Vulkan bindings, retaining argument-offset evidence.

    half_stores: the kernel stores 16-bit values into a byte-addressed buffer. DXC's SPIR-V
    lowering of RWByteAddressBuffer.Store<uint16_t> reads, modifies and writes the 32-bit word,
    so two lanes writing the two halves of one word lose one of the writes (DXIL stores 16 bits).
    Such buffers get a second, 16-bit view at the same binding (rebind_half_views) and the
    kernel's 16-bit stores go through it."""
    bindings=[]
    lines=[]
    pattern=re.compile(r'^(cbuffer|(?:RW)?ByteAddressBuffer|(?:RW)?Texture2D<[^>]+>|SamplerState)\s+(\w+).*register\(([btus])(\d+)\)')
    for line in source.splitlines():
        match=pattern.match(line)
        if match:
            kind,name,register,index=match.groups()
            if kind=='cbuffer': descriptor='uniform-buffer'
            elif 'ByteAddressBuffer' in kind: descriptor='buffer'
            elif kind=='SamplerState': descriptor='sampler'
            elif kind.startswith('RW'): descriptor='storage-image'
            else: descriptor='sampled-image'
            parameter=re.search(r'param \+(\d+)',line)
            binding=len(bindings)
            parameter_offset=int(parameter.group(1)) if parameter else None
            entry={'binding':binding,'name':name,'kind':descriptor,'register':register+index,'parameter_offset':parameter_offset}
            attributes=f'[[vk::binding({binding},0)]] '
            if descriptor=='storage-image':
                image_format=(image_formats or {}).get(parameter_offset)
                if image_format not in ('r16f','rg16f','rgba16f','r32f'):
                    raise ValueError(f'missing image format for {name} parameter {parameter_offset}')
                entry['image_format']=image_format
                attributes+=f'[[vk::image_format("{image_format}")]] '
            bindings.append(entry)
            line=attributes+line
            written=re.match(r'RWByteAddressBuffer PXW(\d+)$',kind+' '+name)
            if half_stores and written:
                entry['half_view']=True
                line+=(f'\n[[vk::binding({HALF_VIEW_BINDING+binding},0)]] RWStructuredBuffer<uint16_t> PXH{written.group(1)};'
                       f'   // the same buffer, 16-bit elements\n#define PXH16_{written.group(1)} PXH{written.group(1)}')
        elif re.search(r'\bregister\s*\(',line):
            raise ValueError(f'unrecognized resource declaration: {line}')
        elif half_stores and re.match(r'#include "\w+_body\.h"',line):
            line=('#undef PX_GST16\n#define PX_GST16(S, a, v) PXH16_##S[((uint)(a)) >> 1] = (uint16_t)(v)\n'+line)
        lines.append(line)
    if not bindings or bindings[0]['kind']!='uniform-buffer':
        raise ValueError('unexpected generated resource declarations')
    return '\n'.join(lines)+'\n',bindings

def rebind_half_views(path):
    """Give every 16-bit view the binding of the buffer it aliases; returns how many were moved."""
    data=path.read_bytes()
    words=list(struct.unpack(f'<{len(data)//4}I',data))
    index,moved=5,0
    while index<len(words):
        count,opcode=words[index]>>16,words[index]&0xffff
        if not count: raise ValueError('malformed SPIR-V')
        if opcode==71 and count==4 and words[index+2]==33 and words[index+3]>=HALF_VIEW_BINDING:   # OpDecorate Binding
            words[index+3]-=HALF_VIEW_BINDING;moved+=1
        index+=count
    path.write_bytes(struct.pack(f'<{len(words)}I',*words))
    return moved

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('setup','ptx','dxc','spirv-opt','compiler','out','graph'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--only',default=','.join(NAMES))
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    lib=args.setup/'lib';runtime=args.setup/'kernels/rt'
    env=dict(os.environ,HELIXSR_PTX_DIR=str(args.ptx.resolve()),HELIXSR_PTX_SM='89',
             HELIXSR_FAMILY_TABLES=str((args.out/'tables').resolve()))
    def execute(label,command,limit=120):
        result=subprocess.run(list(map(str,command)),env=env,capture_output=True,text=True,timeout=limit)
        (args.out/f'{label}.stdout').write_text(result.stdout)
        (args.out/f'{label}.stderr').write_text(result.stderr)
        if result.returncode:
            raise RuntimeError(f'{label}: exit {result.returncode}: {result.stderr[-1200:]} {result.stdout[-600:]}')
        return result
    selected=args.only.split(',')
    if not set(selected)<=set(NAMES): raise ValueError('unknown main-path kernel')
    synth=args.out/'launch_synth'
    records=[]
    manifest={'schema':1,'compiler_sha256':digest(args.compiler),'dxc_sha256':digest(args.dxc),
              'spirv_opt_sha256':digest(args.spirv_opt),
              'numeric_mode':'PX_FAST=1 with published per-kernel recipe flags',
              'source_hashes':{str(p.relative_to(args.setup)):digest(p) for p in sorted(args.setup.rglob('*'))
                               if p.suffix in ('.py','.cpp','.h','.inc','.hlsli','.json')},
              'kernels':records,'scope':'generation, SPIR-V validation and native compilation only; no execution qualification'}
    entry_tree=ast.parse((lib/'kernels.py').read_text())
    entries=next(ast.literal_eval(node.value) for node in entry_tree.body if isinstance(node,ast.Assign)
                 and any(isinstance(t,ast.Name) and t.id=='ENTRY' for t in node.targets))
    image_formats={}
    formats={10:'rgba16f',34:'rg16f',41:'r32f',54:'r16f'}
    for shape in ((192,144,128,96),(256,192,128,96),(128,96,128,96)):
        graph=json.loads(subprocess.check_output([str(args.graph),*map(str,shape),'0','1']))
        for launch in graph['launches']:
            image_formats[launch['kernel']]={b['arg_offset']:formats[graph['images'][b['role']]['dxgi_format']]
                                             for b in launch['bindings'] if b['kind']==3}
    family=[name for name in selected if name in FAMILIES]
    if family:
        execute('build-synth',['g++','-std=c++17','-O2','-I',lib/'model',lib/'model/model.cpp',lib/'model/launch_synth.cpp','-o',synth])
        execute('trace-tables',[sys.executable,'-u',lib/'trace_tables.py',args.out/'tables',*family,'--model',synth])
        print('Reconstructed family tables',flush=True)
    for name in selected:
        record={'name':name,'success':False}
        try:
            if name=='clear_buffer':
                record.update(success=True,implementation='vkCmdFillBuffer',
                              contract='tools/clear_operation.py validates exact full-buffer zero fill')
                records.append(record)
                (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
                print(name+': BUILTIN full-range zero fill',flush=True)
                continue
            execute(name+'-generate',[sys.executable,'-u',lib/'regen_kernels.py',args.out/'generated','--only',name])
            directory=args.out/'generated'/name
            original=directory/f'{name}.hlsl'
            if name in MASKED_SHUFFLES:
                body=directory/f'{name}_body.h';text=body.read_text();pattern,rewrite=MASKED_SHUFFLES[name]
                # the accumulator is the register the block after the ballot loads from shared memory
                found=re.findall(pattern,text);loaded=re.findall(r'(x_f\d+) = LDS32\(',text)
                if len(found)!=1 or len(set(loaded))!=1:
                    raise RuntimeError(f'{name}: the masked reduction is not where it is expected')
                body.write_text(re.sub(pattern,lambda m:rewrite(m,loaded[0]),text))
            half_stores='PX_GST16(' in (directory/f'{name}_body.h').read_text()
            adapted,bindings=adapt(original.read_text(),image_formats.get(entries[name],{}),half_stores)
            source=directory/f'{name}-vulkan.hlsl';source.write_text(adapted)
            spirv=directory/f'{name}.spv'
            record['bindings']=bindings
            waves=re.findall(r'\[WaveSize\((\d+)\)\]',adapted)
            record['required_subgroup_size']=int(waves[0]) if waves else None
            record['generated_source_hashes']={p.name:digest(p) for p in sorted(directory.iterdir())
                                               if p.suffix in ('.h','.hlsl','.hlsli')}
            dxc_command=[args.dxc,'-spirv','-fspv-target-env=vulkan1.2','-T','cs_6_6','-E','main',
                         '-enable-16bit-types','-HV','2021','-O3','-D','PX_FAST=1',
                         '-I',runtime,'-I',directory,source,'-Fo',spirv]
            dxc_result=subprocess.run(list(map(str,dxc_command)),env=env,capture_output=True,text=True,timeout=60)
            (args.out/f'{name}-dxc.stdout').write_text(dxc_result.stdout)
            (args.out/f'{name}-dxc.stderr').write_text(dxc_result.stderr)
            record['dxc_stderr']=dxc_result.stderr
            record['compile_strategy']='dxc-o3'
            if dxc_result.returncode:
                if name!='k5n':
                    raise RuntimeError(f'{name}-dxc: exit {dxc_result.returncode}: '
                                       f'{dxc_result.stderr[-1200:]} {dxc_result.stdout[-600:]}')
                record.update(compile_strategy='dxc-fcgl-external-minimal-legalization',
                              dxc_o3_returncode=dxc_result.returncode,
                              legalization_passes=['compact-ids','fix-storage-class',
                                                   'eliminate-dead-functions','trim-capabilities','compact-ids'])
                prelegal=directory/f'{name}-prelegal.spv'
                compact=directory/f'{name}-compact.spv'
                fixed=directory/f'{name}-fixed.spv'
                fcgl=execute(name+'-fcgl',[args.dxc,'-spirv','-fspv-target-env=vulkan1.2',
                    '-T','cs_6_6','-E','main','-enable-16bit-types','-HV','2021','-fcgl','-Vd',
                    '-D','PX_FAST=1','-I',runtime,'-I',directory,source,'-Fo',prelegal],60)
                record['dxc_fcgl_stderr']=fcgl.stderr
                execute(name+'-compact',[args.spirv_opt,'--target-env=vulkan1.2','--skip-validation',
                                         '--compact-ids',prelegal,'-o',compact])
                execute(name+'-fix-storage-class',[args.spirv_opt,'--target-env=vulkan1.2',
                                                    '--skip-validation','--fix-storage-class',
                                                    compact,'-o',fixed])
                execute(name+'-validate-fixed',['spirv-val','--target-env','vulkan1.2',fixed])
                execute(name+'-prune',[args.spirv_opt,'--target-env=vulkan1.2',
                                        '--eliminate-dead-functions','--trim-capabilities',
                                        '--compact-ids',fixed,'-o',spirv])
                for temporary in (prelegal,compact,fixed):
                    temporary.unlink()
            if half_stores:
                views=sum(1 for b in bindings if b.get('half_view'))
                if rebind_half_views(spirv)!=views: raise RuntimeError('16-bit view rebinding mismatch')
                record['half_views']=views
            execute(name+'-validate',['spirv-val','--target-env','vulkan1.2',spirv])
            descriptors=[f'0:{b["binding"]}:{b["kind"]}:1' for b in bindings]
            result=execute(name+'-native',[args.compiler,spirv,*descriptors],180 if name=='k5n' else 60)
            native=[json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
            if not any(r.get('result')=='success' for r in native) or not any(r.get('adapter_result')==0 for r in native):
                raise RuntimeError('native adapter did not report success')
            record.update(success=True,spirv_sha256=digest(spirv),native=native,native_stderr=result.stderr,
                          source_sha256=digest(source),ptx_sha256=digest(args.ptx/f'{entries[name]}.sm_89.ptx'),
                          capability_warnings='Unsupported SPIR-V capability' in result.stderr)
        except (RuntimeError,subprocess.TimeoutExpired,ValueError) as error:
            record['error']=str(error)
        records.append(record)
        (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print(f'{name}: '+('PASS' if record['success'] else record['error']),flush=True)
    return 0 if all(r['success'] for r in records) else 1

if __name__=='__main__':
    sys.exit(main())
