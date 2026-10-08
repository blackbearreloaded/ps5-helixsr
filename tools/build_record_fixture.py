"""Emit a compact line protocol for host Vulkan object/command recording."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from prepare_bindings import prepare
from runtime_sequence import RuntimeContext

KINDS={'uniform-buffer':'U','buffer':'B','sampled-image':'I','storage-image':'W','sampler':'S'}
STAGES={'TOP_OF_PIPE':1,'COMPUTE_SHADER':0x800,'TRANSFER':0x1000,'HOST':0x4000}
ACCESS={('HOST','read'):0x2000,('HOST','write'):0x4000,('HOST','read-write'):0x6000,
        ('TRANSFER','read'):0x800,('TRANSFER','write'):0x1000,('TRANSFER','read-write'):0x1800,
        ('COMPUTE_SHADER','read'):0x20,('COMPUTE_SHADER','write'):0x40,('COMPUTE_SHADER','read-write'):0x60}

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--planner',type=Path,required=True);parser.add_argument('--inventory',type=Path,required=True)
    parser.add_argument('--generated',type=Path,required=True);parser.add_argument('--compact-inventory',type=Path,required=True)
    parser.add_argument('--compact-dir',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--width',type=int,default=192);parser.add_argument('--height',type=int,default=144)
    parser.add_argument('--render-width',type=int,default=128);parser.add_argument('--render-height',type=int,default=96)
    args=parser.parse_args()
    inventory=json.loads(args.inventory.read_text());compact=json.loads(args.compact_inventory.read_text())
    def make(frame):
        graph=json.loads(subprocess.check_output([args.planner,str(args.width),str(args.height),
            str(args.render_width),str(args.render_height),str(frame),'1']))
        return graph,prepare(graph,inventory,args.generated,compact,args.compact_dir)
    g0,b0=make(0);g1,b1=make(1);context=RuntimeContext([g0,g1],[b0,b1]);record=context.record(g0,b0)
    resources={name:i for i,name in enumerate(sorted(context.resources))}
    pipelines={name:i for i,name in enumerate(sorted(context.pipelines))}
    lines=['HELIXSR_RECORD_V1']
    for name,index in sorted(resources.items(),key=lambda x:x[1]):
        resource=context.resources[name]
        if resource['kind']=='buffer':
            usage=(1 if 'storage' in resource['usage'] else 0)|(2 if 'uniform' in resource['usage'] else 0)|\
                  (4 if 'transfer-src' in resource['usage'] else 0)|(8 if 'transfer-dst' in resource['usage'] else 0)
            lines.append(f'R {index} B {resource["bytes"]} {usage} {resource["owner"]}')
        else:
            lines.append(f'R {index} I {resource["size"][0]} {resource["size"][1]} {resource["format"]} {resource["owner"]}')
    for name,index in sorted(pipelines.items(),key=lambda x:x[1]):
        shader,sha,kinds,wave=context.pipelines[name]
        path=(args.compact_dir/(name+'.spv')) if name in {x['name'] for x in compact['shaders']} else (args.generated/name/(name+'.spv'))
        if digest(path)!=sha: raise ValueError('pipeline SPIR-V identity mismatch: '+name)
        lines.append(f'P {index} {name} {path.resolve()} {len(kinds)} '+''.join(KINDS[x] for x in kinds))
    uniform_data={x['offset']:x['data_hex'] for x in record['uniform_writes']}
    for offset,data in sorted(uniform_data.items()): lines.append(f'U {offset} {data}')
    dispatch_sets={command['descriptor_set']:command for command in record['commands'] if command['command']=='dispatch'}
    if set(dispatch_sets)!=set(range(len(context.descriptor_slots))): raise ValueError('non-dense descriptor slots')
    for _,command in sorted(dispatch_sets.items()):
        lines.append(f'S {command["descriptor_set"]} {pipelines[command["pipeline"]]} {len(command["descriptors"])}')
        for d in command['descriptors']:
            kind=KINDS[d['kind']]
            if kind=='S': lines.append(f'D {command["descriptor_set"]} {d["binding"]} S {d["filter"]}')
            elif kind in 'IW': lines.append(f'D {command["descriptor_set"]} {d["binding"]} {kind} {resources[d["resource"]]}')
            else: lines.append(f'D {command["descriptor_set"]} {d["binding"]} {kind} {resources[d["resource"]]} {d["offset"]} {d["range"]}')
    for command in record['commands']:
        kind=command['command']
        if kind=='transition-images':
            lines.append('C T '+str(STAGES[command['src_stage']])+' '+str(STAGES[command['dst_stage']])+' '+
                         str(0 if command['src_access']=='none' else ACCESS[(command['src_stage'],command['src_access'])])+' '+
                         str(ACCESS[(command['dst_stage'],command['dst_access'])])+' '+str(len(command['resources']))+' '+
                         ' '.join(str(resources[x]) for x in command['resources']))
        elif kind=='fill-buffer':
            lines.append(f'C F {resources[command["resource"]]} {command["offset"]} {command["size"]} {command["value"]}')
        elif kind=='copy-zero-buffer-to-image':
            width,height=command['size'];lines.append(f'C C {resources[command["source"]]} {resources[command["resource"]]} {width} {height}')
        elif kind=='barrier':
            fields=[]
            for barrier in command['barriers']:
                fields.extend((str(resources[barrier['resource']]),str(STAGES[barrier['src_stage']]),
                    str(ACCESS[(barrier['src_stage'],barrier['src_access'])]),str(STAGES[barrier['dst_stage']]),
                    str(ACCESS[(barrier['dst_stage'],barrier['dst_access'])])))
            lines.append('C B '+str(len(command['barriers']))+' '+' '.join(fields))
        elif kind=='dispatch':
            lines.append(f'C D {command["descriptor_set"]} {command["grid"][0]} {command["grid"][1]} {command["grid"][2]}')
        else: raise ValueError('unknown runtime command')
    lines.append('E')
    args.output.write_text('\n'.join(lines)+'\n')
    print(json.dumps({'resources':len(resources),'pipelines':len(pipelines),'sets':len(context.descriptor_slots),
                      'commands':len(record['commands']),'uniform_bytes':context.uniform_bytes,
                      'fixture_sha256':digest(args.output)},sort_keys=True))

if __name__=='__main__': main()
