"""Build a persistent resource layout and one hazard-complete GPU command stream.

This is the backend-independent part of the runtime.  It consumes the checked
descriptor contract, assigns persistent descriptor/uniform slots and emits the
Vulkan operations that a native recorder must encode.  It never allocates while
recording a frame and never inserts a readback or submission.
"""
from collections import defaultdict

FORMAT_BYTES={'r16f':2,'rg16f':4,'rgba16f':8,'r32f':4}

def require(value,message):
    if not value: raise ValueError(message)

def _access(value):
    if value in (1,'read'): return 'read'
    if value in (2,'write'): return 'write'
    if value in (3,'read-write'): return 'read-write'
    raise ValueError('unsupported resource access')

def _merge(a,b):
    if a==b:return a
    return 'read-write'

def _writes(value): return value in ('write','read-write')

def _identity(steps,index):
    step=steps[index]
    siblings=[i for i,s in enumerate(steps[:index]) if s.get('launch_index')==step.get('launch_index') and
              s.get('command')=='dispatch']
    return ':'.join((step.get('original_kernel',step.get('kernel','unknown')),
                    str(step.get('occurrence',0)),step.get('shader',step.get('recipe','unknown')),str(len(siblings))))

class RuntimeContext:
    """Persistent fixed-resolution resources and command-recording state."""
    def __init__(self,graphs,contracts,uniform_alignment=256):
        require(graphs and len(graphs)==len(contracts),'missing runtime profiles')
        require(uniform_alignment>0 and uniform_alignment&(uniform_alignment-1)==0,'invalid uniform alignment')
        base=graphs[0]
        self.shape=(tuple(base['output']),tuple(base['render']),tuple(base['padded']))
        self.frame=0;self.initialized=False;self.in_flight=False;self.states={}
        for graph,contract in zip(graphs,contracts):
            require((tuple(graph['output']),tuple(graph['render']),tuple(graph['padded']))==self.shape,
                    'runtime profile geometry differs')
            require(contract['complete_binding_coverage'] and not contract['missing'],'incomplete binding contract')
            require(graph['buffers']==base['buffers'] and graph['images']==base['images'] and
                    graph['weight_bytes']==base['weight_bytes'],'runtime resource profile differs')
        self.resources={}
        self.resources['weights']={'kind':'buffer','bytes':base['weight_bytes'],'owner':'context','persistent':True,
                                   'usage':['storage']}
        for i,item in enumerate(base['buffers']):
            self.resources[f'buffer:{i}']={'kind':'buffer','bytes':item['bytes'],'owner':'context','persistent':False,
                                           'usage':['storage','transfer-src','transfer-dst'],'name':item['name']}
        transient={}
        for contract in contracts:
            for item in contract['transient_buffers']:
                old=transient.setdefault(item['resource'],item['bytes'])
                require(old==item['bytes'],'transient size differs between profiles')
        for name,size in transient.items():
            self.resources[name]={'kind':'buffer','bytes':size,'owner':'context','persistent':False,
                                  'usage':['storage']}
        max_image_bytes=0
        for i,item in enumerate(base['images']):
            name=f'image:{i}'
            owner='application' if item['game'] or item['name']=='OUTPUT' else 'context'
            if item['name']=='MV_HIST_READ': name='history:motion:a'
            elif item['name']=='MV_HIST_WRITE': name='history:motion:b'
            resource={'kind':'image','size':item['size'],'format':{10:'rgba16f',34:'rg16f',41:'r32f',54:'r16f'}[item['dxgi_format']],
                      'owner':owner,'persistent':item['persistent'],'layout':'GENERAL','name':item['name'],
                      'usage':['sampled','storage'] if owner=='context' else ['application']}
            self.resources[name]=resource
            if owner=='context': max_image_bytes=max(max_image_bytes,item['size'][0]*item['size'][1]*FORMAT_BYTES[resource['format']])
        self.resources['zero-images']={'kind':'buffer','bytes':max_image_bytes,'owner':'context','persistent':True,
                                       'usage':['transfer-src','transfer-dst']}
        slots={};pipelines={}
        for contract in contracts:
            for i,step in enumerate(contract['steps']):
                if step['command']!='dispatch':continue
                identity=_identity(contract['steps'],i)
                layout=tuple(d['kind'] for d in step['descriptors'])
                shader=step.get('shader',step.get('recipe'))
                pipeline=(shader,step['spirv_sha256'],layout,step.get('required_subgroup_size'))
                if identity in slots: require(slots[identity]['pipeline']==pipeline,'dispatch identity changed ABI')
                else: slots[identity]={'pipeline':pipeline}
                if shader in pipelines: require(pipelines[shader]==pipeline,'shader identity changed ABI')
                else:pipelines[shader]=pipeline
        self.pipelines=pipelines
        self.descriptor_slots={name:index for index,name in enumerate(sorted(slots))}
        self.uniform_offsets={name:index*uniform_alignment for index,name in enumerate(sorted(slots))}
        max_uniform=0
        for contract in contracts:
            for i,step in enumerate(contract['steps']):
                if step['command']!='dispatch':continue
                data=bytes.fromhex(step.get('uniform_hex',step['descriptors'][0].get('data_hex','')))
                require(data and len(data)==step['descriptors'][0]['range'],'invalid uniform payload')
                max_uniform=max(max_uniform,self.uniform_offsets[_identity(contract['steps'],i)]+len(data))
        self.uniform_bytes=(max_uniform+uniform_alignment-1)&-uniform_alignment
        self.resources['uniforms']={'kind':'buffer','bytes':self.uniform_bytes,'owner':'context','persistent':True,
                                    'usage':['uniform'],'host_visible':True}
        self.allocation_count=sum(r['owner']=='context' for r in self.resources.values())
        self.states['weights']={'stage':'HOST','access':'write'}

    def _physical(self,name):
        if name=='image:8': return 'history:motion:a' if self.frame%2==0 else 'history:motion:b'
        if name=='image:9': return 'history:motion:b' if self.frame%2==0 else 'history:motion:a'
        return name

    def _barriers(self,uses):
        barriers=[]
        for resource,new in sorted(uses.items()):
            previous=self.states.get(resource)
            if previous and (_writes(previous['access']) or _writes(new['access'])):
                barriers.append({'resource':resource,'src_stage':previous['stage'],'src_access':previous['access'],
                                 'dst_stage':new['stage'],'dst_access':new['access']})
            self.states[resource]=new
        return barriers

    def record(self,graph,contract,reset=False):
        require(not self.in_flight,'context dispatch still in flight')
        require((tuple(graph['output']),tuple(graph['render']),tuple(graph['padded']))==self.shape,
                'dispatch geometry differs from context')
        require(graph['frame']==self.frame,'planner frame differs from context sequence')
        require(contract['complete_binding_coverage'] and not contract['missing'],'incomplete dispatch contract')
        commands=[]
        # The caller has completed the previous use before these coherent writes.
        self.states['uniforms']={'stage':'HOST','access':'write'}
        internal_images=[name for name,r in self.resources.items() if r['kind']=='image' and r['owner']=='context']
        if not self.initialized:
            persistent=sorted(name for name in internal_images if self.resources[name]['persistent'])
            transient=sorted(set(internal_images)-set(persistent))
            commands.append({'command':'transition-images','resources':persistent,
                             'old_layout':'UNDEFINED','new_layout':'GENERAL','src_stage':'TOP_OF_PIPE',
                             'src_access':'none','dst_stage':'TRANSFER','dst_access':'write'})
            commands.append({'command':'transition-images','resources':transient,
                             'old_layout':'UNDEFINED','new_layout':'GENERAL','src_stage':'TOP_OF_PIPE',
                             'src_access':'none','dst_stage':'COMPUTE_SHADER','dst_access':'read-write'})
        if not self.initialized or reset or graph.get('reset'):
            commands.append({'command':'fill-buffer','resource':'zero-images','offset':0,
                             'size':self.resources['zero-images']['bytes'],'value':0})
            self.states['zero-images']={'stage':'TRANSFER','access':'write'}
            commands.append({'command':'barrier','barriers':self._barriers({'zero-images':{'stage':'TRANSFER','access':'read'}})})
            cleared=sorted(name for name in internal_images if self.resources[name]['persistent'])
            for name in cleared:
                commands.append({'command':'copy-zero-buffer-to-image','source':'zero-images','resource':name,
                                 'size':self.resources[name]['size'],'layout':'GENERAL'})
                self.states[name]={'stage':'TRANSFER','access':'write'}
        uniform_writes=[]
        explicit=set()
        for step in contract['steps']:
            if step['command']=='barrier': explicit.update(self._physical(r) for r in step['resources'])
        for i,step in enumerate(contract['steps']):
            if step['command']=='barrier':continue
            if step['command']=='vkCmdFillBuffer':
                resource=f'buffer:{step["role"]}'
                barriers=self._barriers({resource:{'stage':'TRANSFER','access':'write'}})
                if barriers:commands.append({'command':'barrier','barriers':barriers})
                commands.append({'command':'fill-buffer','resource':resource,'offset':step['offset'],
                                 'size':step['size'],'value':step['data']})
                continue
            require(step['command']=='dispatch','unknown prepared command')
            identity=_identity(contract['steps'],i);uses={};descriptors=[]
            data=bytes.fromhex(step.get('uniform_hex',step['descriptors'][0].get('data_hex','')))
            offset=self.uniform_offsets[identity]
            uniform_writes.append({'offset':offset,'bytes':len(data),'data_hex':data.hex(),'slot':identity})
            for descriptor in step['descriptors']:
                item=dict(descriptor)
                if item['kind']=='sampler':
                    descriptors.append(item);continue
                if item['binding']==0:
                    item.update(resource='uniforms',offset=offset,range=len(data),access='read')
                else:
                    item['resource']=self._physical(item['resource'])
                    item['access']=_access(item['access'])
                resource=item['resource']
                require(resource in self.resources,'descriptor references unknown resource')
                current={'stage':'COMPUTE_SHADER','access':item['access']}
                uses[resource]=current if resource not in uses else {'stage':'COMPUTE_SHADER','access':_merge(uses[resource]['access'],current['access'])}
                descriptors.append(item)
            barriers=self._barriers(uses)
            if barriers:commands.append({'command':'barrier','barriers':barriers})
            commands.append({'command':'dispatch','slot':identity,'descriptor_set':self.descriptor_slots[identity],
                             'pipeline':step.get('shader',step.get('recipe')),'grid':step['grid'],'block':step['block'],
                             'descriptors':descriptors,'launch_index':step['launch_index']})
        require(explicit<=set(self.states),'compact barrier names were not used')
        self.initialized=True;self.in_flight=True
        return {'schema':1,'frame':self.frame,'reset':bool(reset or graph.get('reset')),
                'allocation_count':self.allocation_count,'allocations_during_record':0,
                'uniform_writes':uniform_writes,'commands':commands,
                'motion_history':{'read':self._physical('image:8'),'write':self._physical('image:9')},
                'submits':0,'readbacks':0,'execution_qualified':False}

    def complete(self):
        require(self.in_flight,'no dispatch in flight')
        self.in_flight=False;self.frame+=1

    def manifest(self):
        return {'schema':1,'shape':self.shape,'resources':self.resources,'allocation_count':self.allocation_count,
                'uniform_bytes':self.uniform_bytes,'pipelines':sorted(self.pipelines),
                'descriptor_sets':len(self.descriptor_slots),'recording_contract':{
                    'application_owns':['device','queue','command-buffer','GAME_COLOR','GAME_DEPTH','GAME_MV','OUTPUT'],
                    'context_owns':['weights','pipelines','descriptors','uniforms','intermediates','history'],
                    'requires_previous_dispatch_complete':True,'records_only':True}}
