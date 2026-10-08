"""Deterministic analytical frame inputs; no renderer, model asset or GPU required."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

SCENES = ('constant', 'impulse', 'checkerboard', 'slanted_edge', 'depth_step', 'motion')
JITTER = ((0.0, 0.0), (0.25, -0.25), (-0.25, 0.25), (0.0, 0.0))

def foreground(x, y, frame):
    return 8 + 2*frame <= x < 16 + 2*frame and 6 <= y < 18

def sample(scene, x, y, frame):
    """World-space scene color, reversed device depth and surface class."""
    if scene == 'constant':
        return (0.25, 0.5, 1.0, 1.0), 0.25, False
    if scene == 'impulse':
        v = float(12 <= x < 13 and 10 <= y < 11)
        return (v, v, v, 1.0), 0.25, False
    if scene == 'checkerboard':
        v = float((int(x)//2 + int(y)//2) % 2)
        return (v, v, v, 1.0), 0.25, False
    if scene == 'slanted_edge':
        v = float(x >= 0.625*y + 8)
        return (v, v, v, 1.0), 0.25, False
    if scene == 'depth_step':
        front = x < 16
        return ((1.0,0.25,0.0,1.0) if front else (0.0,0.25,1.0,1.0)), (0.75 if front else 0.25), front
    if scene != 'motion':
        raise ValueError('unknown scene')
    front = foreground(x,y,frame)
    if front:
        return (1.0,0.125,0.0,1.0), 0.75, True
    v = 0.125 if (int(x)//2 + int(y)//2) % 2 else 0.5
    return (v,0.25,v,1.0), 0.25, False

def frame_data(scene, width, height, output_width, output_height, frame):
    if not (1 <= width <= output_width <= 4096 and 1 <= height <= output_height <= 4096):
        raise ValueError('unsupported dimensions')
    if not 0 <= frame < len(JITTER):
        raise ValueError('fixture frame outside fixed sequence')
    camera = frame if scene == 'motion' else 0
    jx,jy = JITTER[frame]
    color, depth, motion, revealed = bytearray(), bytearray(), bytearray(), bytearray()
    for y in range(height):
        for x in range(width):
            world_x, world_y = x + 0.5 + jx + camera, y + 0.5 + jy
            rgba,z,front = sample(scene,world_x,world_y,frame)
            # Motion excludes projection jitter; current position -> previous position.
            dx = (1.0 - (2.0 if front else 0.0)) if scene == 'motion' and frame else 0.0
            color += struct.pack('<4e',*rgba)
            depth += struct.pack('<f',z)
            motion += struct.pack('<2e',dx,0.0)
            # Current background was occluded by foreground in the previous frame.
            reveal = scene == 'motion' and frame > 0 and not front and foreground(world_x,world_y,frame-1)
            revealed.append(int(reveal))
    target = bytearray()
    for y in range(output_height):
        for x in range(output_width):
            rgba,_,_ = sample(scene,(x+0.5)*width/output_width+camera,(y+0.5)*height/output_height,frame)
            target += struct.pack('<4e',*rgba)
    return {'color.rgba16f': bytes(color), 'depth.r32f': bytes(depth), 'motion.rg16f': bytes(motion),
            'disocclusion.r8u': bytes(revealed), 'native_target.rgba16f': bytes(target)}

def generate(destination, width=32, height=24, output_width=48, output_height=36):
    destination = Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    manifest = {'schema':1, 'render':[width,height], 'output':[output_width,output_height],
        'contract': {'color':'linear scene-referred RGBA, alpha=1', 'depth':'reversed device depth in [0,1]',
            'motion':'current-to-previous render pixels; excludes jitter; top-left origin; +Y down',
            'pre_exposure':1.0, 'exposure':1.0, 'auto_exposure':False,
            'history':'reset frame zero; later frames retain history',
            'native_target':'unjittered point-sampled scene truth; NOT a neural-network golden output'}, 'frames':[]}
    for scene in SCENES:
        for frame in range(len(JITTER)):
            entry = {'scene':scene, 'frame':frame, 'reset':frame==0, 'jitter':JITTER[frame],
                     'previous_jitter':JITTER[max(0,frame-1)], 'files':{}}
            for role, data in frame_data(scene,width,height,output_width,output_height,frame).items():
                name = f'{scene}/{frame:02d}/{role}'
                path = destination/name
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(data)
                entry['files'][role] = {'path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
            manifest['frames'].append(entry)
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    return manifest

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--render',type=int,nargs=2,default=(32,24))
    parser.add_argument('--display',type=int,nargs=2,default=(48,36))
    args = parser.parse_args()
    result = generate(args.output,*args.render,*args.display)
    print(f'Wrote {len(result["frames"])} deterministic frame inputs to {args.output}')
