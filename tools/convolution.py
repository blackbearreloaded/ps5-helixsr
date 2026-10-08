"""Compact scalar-tensor model of the three standalone 3x3/pool stages.

Arithmetic policy: channel groups of 16, column-major 3x3 taps, lane-rotated
pairs, fp32 partial sums, one half rounding per dot16, then half bias/ReLU.
This policy must be checked against the stage interpreter, not presumed from names.
"""
import numpy as np

def convolution_pool(values, weights, bias):
    # values [I/8,H,W,8]; weights [I/8,K,K,O,8]
    groups,height,width,_=values.shape
    outputs=len(bias)
    channels=groups*8
    kernel=weights.shape[1]
    assert kernel in (1,3) and weights.shape[2]==kernel
    assert channels%16==0 and outputs%8==0 and height%2==0 and width%2==0
    image=values.transpose(0,3,1,2).reshape(channels,height,width).astype(np.float64)
    pad=kernel//2
    padded=np.pad(image,((0,0),(pad,pad),(pad,pad)))
    weight=weights.transpose(3,0,4,1,2).reshape(outputs,channels,kernel,kernel).astype(np.float64)
    result=np.zeros((outputs,height,width),dtype=np.float64)
    rows=np.arange(outputs)
    lane=(rows%8)//2
    for base in range(0,channels,16):
        for kx in range(kernel):
            for ky in range(kernel):
                for pair in range(8):
                    channel=base+((lane+pair)%4)*2+(8 if pair>=4 else 0)
                    for half in range(2):
                        c=channel+half
                        result=padded[c,ky:ky+height,kx:kx+width]*weight[rows,c,ky,kx,None,None]+result
                        if pair==7 and half==1:
                            result=result.astype(np.float16).astype(np.float64)
                        else:
                            result=result.astype(np.float32).astype(np.float64)
    result=(result+bias.astype(np.float64)[:,None,None]).astype(np.float16)
    result=np.maximum(result,np.float16(0))
    pooled=result.reshape(outputs,height//2,2,width//2,2).max(axis=(2,4))
    pack=lambda t:t.reshape(outputs//8,8,height//2,width//2).transpose(0,2,3,1).copy()
    unpooled=np.stack([pack(result[:,dy::2,dx::2]) for dy,dx in ((0,0),(0,1),(1,0),(1,1))])
    return pack(pooled),unpooled
