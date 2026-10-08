"""Independent byte expectations for the fused-stage copy-group model."""
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from ptx_reference import CTA,ptxsim

memory=ptxsim.Memory()
memory.add(4096,np.arange(64,dtype=np.uint8))
cta=CTA(SimpleNamespace(smem=128,name='copy_test'),memory,b'',(0,0,0),(2,1,1))
lanes=np.array([0,1])
cta.setr('%dst',lanes,[0,16])
cta.setr('%src',lanes,[4096,999999],64)
cta.setr('%valid',lanes,[5,0])
cta.op_cp('cp.async.cg.shared.global',['[%dst]','[%src]','16','%valid'],lanes)
assert not cta.smem.any(), 'copy must not be visible before the selected completion point'
cta.op_cp('cp.async.commit_group',[],lanes)
cta.setr('%dst',lanes,[32,48])
cta.setr('%src',lanes,[4112,4128],64)
cta.setr('%valid',lanes,[16,16])
cta.op_cp('cp.async.cg.shared.global',['[%dst]','[%src]','16','%valid'],lanes)
cta.op_cp('cp.async.commit_group',[],lanes)
cta.op_cp('cp.async.wait_group',['1'],lanes)
assert bytes(cta.smem[:16])==bytes(range(5))+bytes(11)
assert not cta.smem[16:64].any(), 'newest group must remain pending'
cta.op_cp('cp.async.wait_group',['0'],np.array([0]))
assert bytes(cta.smem[32:48])==bytes(range(16,32))
assert not cta.smem[48:64].any(), 'waits are per thread'
cta.op_cp('cp.async.wait_group',['0'],np.array([1]))
assert bytes(cta.smem[48:64])==bytes(range(32,48))
cta.setr('%valid',lanes,[17,0])
try:
    cta.op_cp('cp.async.cg.shared.global',['[%dst]','[%src]','16','%valid'],lanes)
    raise AssertionError('invalid size accepted')
except ValueError:
    pass
print('Partial copy, zero fill, unmapped zero-byte source and per-thread group completion passed')
