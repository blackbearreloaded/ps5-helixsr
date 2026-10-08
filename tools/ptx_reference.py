"""Host-only extensions needed by the missing main convolution stages.

The pinned upstream interpreter defines a lane-rotated arithmetic policy; it is
not an NVIDIA execution oracle. Extracted PTX stays in an external asset directory.
"""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'third_party/helixsr'))
import ptxsim

class CTA(ptxsim.CTA):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.copy_pending=[[] for _ in range(self.n)]
        self.copy_groups=[[] for _ in range(self.n)]

    def op_cp(self,op,operands,idx):
        if op=='cp.async.cg.shared.global':
            destination=self.addr(operands[0],idx)
            source=self.addr(operands[1],idx)
            size=int(operands[2])
            if size!=16:
                raise NotImplementedError('only the fused stage 16-byte copy is supported')
            valid=self.val(operands[3],idx) if len(operands)==4 else np.full(len(idx),size)
            for lane,dst,src,n in zip(idx,destination,source,valid):
                n=int(n)
                if not 0<=n<=size:
                    raise ValueError('invalid asynchronous copy source length')
                data=np.zeros(size,dtype=np.uint8)
                if n:
                    data[:n]=self.mem.read(np.array([src],dtype=np.uint64),n)[0]
                if int(dst)+size>len(self.smem):
                    raise ValueError('asynchronous shared destination outside allocation')
                self.copy_pending[int(lane)].append((int(dst),data))
            return
        if op=='cp.async.commit_group':
            for lane in idx:
                self.copy_groups[int(lane)].append(self.copy_pending[int(lane)])
                self.copy_pending[int(lane)]=[]
            return
        if op=='cp.async.wait_group':
            remaining=int(operands[0])
            if not 0<=remaining<=7:
                raise ValueError('unsupported asynchronous wait count')
            for lane in idx:
                groups=self.copy_groups[int(lane)]
                while len(groups)>remaining:
                    for dst,data in groups.pop(0):
                        self.smem[dst:dst+len(data)]=data
            return
        raise NotImplementedError(op)

    def out(self,d,idx,value,kind):
        if kind in ('b16','u16','s16'):
            value = np.asarray(value).astype(np.uint64)&65535
        return super().out(d,idx,value,kind)

    def op_cvt(self,op,operands,idx):
        if op in ('cvt.u16.u32','cvt.u32.u16'):
            return self.out(operands[0],idx,self.val(operands[1],idx)&65535,'u32')
        return super().op_cvt(op,operands,idx)

    def op_mul(self,op,operands,idx):
        if op in ('mul.hi.s32','mul.hi.u32'):
            kind=op.rsplit('.',1)[1]
            a,b=(self.sval(v,idx,kind) for v in operands[1:])
            # Signed products fit int64; unsigned products require uint64.
            if kind=='u32':
                a,b=a.astype(np.uint64),b.astype(np.uint64)
            return self.out(operands[0],idx,(a*b)>>32,'u32')
        return super().op_mul(op,operands,idx)

    def op_shl(self,op,operands,idx):
        if op=='shl.b16':
            a,b=(self.val(v,idx) for v in operands[1:])
            return self.out(operands[0],idx,np.where(b>=16,0,(a&65535)<<np.minimum(b,15)),'u16')
        return super().op_shl(op,operands,idx)

    def op_dp2a(self,op,operands,idx):
        if op!='dp2a.lo.u32.u32':
            raise NotImplementedError(op)
        a,b,c=(self.val(v,idx).astype(np.uint64) for v in operands[1:])
        return self.out(operands[0],idx,(a&65535)*(b&255)+((a>>16)&65535)*((b>>8)&255)+c,'u32')

    def op_max(self,op,operands,idx):
        if op=='max.f16x2':
            # np.fmax implements the PTX max-number treatment for one NaN operand.
            return self.f16op(operands,idx,np.fmax)
        return super().op_max(op,operands,idx)

    def exec(self,pc,instruction,idx):
        if instruction['op']=='bar.sync' and len(idx)!=self.n:
            raise RuntimeError('reference cannot qualify divergent CTA barrier')
        return super().exec(pc,instruction,idx)
