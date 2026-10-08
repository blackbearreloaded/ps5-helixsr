"""CPU-only PTX interpreter for the DLSS 1x1-conv kernels (K9 and family: K4/K10/K7/K11), one CTA at a time.

All threads of a CTA run as one vector (numpy, lane = thread). Scheduling is min-PC: every step executes the lowest
pending instruction for all threads at that PC, which reconverges the forward bounds-check branches of these kernels and
keeps warps together at bar.sync and at the warp-collective ops (ldmatrix, mma, shfl), which assert whole warps.

Numerics follow the original kernels on gfx1013:
  f16x2 add/sub/fma: IEEE fp16, round to nearest even (computed in float64 then rounded; exact for these operand sizes
  except astronomically rare double-rounding cases, which the capture comparison would expose).
  mma.m16n8k16 f16: per output, C to fp32, then 8 k-pairs in the lane-rotated order (t+k)&3 of the installed helpers,
  each pair two fp32 FMAs; 'direct' sites do the last FMA in double and round once to fp16, 'two' sites round fp32->fp16.
  mma.m16n8k8 f16: same with 4 k-pairs (k = (t+i)&3).
A 'math' hook lets callers replace the MMA chains by a candidate formulation (see model.py).
Usage as a library: Kernel(ptx_path) ; run_cta(kernel, mem, args, ctaid, block, hooks)."""
import os as _o, sys as _s; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))  # helixsr-setup
import re
import numpy as np

PARAM_BASE = 1 << 60
F16 = np.float16


def h2f(u16):
    return u16.astype(np.uint16).view(F16).astype(np.float64)


def f2h(x):
    return np.asarray(x, dtype=np.float64).astype(F16).view(np.uint16).astype(np.uint32)


def lo(w):
    return (w & 0xffff).astype(np.uint32)


def hi(w):
    return ((w >> 16) & 0xffff).astype(np.uint32)


def pack(l, h):
    return (l.astype(np.uint32) | (h.astype(np.uint32) << 16)).astype(np.uint32)


def fma32(a, b, c):
    """fp32 fma of exact fp16 products a*b (float64) and fp32 c (float64 holding an fp32 value)."""
    return (a * b + c).astype(np.float32).astype(np.float64)


class Memory:
    def __init__(self):
        self.allocs = []   # (base, np.uint8 array)

    def add(self, base, data):
        self.allocs.append((base, data))

    def _loc(self, addr, n):
        addr = addr.astype(np.int64)
        ai = np.full(addr.shape, -1)
        off = np.zeros(addr.shape, dtype=np.int64)
        for i, (b, d) in enumerate(self.allocs):
            m = (addr >= b) & (addr + n <= b + len(d))
            ai[m] = i
            off[m] = addr[m] - b
        assert (ai >= 0).all(), ('unmapped global access', [hex(int(a)) for a in addr[ai < 0][:4]])
        return ai, off

    def read(self, addr, n):
        addr = addr.astype(np.uint64)
        ai, off = self._loc(addr, n)
        out = np.zeros((len(addr), n), dtype=np.uint8)
        for i, (b, d) in enumerate(self.allocs):
            m = ai == i
            if m.any():
                out[m] = d[off[m][:, None] + np.arange(n)]
        return out

    def write(self, addr, data):
        addr = addr.astype(np.uint64)
        n = data.shape[1]
        ai, off = self._loc(addr, n)
        for i, (b, d) in enumerate(self.allocs):
            m = ai == i
            if m.any():
                d[off[m][:, None] + np.arange(n)] = data[m]


def statements(text):
    body = re.sub(r'//[^\n]*', '', text)
    start = body.index('{', body.index('.entry')) + 1
    end = body.rindex('}')
    out = []
    for raw in body[start:end].split(';'):
        s = ' '.join(raw.replace('{', ' { ').replace('}', ' } ').split()) if not re.search(r'\{\s*%', raw) else ' '.join(raw.split())
        # scope braces are not vector braces: drop lone '{' / '}' tokens at the start
        while True:
            s = s.strip()
            if s.startswith('{ ') or s == '{' or s.startswith('} ') or s == '}':
                s = s[1:]
                continue
            m = re.match(r'^(\$[\w]+):\s*(.*)$', s)
            if m:
                out.append(('label', m.group(1)))
                s = m.group(2)
                continue
            break
        if not s or s.startswith(('.reg', '.shared', '.pragma', '.local')):
            continue
        out.append(('ins', s))
    return out


class Kernel:
    def __init__(self, ptx_path):
        text = open(ptx_path).read()
        self.name = re.search(r'\.entry\s+(\w+)', text).group(1)
        m = re.search(r'\.shared\s+\.align\s+\d+\s+\.b8\s+smem_\[(\d+)\]', text)
        self.smem = int(m.group(1)) if m else 0
        self.ins = []
        self.labels = {}
        for kind, s in statements(text):
            if kind == 'label':
                self.labels[s] = len(self.ins)
            else:
                self.ins.append(self.parse(s))
        self.mma_sites = [i for i, x in enumerate(self.ins) if x['op'].startswith('mma.')]

    @staticmethod
    def parse(s):
        pred = None
        m = re.match(r'^@(!?)(%p\d+)\s+(.*)$', s)
        if m:
            pred = (m.group(1) == '!', m.group(2))
            s = m.group(3)
        op, _, rest = s.partition(' ')
        ops = []
        depth = 0
        cur = ''
        for ch in rest:
            if ch in '{[':
                depth += 1
            if ch in '}]':
                depth -= 1
            if ch == ',' and depth == 0:
                ops.append(cur.strip())
                cur = ''
            else:
                cur += ch
        if cur.strip():
            ops.append(cur.strip())
        return dict(op=op, ops=ops, pred=pred, text=s)


class CTA:
    def __init__(self, kernel, mem, args, ctaid, block, two_round=(), hooks=None, smem_size=None):
        self.k = kernel
        self.mem = mem
        self.args = np.frombuffer(args, dtype=np.uint8).copy()
        self.n = block[0] * block[1] * block[2]
        self.block = block
        self.ctaid = ctaid
        self.r = {}
        self.smem = np.zeros(smem_size or max(kernel.smem, 1), dtype=np.uint8)
        self.two_round = set(two_round)
        self.hooks = hooks or {}
        t = np.arange(self.n)
        self.tid = (t % block[0], (t // block[0]) % block[1], t // (block[0] * block[1]))
        self.mma_count = 0

    # ---- operand access -------------------------------------------------------------------------
    def val(self, s, idx, width=32):
        s = s.strip()
        if s.startswith('%'):
            if s in self.r:
                return self.r[s][idx]
            sp = {'%tid.x': self.tid[0], '%tid.y': self.tid[1], '%tid.z': self.tid[2]}
            if s in sp:
                return sp[s][idx].astype(np.uint64)
            sc = {'%ctaid.x': self.ctaid[0], '%ctaid.y': self.ctaid[1], '%ctaid.z': self.ctaid[2],
                  '%ntid.x': self.block[0], '%ntid.y': self.block[1], '%ntid.z': self.block[2]}
            if s in sc:
                return np.full(len(idx), sc[s], dtype=np.uint64)
            raise KeyError(s)
        if s == 'smem_':
            return np.zeros(len(idx), dtype=np.uint64)
        if s == self.k.name + '_param_0':
            return np.full(len(idx), PARAM_BASE, dtype=np.uint64)
        if re.match(r'^-?0[xX][0-9a-fA-F]+$', s) or re.match(r'^-?\d+$', s):
            v = int(s, 0)
            return np.full(len(idx), v & ((1 << 64) - 1), dtype=np.uint64)
        if re.match(r'^0[fF][0-9a-fA-F]{8}$', s):
            return np.full(len(idx), int(s[2:], 16), dtype=np.uint64)
        raise KeyError(s)

    def setr(self, name, idx, v, bits=32):
        if name not in self.r:
            dt = bool if name.startswith('%p') else np.uint64
            self.r[name] = np.zeros(self.n, dtype=dt)
        if name.startswith('%p'):
            self.r[name][idx] = v
        else:
            v = np.asarray(v).astype(np.uint64)
            if bits == 32:
                v = v & 0xffffffff
            self.r[name][idx] = v

    def addr(self, s, idx):
        s = s.strip()[1:-1]
        m = re.match(r'^(.+?)\s*\+\s*(-?\d+)$', s)
        base, off = (m.group(1), int(m.group(2))) if m else (s, 0)
        return (self.val(base, idx).astype(np.int64) + off).astype(np.uint64)

    # ---- memory spaces --------------------------------------------------------------------------
    def load(self, space, a, n):
        if space == 'param':
            off = (a - np.uint64(PARAM_BASE)).astype(np.int64)
            return self.args[off[:, None] + np.arange(n)]
        if space == 'shared':
            a = a.astype(np.int64)
            assert (a >= 0).all() and (a + n <= len(self.smem)).all(), 'shared OOB'
            return self.smem[a[:, None] + np.arange(n)]
        return self.mem.read(a, n)

    def store(self, space, a, data):
        if space == 'shared':
            a = a.astype(np.int64)
            assert (a >= 0).all() and (a + data.shape[1] <= len(self.smem)).all(), 'shared OOB'
            self.smem[a[:, None] + np.arange(data.shape[1])] = data
        else:
            self.mem.write(a, data)

    # ---- run ------------------------------------------------------------------------------------
    def run(self):
        k = self.k
        END = len(k.ins)
        pc = np.zeros(self.n, dtype=np.int64)
        steps = 0
        while True:
            act = pc < END
            if not act.any():
                break
            p = pc[act].min()
            sel = np.nonzero(pc == p)[0]
            ins = k.ins[p]
            idx = sel
            lh = self.hooks.get('labels', {}).get(p)
            if lh is not None:
                lh(self, sel)
            if ins['pred'] is not None:
                neg, pr = ins['pred']
                pv = self.r[pr][sel]
                if neg:
                    pv = ~pv
                take = sel[pv]
                skip = sel[~pv]
            else:
                take, skip = sel, sel[:0]
            nxt = self.exec(p, ins, take)
            pc[skip] = p + 1
            if isinstance(nxt, str):
                if nxt == 'ret':
                    pc[take] = END
                else:
                    pc[take] = k.labels[nxt]
            else:
                pc[take] = p + 1
            steps += 1
        return steps

    def whole_warps(self, idx):
        assert len(idx) % 32 == 0 and (idx.reshape(-1, 32) == (idx.reshape(-1, 32)[:, :1] + np.arange(32))).all() and \
            (idx.reshape(-1, 32)[:, 0] % 32 == 0).all(), 'collective op without whole warps'

    def exec(self, p, ins, idx):
        op = ins['op']
        o = ins['ops']
        if len(idx) == 0:
            return None if not op.startswith(('bra', 'ret')) else ('ret' if op == 'ret' else o[0])
        base = op.split('.')[0]
        if base in ('bra',):
            return o[0]
        if base in ('ret', 'exit'):
            return 'ret'
        if base == 'bar':
            return None
        if op.startswith('mma.'):
            self.mma(p, ins, idx)
            return None
        if op.startswith('ldmatrix'):
            self.ldmatrix(ins, idx)
            return None
        if op.startswith('shfl.sync.idx'):
            self.shfl(ins, idx)
            return None
        fn = getattr(self, 'op_' + base, None)
        if fn is None:
            raise NotImplementedError(ins['text'])
        fn(op, o, idx)
        return None

    # ---- integer / predicate ops ----------------------------------------------------------------
    @staticmethod
    def typ(op):
        for t in ('s64', 'u64', 'b64', 's32', 'u32', 'b32', 's16', 'u16', 'b16', 'pred', 'f16x2'):
            if op.endswith('.' + t) or ('.' + t + '.') in op + '.':
                return t
        return 'b32'

    def sval(self, s, idx, t):
        v = self.val(s, idx).astype(np.uint64)
        if t in ('s32',):
            return (v & 0xffffffff).astype(np.uint32).view(np.int32).astype(np.int64)
        if t in ('u32', 'b32'):
            return (v & 0xffffffff).astype(np.int64)
        if t == 's64':
            return v.view(np.int64)
        return v.astype(np.uint64)

    def out(self, d, idx, v, t):
        bits = 64 if t in ('s64', 'u64', 'b64') else 32
        if bits == 64:
            v = np.asarray(v).astype(np.int64).view(np.uint64) if np.asarray(v).dtype == np.int64 else np.asarray(v).astype(np.uint64)
        else:
            v = np.asarray(v).astype(np.int64) & 0xffffffff
        self.setr(d, idx, v, bits)

    def op_mov(self, op, o, idx):
        t = self.typ(op)
        if t == 'pred':
            self.setr(o[0], idx, self.r[o[1]][idx] if o[1] in self.r else bool(int(o[1])))
            return
        self.out(o[0], idx, self.val(o[1], idx).astype(np.uint64), t if t in ('b64', 'u64', 's64') else 'b32')

    def binop(self, op, o, idx, f):
        t = self.typ(op)
        a = self.sval(o[1], idx, t)
        b = self.sval(o[2], idx, t)
        self.out(o[0], idx, f(a, b), t)

    def op_add(self, op, o, idx):
        if op.endswith('f16x2'):
            return self.f16op(o, idx, lambda a, b: a + b)
        self.binop(op, o, idx, lambda a, b: a + b)

    def op_sub(self, op, o, idx):
        if op.endswith('f16x2'):
            return self.f16op(o, idx, lambda a, b: a - b)
        self.binop(op, o, idx, lambda a, b: a - b)

    def op_and(self, op, o, idx):
        if op.endswith('.pred'):
            return self.setr(o[0], idx, self.r[o[1]][idx] & self.r[o[2]][idx])
        self.binop(op, o, idx, lambda a, b: a & b)

    def op_or(self, op, o, idx):
        if op.endswith('.pred'):
            return self.setr(o[0], idx, self.r[o[1]][idx] | self.r[o[2]][idx])
        self.binop(op, o, idx, lambda a, b: a | b)

    def op_xor(self, op, o, idx):
        if op.endswith('.pred'):
            return self.setr(o[0], idx, self.r[o[1]][idx] ^ self.r[o[2]][idx])
        self.binop(op, o, idx, lambda a, b: a ^ b)

    def op_not(self, op, o, idx):
        if op.endswith('.pred'):
            return self.setr(o[0], idx, ~self.r[o[1]][idx])
        t = self.typ(op)
        self.out(o[0], idx, ~self.sval(o[1], idx, t), t)

    def op_min(self, op, o, idx):
        self.binop(op, o, idx, np.minimum)

    def op_max(self, op, o, idx):
        self.binop(op, o, idx, np.maximum)

    def op_shl(self, op, o, idx):
        t = self.typ(op)
        a = self.sval(o[1], idx, t).astype(np.uint64)
        b = self.sval(o[2], idx, 'u32')
        bits = 64 if t.endswith('64') else 32
        r = np.where(b >= bits, 0, a << np.minimum(b, bits - 1).astype(np.uint64))
        self.out(o[0], idx, r.astype(np.uint64), t if bits == 64 else 'b32')

    def op_shr(self, op, o, idx):
        t = self.typ(op)
        a = self.sval(o[1], idx, t)
        b = self.sval(o[2], idx, 'u32')
        bits = 64 if t.endswith('64') else 32
        if t.startswith('s'):
            r = a >> np.minimum(b, bits - 1)
        else:
            r = np.where(b >= bits, 0, a.astype(np.uint64) >> np.minimum(b, bits - 1).astype(np.uint64))
        self.out(o[0], idx, r, t)

    def op_mul(self, op, o, idx):
        if '.wide.' in op:
            t = 's32' if '.s32' in op else 'u32'
            a = self.sval(o[1], idx, t)
            b = self.sval(o[2], idx, t)
            return self.out(o[0], idx, a * b, 's64')
        assert '.lo.' in op, op
        self.binop(op, o, idx, lambda a, b: a * b)

    def op_mad(self, op, o, idx):
        assert '.lo.' in op, op
        t = self.typ(op)
        a, b, c = (self.sval(x, idx, t) for x in o[1:4])
        self.out(o[0], idx, a * b + c, t)

    def op_setp(self, op, o, idx):
        parts = op.split('.')
        cmp, t = parts[1], parts[2]
        a = self.sval(o[1], idx, t)
        b = self.sval(o[2], idx, t)
        f = dict(lt=np.less, le=np.less_equal, gt=np.greater, ge=np.greater_equal, eq=np.equal, ne=np.not_equal)[cmp]
        self.setr(o[0], idx, f(a, b))

    def op_selp(self, op, o, idx):
        t = self.typ(op)
        a = self.val(o[1], idx)
        b = self.val(o[2], idx)
        self.out(o[0], idx, np.where(self.r[o[3]][idx], a, b).astype(np.uint64), t)

    def op_cvt(self, op, o, idx):
        if op == 'cvt.s64.s32':
            return self.out(o[0], idx, self.sval(o[1], idx, 's32'), 's64')
        if op == 'cvt.u32.u64':
            return self.out(o[0], idx, self.sval(o[1], idx, 'u64') & 0xffffffff, 'b32')
        if op == 'cvt.u64.u32':
            return self.out(o[0], idx, self.sval(o[1], idx, 'u32'), 'u64')
        raise NotImplementedError(op)

    def op_cvta(self, op, o, idx):
        self.out(o[0], idx, self.val(o[1], idx), 'u64')

    def op_prmt(self, op, o, idx):
        assert op == 'prmt.b32', op
        a = self.sval(o[1], idx, 'u32')
        b = self.sval(o[2], idx, 'u32')
        c = self.sval(o[3], idx, 'u32')
        bytes_ = np.stack([(a >> (8 * i)) & 0xff for i in range(4)] + [(b >> (8 * i)) & 0xff for i in range(4)], 1)
        r = np.zeros(len(idx), dtype=np.int64)
        for i in range(4):
            sel = (c >> (4 * i)) & 0xf
            v = bytes_[np.arange(len(idx)), sel & 7]
            sign = ((v >> 7) & 1) * 0xff
            v = np.where(sel & 8, sign, v)
            r |= v << (8 * i)
        self.out(o[0], idx, r, 'b32')

    def op_bfi(self, op, o, idx):
        a = self.sval(o[1], idx, 'u32')
        b = self.sval(o[2], idx, 'u32')
        pos = self.sval(o[3], idx, 'u32') & 0xff
        ln = self.sval(o[4], idx, 'u32') & 0xff
        mask = np.where(ln >= 32, 0xffffffff, ((1 << np.minimum(ln, 31)) - 1)) << pos
        mask &= 0xffffffff
        r = (b & ~mask) | ((a << pos) & mask)
        self.out(o[0], idx, r, 'b32')

    # ---- loads / stores -------------------------------------------------------------------------
    def op_ld(self, op, o, idx):
        parts = op.split('.')
        space = parts[1]
        vec = 1
        if parts[2].startswith('v'):
            vec = int(parts[2][1:])
            t = parts[3]
        else:
            t = parts[2]
        sz = int(re.sub(r'\D', '', t)) // 8
        a = self.addr(o[1], idx)
        data = self.load(space, a, vec * sz)
        regs = re.findall(r'%[\w]+', o[0]) if vec > 1 else [o[0]]
        for i, rg in enumerate(regs):
            chunk = data[:, i * sz:(i + 1) * sz]
            v = np.zeros(len(idx), dtype=np.uint64)
            for j in range(sz):
                v |= chunk[:, j].astype(np.uint64) << np.uint64(8 * j)
            self.setr(rg, idx, v, 64 if sz == 8 else 32)

    def op_st(self, op, o, idx):
        parts = op.split('.')
        space = parts[1]
        vec = 1
        if parts[2].startswith('v'):
            vec = int(parts[2][1:])
            t = parts[3]
        else:
            t = parts[2]
        sz = int(re.sub(r'\D', '', t)) // 8
        a = self.addr(o[0], idx)
        regs = re.findall(r'%[\w]+|\b\d+\b', o[1]) if vec > 1 else [o[1]]
        data = np.zeros((len(idx), vec * sz), dtype=np.uint8)
        for i, rg in enumerate(regs):
            v = self.val(rg, idx).astype(np.uint64)
            for j in range(sz):
                data[:, i * sz + j] = ((v >> np.uint64(8 * j)) & np.uint64(0xff)).astype(np.uint8)
        self.store(space, a, data)

    # ---- fp16 -----------------------------------------------------------------------------------
    def f16op(self, o, idx, f):
        a = self.sval(o[1], idx, 'u32')
        b = self.sval(o[2], idx, 'u32')
        r = pack(f2h(f(h2f(lo(a)), h2f(lo(b)))), f2h(f(h2f(hi(a)), h2f(hi(b)))))
        self.out(o[0], idx, r, 'b32')

    def op_fma(self, op, o, idx):
        assert op.startswith('fma.rn.f16x2'), op
        relu = op.endswith('.relu')
        a, b, c = (self.sval(x, idx, 'u32') for x in o[1:4])
        def one(x, y, z):
            r = f2h(h2f(x) * h2f(y) + h2f(z))
            if relu:
                neg = (r & 0x8000) != 0
                nan = ((r & 0x7c00) == 0x7c00) & ((r & 0x3ff) != 0)
                r = np.where(nan, 0x7fff, np.where(neg, 0, r))
            return r
        self.out(o[0], idx, pack(one(lo(a), lo(b), lo(c)), one(hi(a), hi(b), hi(c))), 'b32')

    # ---- warp collectives -----------------------------------------------------------------------
    def shfl(self, ins, idx):
        self.whole_warps(idx)
        o = ins['ops']
        d = o[0]
        dp = None
        if '|' in d:
            d, dp = d.split('|')
        a = self.sval(o[1], idx, 'u32')
        b = self.sval(o[2], idx, 'u32')
        c = self.sval(o[3], idx, 'u32')
        lane = idx % 32
        w = idx - lane
        maxl = c & 0x1f
        segmask = (c >> 8) & 0x1f
        minl = lane & segmask
        maxl = minl | (maxl & ~segmask)
        j = minl | (b & ~segmask & 0x1f)
        ok = j <= maxl
        j = np.where(ok, j, lane)
        full = np.zeros(self.n, dtype=np.int64)
        full[idx] = a
        self.out(d, idx, full[w + j], 'b32')
        if dp:
            self.setr(dp, idx, ok)

    def ldmatrix(self, ins, idx):
        self.whole_warps(idx)
        op = ins['op']
        o = ins['ops']
        x = int(re.search(r'\.x(\d)', op).group(1))
        assert '.trans' not in op
        regs = re.findall(r'%\w+', o[0])
        a = self.addr(o[1], idx).astype(np.int64)
        full = np.zeros(self.n, dtype=np.int64)
        full[idx] = a
        lane = idx % 32
        w = idx - lane
        for j in range(x):
            rowaddr = full[w + 8 * j + lane // 4] + (lane % 4) * 4
            data = self.smem[rowaddr[:, None] + np.arange(4)]
            v = data[:, 0].astype(np.uint64) | (data[:, 1].astype(np.uint64) << 8) | (data[:, 2].astype(np.uint64) << 16) | (data[:, 3].astype(np.uint64) << 24)
            self.setr(regs[j], idx, v)

    def mma(self, p, ins, idx):
        self.whole_warps(idx)
        site = self.k.mma_sites.index(p)
        o = ins['ops']
        D, A, B, C = (re.findall(r'%\w+', x) for x in o)
        hook = self.hooks.get('mma')
        if hook is not None:
            res = hook(self, site, idx, D, A, B, C)
            if res is not None:
                return
        k16 = 'k16' in ins['op']
        two = site in self.two_round
        self.mma_count += 1
        av = [self.sval(r, idx, 'u32') for r in A]
        bv = [self.sval(r, idx, 'u32') for r in B]
        cv = [self.sval(r, idx, 'u32') for r in C]
        lane = idx % 32
        w = idx - lane
        g = lane >> 2
        t = lane & 3
        def gath(vals, src):
            full = np.zeros(self.n, dtype=np.int64)
            out = []
            for v in vals:
                full[idx] = v
                out.append(full[w + src])
            return out
        res = [None, None]
        npair = 8 if k16 else 4
        for r in range(2):
            halves = []
            for oo in range(2):
                d = h2f((cv[r] >> (16 * oo)) & 0xffff)
                for k in range(npair):
                    tp = (t + k) & 3
                    srca = (g << 2) | tp
                    srcb = ((2 * t + oo) << 2) | tp
                    if k16:
                        areg = (k >> 2) * 2 + r
                        breg = k >> 2
                    else:
                        areg = r
                        breg = 0
                    aw = gath([av[areg]], srca)[0]
                    bw = gath([bv[breg]], srcb)[0]
                    d = fma32(h2f(lo(aw)), h2f(lo(bw)), d)
                    if k == npair - 1 and not two:
                        d = h2f(f2h(h2f(hi(aw)) * h2f(hi(bw)) + d))
                    else:
                        d = fma32(h2f(hi(aw)), h2f(hi(bw)), d)
                halves.append(f2h(d))
            res[r] = pack(halves[0], halves[1])
        for r in range(2):
            self.out(D[r], idx, res[r], 'b32')


def load_capture(cdir, extra_size=None):
    """Memory + args from a replay capture directory (manifest.txt, alloc-*.bin, args.bin)."""
    from pathlib import Path
    cdir = Path(cdir)
    mem = Memory()
    launch = None
    for line in (cdir / 'manifest.txt').read_text().splitlines():
        f = line.split()
        if f[0] == 'alloc':
            mem.add(int(f[2], 16), np.fromfile(cdir / f'alloc-{f[1]}.bin', dtype=np.uint8).copy())
        if f[0] == 'launch':
            launch = tuple(int(x) for x in f[1:8])
    return mem, (cdir / 'args.bin').read_bytes(), launch
