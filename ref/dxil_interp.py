# Copyright 2026 lonewolf0622
# SPDX-License-Identifier: GPL-2.0-or-later
"""dxil_interp.py: run ONE thread of a DXIL compute shader (text from `dxc -dumpbin`) on the CPU.

Covers the instruction subset of the FSR 4 network passes. Buffers are bytearrays keyed by binding register
(('t', 21) SRV, ('u', 11) UAV); cbuffers as bytes. Every value loaded from a buffer carries a tag
(space, register, byte offset) that follows it through extractvalue / alloca stores and loads / phi / select /
bitcast, so dot4AddPacked records exactly which bytes it multiplied (trace).
"""
import re, struct, math
import numpy as np

M32 = 0xFFFFFFFF
def s32(x): x &= M32; return x - (1 << 32) if x & 0x80000000 else x
def f32(x): return float(np.float32(x))
def f2b(x): return struct.unpack('<I', struct.pack('<f', np.float32(x)))[0]
def b2f(x): return struct.unpack('<f', struct.pack('<I', x & M32))[0]

class V:
    """a value: python int (i1/i8/i16/i32/i64, kept masked to the type) or float; tag = origin or None"""
    __slots__ = ('v', 'tag')
    def __init__(self, v, tag=None): self.v, self.tag = v, tag

TOK = re.compile(r'\s*(%[\w.]+|@[\w.]+|-?0x[0-9A-Fa-f]+|-?[\d.]+(?:e[+-]?\d+)?|undef|true|false|zeroinitializer)')

class Shader:
    def __init__(self, text):
        fn = text[text.index('\ndefine void @'):]
        fn = fn[:fn.index('\n}\n')]
        self.blocks = {}; cur = '0'; self.order = []
        self.blocks[cur] = []
        for line in fn.split('\n')[2:]:
            m = re.match(r'; <label>:(\d+)', line) or re.match(r'^(\d+):', line)
            if m:
                cur = m.group(1); self.blocks[cur] = []; continue
            if '  ;' in line: line = line[:line.index('  ;')]
            line = line.strip()
            if line.startswith(';'): continue
            if line: self.blocks[cur].append(line)
        self.res = {}   # handle name -> (space, reg)
        for line in fn.split('\n'):
            m = re.search(r'(%\d+) = call %dx.types.Handle @dx.op.createHandleFromBinding\(i32 217, %dx.types.ResBind \{ i32 (\d+), i32 \d+, i32 \d+, i8 (\d+) \}', line)
            if m: self.res[m.group(1)] = ({0: 't', 1: 'u', 2: 'b', 3: 's'}[int(m.group(3))], int(m.group(2)))
            m = re.search(r'(%\d+) = call %dx.types.Handle @dx.op.createHandleFromBinding\(i32 217, %dx.types.ResBind zeroinitializer', line)
            if m: self.res[m.group(1)] = ('t', 0)   # (t0, space 0: printed as zeroinitializer)

class Thread:
    def __init__(self, sh, tid, gid, gtid, buffers, cbuf, trace=None):
        self.sh, self.tid, self.gid, self.gtid = sh, tid, gid, gtid
        self.buf, self.cb = buffers, cbuf
        self.env = {}; self.mem = {}; self.trace = trace if trace is not None else []
        self.stores = []; self.loads = []

    def val(self, tok, ty=None):
        if tok.startswith('%'): return self.env[tok]
        if tok in ('undef', 'zeroinitializer', 'false'): return V(0)
        if tok == 'true': return V(1)
        if ty and ty in ('float', 'half', 'double', 'f32', 'f16', 'f64'):
            if tok.startswith('0xH'): return V(float(np.uint16(int(tok[3:], 16)).view(np.float16)))   # (half constant)
            if tok.startswith('0x'):
                return V(struct.unpack('>d', bytes.fromhex(tok[2:].rjust(16, '0')))[0])
            return V(float(tok))
        return V(int(tok, 0))

    def run(self):
        blk, prev = '0', None
        steps = 0
        while True:
            for ins in self.sh.blocks[blk]:
                steps += 1
                r = self.step(ins, prev, blk)
                if r is not None:
                    if r == 'ret': return steps
                    prev, blk = blk, r; break
            else:
                raise RuntimeError('fell off block ' + blk)

    def step(self, ins, prev, blk):
        if ins.startswith('br '):
            m = re.match(r'br i1 (\S+), label %(\d+), label %(\d+)', ins)
            if m: return m.group(2) if self.val(m.group(1)).v & 1 else m.group(3)
            return re.match(r'br label %(\d+)', ins).group(1)
        if ins.startswith('ret'): return 'ret'
        if ins.startswith('switch'):
            m = re.match(r'switch i32 (\S+), label %(\d+) \[(.*)\]', ins)
            x = self.val(m.group(1)).v & M32
            for c, l in re.findall(r'i32 (-?\d+), label %(\d+)', m.group(3)):
                if int(c) & M32 == x: return l
            return m.group(2)
        if ins.startswith('store '):
            m = re.match(r'store (\w+) (\S+), \w+\* (%[\w.]+)', ins)
            self.mem[self.env[m.group(3)].v] = self.val(m.group(2), m.group(1)); return None
        if ins.startswith('call void @llvm.lifetime') or ins.startswith('call void @dx.op.barrier'): return None
        if ins.startswith('call void @dx.op.textureStore'):
            a = self.args(ins); h = self.hres(a[1]); co = tuple(self.val(t).v & M32 for t in a[2:4])
            vals = tuple(self.val(t, 'f32').v for t in a[5:9])
            if not hasattr(self, 'texstores'): self.texstores = []
            self.texstores.append((h, co, vals)); return None
        if ins.startswith('call void @dx.op.rawBufferStore'):
            a = self.args(ins)
            h = self.hres(a[1]); off = self.val(a[2]).v & M32
            mask = int(a[8]) if len(a) > 8 else 15
            vals = [self.val(a[4 + k]).v for k in range(4)]
            for k in range(4):
                if mask >> k & 1:
                    self.stores.append((h, off + 4 * k, vals[k] & M32))
                    if h in self.buf and off + 4 * k + 4 <= len(self.buf[h]): struct.pack_into('<I', self.buf[h], off + 4 * k, vals[k] & M32)
            return None
        m = re.match(r'(%[\w.]+) = (.*)', ins)
        if not m: raise RuntimeError('cannot run: ' + ins)
        dst, rhs = m.groups()
        self.env[dst] = self.expr(rhs, prev)
        return None

    def args(self, ins):
        inner = ins[ins.index('(') + 1:ins.rindex(')')]
        out, depth, cur = [], 0, ''
        for ch in inner:
            if ch in '{(': depth += 1
            if ch in '})': depth -= 1
            if ch == ',' and depth == 0: out.append(cur.strip()); cur = ''
            else: cur += ch
        out.append(cur.strip())
        return [a.split()[-1] if a else a for a in out]

    def expr(self, rhs, prev):
        op = rhs.split()[0]
        if op == 'phi':
            ty = rhs.split()[1]
            for v, l in re.findall(r'\[ (\S+), %(\d+) \]', rhs):
                if l == prev: return self.val(v, ty)
            raise RuntimeError('phi without ' + str(prev))
        if op == 'alloca':
            self.nalloca = getattr(self, 'nalloca', 0) + 1
            return V(('a', self.nalloca))
        if op == 'getelementptr':
            p = re.findall(r'(%[\w.]+), i32 0, i32 (\S+)', rhs)[0]
            base = self.env[p[0]].v; idx = self.val(p[1]).v
            return V((base, s32(idx)))
        if op == 'load':
            p = re.match(r'load \w+, \w+\* (%[\w.]+)', rhs).group(1)
            return self.mem.get(self.env[p].v, V(0))
        if op == 'extractvalue':
            m = re.match(r'extractvalue %?[\w.]+ (%[\w.]+), (\d+)', rhs)
            s = self.env[m.group(1)]; k = int(m.group(2))
            v = s.v[k]; tag = None
            if s.tag: tag = (s.tag[0], s.tag[1], s.tag[2] + 4 * k)
            return V(v, tag)
        if op == 'select':
            m = re.match(r'select i1 (\S+), (\w+) (\S+), \w+ (\S+)', rhs)
            return self.val(m.group(3), m.group(2)) if self.val(m.group(1)).v & 1 else self.val(m.group(4), m.group(2))
        if op == 'bitcast':
            if '*' in rhs: return V(None)
            m = re.match(r'bitcast (\S+) (\S+) to (\S+)', rhs)
            v = self.val(m.group(2), m.group(1))
            if m.group(1) == 'float' and m.group(3) == 'i32': return V(f2b(v.v), v.tag)
            if m.group(1) == 'i32' and m.group(3) == 'float': return V(b2f(v.v), v.tag)
            return v
        if op in ('sext', 'zext', 'trunc', 'sitofp', 'uitofp', 'fptosi', 'fptoui', 'fpext', 'fptrunc'):
            m = re.match(r'\w+ (\w+) (\S+) to (\w+)', rhs)
            x = self.val(m.group(2), m.group(1)).v; fr, to = m.group(1), m.group(3)
            bits = lambda t: int(t[1:]) if t.startswith('i') else 0
            if op == 'sext': b = bits(fr); x &= (1 << b) - 1; x = x - (1 << b) if x >> (b - 1) & 1 else x; return V(x & ((1 << bits(to)) - 1))
            if op == 'zext': return V(x & ((1 << bits(fr)) - 1))
            if op == 'trunc': return V(x & ((1 << bits(to)) - 1))
            if op == 'sitofp': b = bits(fr); x &= (1 << b) - 1; x = x - (1 << b) if x >> (b - 1) & 1 else x; return V(f32(x) if to == 'float' else float(np.float16(x)))
            if op == 'uitofp': return V(f32(x & ((1 << bits(fr)) - 1)))
            if op in ('fptosi', 'fptoui'):
                if math.isnan(x): return V(0)
                return V(int(x) & ((1 << bits(to)) - 1))
            if op == 'fpext': return V(float(x))
            if op == 'fptrunc': return V(float(np.float16(x)) if to == 'half' else f32(x))
        if op == 'icmp':
            m = re.match(r'icmp (\w+) (\w+) (\S+), (\S+)', rhs)
            c, ty = m.group(1), m.group(2); b = int(ty[1:])
            a, bb = self.val(m.group(3)).v & ((1 << b) - 1), self.val(m.group(4)).v & ((1 << b) - 1)
            sa = a - (1 << b) if a >> (b - 1) & 1 else a; sb = bb - (1 << b) if bb >> (b - 1) & 1 else bb
            return V(int({'eq': a == bb, 'ne': a != bb, 'ugt': a > bb, 'uge': a >= bb, 'ult': a < bb, 'ule': a <= bb,
                          'sgt': sa > sb, 'sge': sa >= sb, 'slt': sa < sb, 'sle': sa <= sb}[c]))
        if op == 'fcmp':
            m = re.match(r'fcmp (?:fast )?(\w+) (\w+) (\S+), (\S+)', rhs)
            a, b = self.val(m.group(3), m.group(2)).v, self.val(m.group(4), m.group(2)).v; c = m.group(1)
            return V(int({'oeq': a == b, 'une': a != b, 'ogt': a > b, 'oge': a >= b, 'olt': a < b, 'ole': a <= b,
                          'one': a != b, 'ueq': a == b, 'ugt': not a <= b, 'ult': not a >= b, 'uge': not a < b, 'ule': not a > b}[c]))
        if op in ('add', 'sub', 'mul', 'shl', 'lshr', 'ashr', 'and', 'or', 'xor', 'sdiv', 'udiv', 'srem', 'urem'):
            m = re.match(r'(\w+)(?: nuw| nsw| exact)* (\w+) (\S+), (\S+)', rhs)
            ty = m.group(2); b = int(ty[1:]); mk = (1 << b) - 1
            a, c = self.val(m.group(3)).v & mk, self.val(m.group(4)).v & mk
            sa = a - (1 << b) if a >> (b - 1) & 1 else a; sc = c - (1 << b) if c >> (b - 1) & 1 else c
            r = {'add': lambda: a + c, 'sub': lambda: a - c, 'mul': lambda: a * c, 'shl': lambda: a << (c & (b - 1)),
                 'lshr': lambda: a >> (c & (b - 1)), 'ashr': lambda: sa >> (c & (b - 1)), 'and': lambda: a & c,
                 'or': lambda: a | c, 'xor': lambda: a ^ c, 'sdiv': lambda: int(sa / sc), 'udiv': lambda: a // c,
                 'srem': lambda: sa - int(sa / sc) * sc, 'urem': lambda: a % c}[op]()
            return V(r & mk)
        if op in ('fadd', 'fsub', 'fmul', 'fdiv', 'frem'):
            m = re.match(r'(\w+)(?: fast| nnan| ninf| nsz| arcp| contract| reassoc| afn)* (\w+) (\S+), (\S+)', rhs)
            ty = m.group(2); a, c = self.val(m.group(3), ty).v, self.val(m.group(4), ty).v
            r = {'fadd': a + c, 'fsub': a - c, 'fmul': a * c, 'fdiv': a / c if c else (math.copysign(math.inf, a) if a else math.nan)}[op]
            return V(float(np.float16(r)) if ty == 'half' else f32(r))
        if op == 'call':
            return self.call(rhs)
        raise RuntimeError('cannot run: ' + rhs)

    def call(self, rhs):
        fn = re.search(r'@dx\.op\.(\w+)(?:\.(\w+))?\(', rhs)
        if not fn: raise RuntimeError('call ' + rhs)
        name, ty = fn.groups(); a = self.args(rhs)
        if name in ('createHandleFromBinding',): return V(None)
        if name == 'annotateHandle':
            # remember which binding this handle refers to
            return V(('h', a[1]))
        if name == 'threadId': return V(self.tid[int(a[1])])
        if name == 'groupId': return V(self.gid[int(a[1])])
        if name == 'threadIdInGroup': return V(self.gtid[int(a[1])])
        if name == 'flattenedThreadIdInGroup': return V(self.gtid[0])
        if name == 'legacyF32ToF16': return V(int(np.float16(np.float32(self.val(a[1], 'f32').v)).view(np.uint16)))
        if name == 'legacyF16ToF32': return V(f32(np.uint16(self.val(a[1]).v & 0xFFFF).view(np.float16)))
        if name == 'rawBufferLoad':
            h = self.hres(a[1]); off = self.val(a[2]).v & M32
            b = self.buf[h]
            vals = tuple(struct.unpack_from('<I', b, off + 4 * k)[0] if off + 4 * k + 4 <= len(b) else 0 for k in range(4))
            if ty == 'f32': vals = tuple(b2f(x) for x in vals)
            self.loads.append((h, off, vals))
            return V(vals, (h[0], h[1], off))
        if name == 'cbufferLoadLegacy':
            h = self.hres(a[1]); row = self.val(a[2]).v & M32; cb = self.cb[h]
            vals = tuple(struct.unpack_from('<I', cb, 16 * row + 4 * k)[0] if 16 * row + 4 * k + 4 <= len(cb) else 0 for k in range(4))
            if ty == 'f32': vals = tuple(b2f(x) for x in vals)
            return V(vals, ('cb', h[1], 16 * row))
        if name == 'dot4AddPacked':
            acc, x, y = self.val(a[1]), self.val(a[2]), self.val(a[3])
            xb, yb = struct.pack('<I', x.v & M32), struct.pack('<I', y.v & M32)
            s = s32(acc.v) + sum(((xb[k] ^ 0x80) - 0x80) * ((yb[k] ^ 0x80) - 0x80) for k in range(4))
            if acc.tag and acc.tag[0] == 'acc': cid = acc.tag[1]
            else: self.nacc = getattr(self, 'nacc', 0) + 1; cid = self.nacc; self.trace.append(('init', cid, s32(acc.v), acc.tag))
            self.trace.append(('dot4', cid, x.v & M32, x.tag, y.v & M32, y.tag))
            return V(s & M32, ('acc', cid))
        if name == 'dot2AddHalf':   # acc + a.x * b.x + a.y * b.y: float16 products (rounded) added in float32, as the driver lowers it
            acc, ax, ay, bx, by = (self.val(t, 'f32').v for t in a[1:6])
            p0 = float(np.float16(np.float16(ax) * np.float16(bx))); p1 = float(np.float16(np.float16(ay) * np.float16(by)))
            return V(f32(f32(acc + p0) + p1))
        if name == 'pack4x8':
            mode = int(a[1]); xs = [s32(self.val(t).v) for t in a[2:6]]
            if mode == 2: xs = [max(-128, min(127, x)) for x in xs]
            elif mode == 1: xs = [max(0, min(255, x)) for x in xs]
            return V(sum((x & 0xFF) << (8 * k) for k, x in enumerate(xs)))
        if name == 'unpack4x8':
            mode = int(a[1]); x = self.val(a[2]); bs = struct.pack('<I', x.v & M32)
            vals = tuple((((bs[k] ^ 0x80) - 0x80) if mode == 1 else bs[k]) & M32 for k in range(4))
            return V(vals, x.tag)
        if name == 'binary':
            o = int(a[0]); x, y = self.val(a[1], ty).v, self.val(a[2], ty).v
            if ty == 'i32':
                sx, sy = s32(x), s32(y)
                return V({37: max(sx, sy), 38: min(sx, sy), 39: max(x & M32, y & M32), 40: min(x & M32, y & M32)}[o] & M32)
            return V(f32({35: max(x, y), 36: min(x, y)}[o]))
        if name == 'tertiary':
            o = int(a[0]); x, y, z = (self.val(t, ty).v for t in a[1:4])
            if ty == 'i32': return V((s32(x) * s32(y) + s32(z)) & M32 if o == 48 else ((x & M32) * (y & M32) + (z & M32)) & M32)
            return V(f32(x * y + z))
        if name == 'unary':
            o = int(a[0]); x = self.val(a[1], ty).v
            fn_ = {6: abs, 7: lambda v: min(1.0, max(0.0, v)), 21: lambda v: 2.0 ** v, 22: lambda v: v - math.floor(v),
                   23: lambda v: math.log2(v) if v > 0 else -math.inf, 24: math.sqrt, 25: lambda v: 1 / math.sqrt(v),
                   26: lambda v: float(np.round(np.float32(v))), 27: math.floor, 28: math.ceil, 29: math.trunc}[o]
            return V(f32(fn_(x)))
        if name in ('textureLoad', 'sampleLevel', 'sample', 'textureGather'):
            # textures: deterministic data from TEXFN (handle, coords) if set (else zeros); recorded in self.texloads
            h = self.hres(a[1]); cs = a[3:6]   # (textureLoad: mip level, then x y z; sample ops: the sampler handle, then the coordinates)
            co = [self.val(t, 'f32' if name != 'textureLoad' else 'i32').v if t not in ('undef',) else 0 for t in cs]
            fn = getattr(self, 'texfn', None)
            vals = fn(h, name, co) if fn else (0.0, 0.0, 0.0, 0.0)
            if not hasattr(self, 'texloads'): self.texloads = []
            self.texloads.append((h, name, tuple(co), vals))
            return V(tuple(vals) + (0,))
        if name == 'textureStore': return V(None)
        if name == 'quadOp': return self.val(a[1], ty)
        if name in ('waveReadLaneFirst', 'waveActiveOp'): return self.val(a[1], ty)
        if name == 'isSpecialFloat':
            o = int(a[0]); x = self.val(a[1], ty).v
            return V(int({8: math.isnan(x), 9: math.isinf(x), 10: math.isfinite(x)}[o]))
        raise RuntimeError('dx.op not supported: ' + name)

    def hres(self, tok):
        v = self.env[tok].v
        while isinstance(v, tuple) and v and v[0] == 'h': tok = v[1]; v = self.env[tok].v
        return self.sh.res[tok]
