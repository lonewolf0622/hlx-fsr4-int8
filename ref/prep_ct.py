# Copyright 2026 lonewolf0622
# SPDX-License-Identifier: GPL-2.0-or-later
"""prep_ct.py SHADER C O W H OUT.pkl: FNB_CT2D_ADD pass: block (P, Q fitted vs the block outputs the shader feeds its
transposed conv) + transposed conv (Wt[4][O][C], bt) + output exponents per 16-ch group (fitted vs final stores)"""
import os as _os
ROOT = _os.environ.get('HLX_ROOT') or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys, struct, itertools, pickle, numpy as np
from dxil_interp import Shader, Thread
from convnext import blob, i8
import fasternet as fn, ct2d
k, C, O, W, H, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
OW, OH = 2 * W, 2 * H
sh = Shader(open(ROOT + '/work/dis/%04d.dxil.txt' % k).read())
scratch = bytearray(np.random.default_rng(0).integers(-60, 60, 160 << 20, dtype=np.int8).tobytes()); cb = bytearray(4096)
for r in range(32): struct.pack_into('<II', cb, 16 * r, OW, OH)
bufs = {r: (blob() if r[0] == 't' else scratch) for r in set(sh.res.values()) if r[0] in 'tu'}
cbs = {r: bytes(cb) for r in set(sh.res.values()) if r[0] == 'b'}
def th(x, y):
    t = Thread(sh, (x, y, 0), (x // 64, y, 0), (x % 64, 0, 0), bufs, cbs); t.run(); return t
t0 = th(30, 30); P = fn.extract(t0, 30, 30, C, W, H); base = P['in_base']
row, plane = (W + 2) * 16, (W + 2) * 16 * (H + 2)
raw = np.frombuffer(bytes(scratch[base:base + plane * C // 16]), np.int8).reshape(C // 16, H + 2, W + 2, 16)
xpad = np.concatenate(list(raw), axis=2)
pts = [(30, 30), (0, 0), (W - 1, H - 1), (W // 2, 1), (2, H // 2), (W - 3, 7)]
T = {p: th(*p) for p in pts}
Q = fn.fit(P, xpad, [(p, ct2d.extract_ct(T[p], C, O)['fnb']) for p in pts]); assert Q, 'block fit failed'
E = ct2d.extract_ct(T[pts[0]], C, O)
orow, oplane = (OW + 2) * 16, (OW + 2) * 16 * (OH + 2)
samples = []
for p in pts:
    t = T[p]; e = ct2d.extract_ct(t, C, O); used = set(x[3][2] for x in t.trace if x[0] == 'dot4' and x[3])
    sk = {o: v for h, o, v in t.loads if h[0] == 'u' and o not in used}; st = {o: v for h, o, v in t.stores}
    skb, stb = min(sk), min(st); skip = np.zeros((4, O), np.int64); res = np.zeros((4, O), np.int64)
    for pos, (dx, dy) in enumerate([(0, 0), (1, 0), (0, 1), (1, 1)]):
        for pl in range(O // 16):
            for q in range(4):
                o_ = pl * oplane + dy * orow + dx * 16 + 4 * q
                skip[pos, 16 * pl + 4 * q:16 * pl + 4 * q + 4] = i8(sk[skb + o_][0] if isinstance(sk[skb + o_], tuple) else sk[skb + o_])
                res[pos, 16 * pl + 4 * q:16 * pl + 4 * q + 4] = i8(st[stb + o_])
    samples.append((e['bt'] + np.einsum('poc,c->po', e['Wt'], e['fnb']), skip, res))
cq = []
for g in range(O // 16):
    sl = slice(16 * g, 16 * g + 16)
    cq.append(next((a, r, o) for a, r, o in itertools.product(range(4, 20), range(0, 10), range(0, 10))
                   if all(np.array_equal(ct2d.ct_out(acc[:, sl], sk[:, sl], a, r, o), rr[:, sl]) for acc, sk, rr in samples)))
pickle.dump(dict(P=P, Q=Q, Wt=E['Wt'], bt=E['bt'], cq=cq, C=C, O=O, W=W, H=H, k=k), open(out, 'wb')); print('saved', out, Q, cq)
