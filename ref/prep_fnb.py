# Copyright 2026 lonewolf0622
# SPDX-License-Identifier: GPL-2.0-or-later
"""prep_fnb.py SHADER C W H OUT.pkl: extract + fit a FasterNet pass once (weights, biases, shifts), save for generators"""
import os as _os
ROOT = _os.environ.get('HLX_ROOT') or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys, struct, pickle, numpy as np
from dxil_interp import Shader
from convnext import blob, trace_thread
import fasternet as fn
k, C, W, H, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
sh = Shader(open(ROOT + '/work/dis/%04d.dxil.txt' % k).read())
rng = np.random.default_rng(2)
xpad = np.zeros((H + 2, W + 2, C), np.int8); xpad[1:-1, 1:-1] = rng.integers(-60, 60, (H, W, C))
scratch = bytearray(160 << 20); cb = bytearray(4096)
for r in range(32): struct.pack_into('<II', cb, 16 * r, W, H)
res = set(sh.res.values())
bufs = {r: (blob() if r[0] == 't' else scratch) for r in res if r[0] in 'tu'}; cbs = {r: bytes(cb) for r in res if r[0] == 'b'}
base = fn.extract(trace_thread(sh, 60, 60, bufs, cbs), 60, 60, C, W, H)['in_base']
raw = fn.to_planes(xpad); scratch[base:base + len(raw)] = raw
P = fn.extract(trace_thread(sh, 60, 60, bufs, cbs), 60, 60, C, W, H)
def amd(x, y):
    st = sorted(trace_thread(sh, x, y, bufs, cbs).stores, key=lambda s: s[1])
    return np.frombuffer(b''.join(struct.pack('<I', v) for _, _, v in st), np.int8)[:C].astype(np.int64)
pts = [(60, 60), (1, 1), (W - 1, H - 1), (200 % W, 100 % H), (0, H // 2), (W // 2, 0)]
Q = fn.fit(P, xpad, [(p, amd(*p)) for p in pts]); assert Q, 'fit failed'
pickle.dump(dict(P=P, Q=Q, C=C, W=W, H=H, k=k), open(out, 'wb')); print('saved', out, Q)
