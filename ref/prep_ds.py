# Copyright 2026 lonewolf0622
# SPDX-License-Identifier: GPL-2.0-or-later
"""prep_ds.py SHADER C IW IH OUT.pkl: extract + fit a 2x2 stride-2 downscale pass (C -> 2C)"""
import os as _os
ROOT = _os.environ.get('HLX_ROOT') or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys, struct, pickle, numpy as np
from dxil_interp import Shader
from convnext import blob
import downscale as ds
k, C, IW, IH, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
O, OW, OH = 2 * C, IW // 2, IH // 2
sh = Shader(open(ROOT + '/work/dis/%04d.dxil.txt' % k).read())
scratch = bytearray(np.random.default_rng(0).integers(-60, 60, 160 << 20, dtype=np.int8).tobytes()); cb = bytearray(4096)
for r in range(32): struct.pack_into('<II', cb, 16 * r, OW, OH)
bufs = {r: (blob() if r[0] == 't' else scratch) for r in set(sh.res.values()) if r[0] in 'tu'}
cbs = {r: bytes(cb) for r in set(sh.res.values()) if r[0] == 'b'}
P = ds.extract(ds.run(sh, 20, 20, bufs, cbs), C, O, IW, IH)
pts = [(20, 20), (0, 0), (OW - 1, OH - 1), (OW - 1, 0), (0, OH - 1), (OW // 3, OH // 2), (7, 11), (OW - 4, 5)]
S = [(ds.inputs(t, C, IW, IH), ds.outputs(t, O, OW, OH)) for t in (ds.run(sh, x, y, bufs, cbs) for x, y in pts)]
ex = [next(a for a in range(2, 20) if all(np.array_equal(ds.out(P, p, a)[16 * g:16 * g + 16], o[16 * g:16 * g + 16]) for p, o in S)) for g in range(O // 16)]
pickle.dump(dict(Wd=P['Wd'], b=P['b'], ex=ex, C=C, IW=IW, IH=IH), open(out, 'wb')); print('saved', out, ex)
