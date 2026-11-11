# Copyright 2026 lonewolf0622
# SPDX-License-Identifier: GPL-2.0-or-later
"""offsets.py FAMILY: AMD's scratch byte offsets (input / output / skip tensor bases, padded layout) of every network pass of one
layout family (data/families.json: std1080, std4k, up1080, up4k) -> work/passes/<family's offsets file>"""
import os as _os
ROOT = _os.environ.get('HLX_ROOT') or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import struct, json, sys, numpy as np
from dxil_interp import Shader, Thread
from convnext import blob
import convnext as cn, fasternet as fn
FAM = json.load(open(ROOT + '/data/families.json'))[sys.argv[1]]
big = max(v['W'] for v in FAM['passes'].values()) > 1000
scratch = bytearray(np.random.default_rng(0).integers(-60, 60, (400 if big else 160) << 20, dtype=np.int8).tobytes())
res = {}
for name, v in FAM['passes'].items():
    k, kind, W, H, C = v['shader'], v['kind'], v['W'], v['H'], v['C']
    sh = Shader(open(ROOT + '/work/dis/%04d.dxil.txt' % k).read())
    OW, OH = (2 * W, 2 * H) if kind == 'ct' else (W, H)
    cb = bytearray(4096)
    for r in range(32): struct.pack_into('<II', cb, 16 * r, OW, OH)
    bufs = {r: (blob() if r[0] == 't' else scratch) for r in set(sh.res.values()) if r[0] in 'tu'}
    cbs = {r: bytes(cb) for r in set(sh.res.values()) if r[0] == 'b'}
    x, y = 30, 30
    th = Thread(sh, (x, y, 0), (x // 64, y, 0), (x % 64, 0, 0), bufs, cbs); th.run()
    st = min(o for h, o, v_ in th.stores)
    if kind == 'cnx':
        P = cn.extract(th, x, y, row=(W + 2) * 16); ib = P['in_base']; ob = st - ((y + 1) * (W + 2) * 16 + (x + 1) * 16); sb = None
    elif kind == 'fnb':
        P = fn.extract(th, x, y, C, W, H); ib = P['in_base']; ob = st - ((y + 1) * (W + 2) * 16 + (x + 1) * 16); sb = None
    elif kind == 'ds':
        IW, IH = 2 * W, 2 * H; base = min(t[3][2] for t in th.trace if t[0] == 'dot4' and t[3])
        ib = base - ((2 * y + 1) * (IW + 2) * 16 + (2 * x + 1) * 16); ob = st - ((y + 1) * (W + 2) * 16 + (x + 1) * 16); sb = None
    else:
        P = fn.extract(th, x, y, C, W, H); ib = P['in_base']
        used = set(t[3][2] for t in th.trace if t[0] == 'dot4' and t[3])
        sk = min(o for h, o, v_ in th.loads if h[0] == 'u' and o not in used)
        orow = (OW + 2) * 16
        sb = sk - ((2 * y + 1) * orow + (2 * x + 1) * 16); ob = st - ((2 * y + 1) * orow + (2 * x + 1) * 16)
    res[name] = dict(shader=k, kind=kind, W=W, H=H, C=C, inBase=ib, outBase=ob, skipBase=sb, **({'wshader': v['wshader']} if 'wshader' in v else {}))
    print(name, kind, 'in', ib, 'out', ob, 'skip', sb)
_os.makedirs(ROOT + '/work/passes', exist_ok=True)
json.dump(res, open(ROOT + '/work/passes/' + FAM['offsets'], 'w'), indent=1)
