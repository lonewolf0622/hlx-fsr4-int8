# Copyright 2026 lonewolf0622
# SPDX-License-Identifier: GPL-2.0-or-later
"""post.py: FSR 4.1.2 postpass (#411) - extraction of the final ConvNeXt block + 2x2 transposed conv (16 -> 8) from
interpreter traces, and an exact-ish numpy reference of the reconstruction (port of AMD's post_common.hlsli)."""
import os as _os
ROOT = _os.environ.get('HLX_ROOT') or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import struct, numpy as np
from dxil_interp import Shader, Thread
from convnext import blob, extract, i8, rne_shift, clamp8

import os
SH = int(os.environ.get("POSTSH", "411"))
def make(cbvals=None, scratch=None):
    sh = Shader(open(ROOT + '/work/dis/%04d.dxil.txt' % SH).read())
    if scratch is None: scratch = bytearray(np.random.default_rng(0).integers(-60, 60, 40 << 20, dtype=np.int8).tobytes())
    cb = bytearray(8192)
    c = cbvals or dict(inv_size=(1 / 1920, 1 / 1080), scale=(1.5, 1.5), inv_scale=(1 / 1.5, 1 / 1.5), jitter=(0.25, -0.125),
                       width=1920, height=1080, reset=0, width_lr=1280, height_lr=720, pre=1.0, prev=1.0, rcas=0)
    struct.pack_into('<8f', cb, 0, *c['inv_size'], *c['scale'], *c['inv_scale'], *c['jitter'])
    struct.pack_into('<4I', cb, 64, c['width'], c['height'], c['reset'], c['width_lr'])
    struct.pack_into('<IffI', cb, 80, c['height_lr'], c['pre'], c['prev'], c['rcas'])
    bufs = {r: (blob() if r[0] == 't' else scratch) for r in set(sh.res.values()) if r[0] in 'tu'}
    cbs = {r: bytes(cb) for r in set(sh.res.values()) if r[0] == 'b'}
    return sh, bufs, cbs, c

def texfn(h, name, co):
    k = [int(v) if isinstance(v, int) else int(v * 997) for v in co]
    return tuple(0.1 + 0.37 * ((h[1] * 7 + k[0] * 13 + k[1] * 17 + c * 29) % 23) / 23 for c in range(4))

def run(sh, bufs, cbs, lx, ly):
    """thread for low-res pixel (lx, ly): 16x16 groups of 256 threads, row-major inside the group (as traced)"""
    gid = (lx // 16, ly // 16, 0); l = (ly % 16) * 16 + (lx % 16)
    th = Thread(sh, (lx, ly, 0), gid, (l, 0, 0), bufs, cbs); th.texfn = texfn; th.run(); return th

def pixel_of(th):
    acts = [t[3][2] for t in th.trace if t[0] == 'dot4' and t[3] and t[3][0] == 'u']
    b = min(acts); return (b % 15392) // 16, b // 15392        # tap (0,0) padded = output pixel (unpadded)

def chains(th):
    d = [t for t in th.trace if t[0] == 'dot4']; order = []
    for t in d:
        if t[1] not in order: order.append(t[1])
    by = {c: [t for t in d if t[1] == c] for c in order}
    init = {t[1]: (t[2], t[3]) for t in th.trace if t[0] == 'init'}
    return order, by, init

def extract_post(th):
    x, y = pixel_of(th)
    P = extract(th, x, y)                                       # ConvNeXt block (first 16 + 32 + 16 chains)
    order, by, init = chains(th)
    ct = order[64:96]
    # each CT chain: 4 dot4 over the 16 block outputs (packed int8); weight operand = the one not equal to block output
    acts = [t[2] for t in by[ct[0]]]
    Wc = []
    for c in ct:
        xs = [t[2] for t in by[c]]; ys = [t[4] for t in by[c]]
        w = ys if xs == acts else xs
        Wc.append(np.concatenate([i8(v) for v in w]))
    fnb = np.concatenate([i8(v) for v in acts])
    return dict(P=P, x=x, y=y, Wct=np.array(Wc, np.int64), bct=np.array([init[c][0] for c in ct]), fnb=fnb, ctorder=ct)

def mulaw(x): return np.maximum(0.1174 * np.log(1.0 + 150.0 * x), 0)
def unmulaw(x): return np.maximum((np.exp(8.51788 * x) - 1.0) / 150.0, 0)

def recon(params, X, Y, c, color, reproj, expo):
    """params: 4 floats (model filter); color(x, y) low-res rgb, reproj(X, Y) hi-res rgb; returns (model colour, output)"""
    corr = np.tanh(params[0]); sfx = 2 / (1 + np.exp(-params[1])); sfy = 2 / (1 + np.exp(-params[2]))
    inv, sc, jit = np.array(c['inv_scale']), np.array(c['scale']), np.array(c['jitter'])
    lr = (-0.5 + inv / 2 + np.array([X, Y]) * inv) + jit
    li = np.round(lr).astype(int)       # HLSL round: half away from zero? (checked against the shader)
    up = np.zeros(3); tw = 0.0
    for dy in range(3):
        yi = li[1] + dy - 1; yd = (yi - lr[1]) * sc[1]
        for dx in range(3):
            xi = li[0] + dx - 1; xd = (xi - lr[0]) * sc[0]
            g = np.exp((-0.5 / (0.47 * 0.47)) * (xd * xd * sfx * sfx + 2 * corr * xd * yd * sfx * sfy + yd * yd * sfy * sfy))
            tw += g
            cx, cy = min(max(xi, 0), c['width_lr'] - 1), min(max(yi, 0), c['height_lr'] - 1)
            up += mulaw(np.array(color(int(cx), int(cy))) * expo) * g
    up /= tw
    b = 1 / (1 + np.exp(-params[3]))
    model = up * (1 - b) + np.array(reproj(int(X), int(Y))) * b
    out = np.clip(unmulaw(model) / expo, 0, 64000)
    return model, out
