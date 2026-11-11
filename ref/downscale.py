# Copyright 2026 lonewolf0622
# SPDX-License-Identifier: GPL-2.0-or-later
"""downscale.py: FSR 4.1.2 2x2 stride-2 conv C -> 2C (pass3 16->32, pass6 32->64), plane-of-16 layout in and out.
  out = clamp8(round_ne(acc * 2^-a)), acc = b + sum W[o, ky, kx, c] * x[2y+ky, 2x+kx, c]  (fitted exponent per 16-ch group)"""
import struct, itertools, numpy as np
from dxil_interp import Thread
from convnext import i8, clamp8

def run(sh, x, y, bufs, cbs):
    t = Thread(sh, (x, y, 0), (x // 64, y, 0), (x % 64, 0, 0), bufs, cbs); t.run(); return t

def extract(th, C, O, IW, IH):
    row, plane = (IW + 2) * 16, (IW + 2) * 16 * (IH + 2)
    d = [t for t in th.trace if t[0] == 'dot4']; init = {t[1]: t[2] for t in th.trace if t[0] == 'init'}
    order = []
    for t in d:
        if t[1] not in order: order.append(t[1])
    by = {c: [t for t in d if t[1] == c] for c in order}
    assert len(order) == O and all(len(by[c]) == C for c in order), (len(order), len(by[order[0]]))
    base = min(t[3][2] for t in d if t[3])
    Wd = np.zeros((O, 2, 2, C), np.int64)
    for o, c in enumerate(order):
        for t in by[c]:
            rel = t[3][2] - base; pl, r = divmod(rel, plane); ky, r = divmod(r, row); kx, ch = divmod(r, 16)
            Wd[o, ky, kx, 16 * pl + ch:16 * pl + ch + 4] = i8(t[4])
    return dict(Wd=Wd, b=np.array([init[c] for c in order]), base=base)

def outputs(th, O, OW, OH):
    st = {o: v for h, o, v in th.stores}; b0 = min(st); plane = (OW + 2) * 16 * (OH + 2)
    return np.concatenate([np.concatenate([i8(st[b0 + p * plane + 4 * q]) for q in range(4)]) for p in range(O // 16)])

def inputs(th, C, IW, IH):
    """the 2x2 input patch (int8 [2, 2, C]) this thread read"""
    row, plane = (IW + 2) * 16, (IW + 2) * 16 * (IH + 2)
    d = [t for t in th.trace if t[0] == 'dot4']; base = min(t[3][2] for t in d if t[3])
    x = np.zeros((2, 2, C), np.int64)
    for t in d:
        rel = t[3][2] - base; pl, r = divmod(rel, plane); ky, r = divmod(r, row); kx, ch = divmod(r, 16)
        x[ky, kx, 16 * pl + ch:16 * pl + ch + 4] = i8(t[2])
    return x

def out(P, patch, a):
    acc = P['b'] + np.einsum('oyxc,yxc->o', P['Wd'], patch)
    return clamp8(np.round(acc.astype(np.float32) * np.float32(2.0 ** -a)).astype(np.int64))
