# Copyright 2026 lonewolf0622
# SPDX-License-Identifier: GPL-2.0-or-later
"""convnext.py: FSR 4.1.2 ConvNeXt pass (16 ch, e.g. pass1/2/12) - weights from an interpreter trace + exact numpy.

Math (int8, power-of-2 fixed point, every shift rounds half to even):
  c0  = clamp8(rne(b0 + conv3x3(x) , s0))
  r1  = clamp8(rne(max(b1 + W1 c0, 0), s1))
  out = clamp8(round_ne(acc2 * 2^-e2 + x_center * 2^-ex) * 2^eo),  acc2 = b2 + W2 r1
"""
import os as _os
ROOT = _os.environ.get('HLX_ROOT') or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import os, struct, numpy as np
from dxil_interp import Shader, Thread

import os as _os
BLOB_MAIN = int(_os.environ.get("BLOB", "0"), 0) or 0x1638460      # file offset of the main InitializerBuffer in the SDK 2.4.0 DLL
DLL = _os.environ.get('AMD_FSR_DLL') or ROOT + '/work/amd_fidelityfx_upscaler_dx12.dll'   # prepare.sh copies AMD's DLL there

def blob(start=BLOB_MAIN, n=128 << 10):
    return bytearray(open(DLL, 'rb').read()[start:start + n])

def i8(dw): return np.frombuffer(struct.pack('<I', dw & 0xFFFFFFFF), np.int8)

def rne_shift(x, s):
    x = np.asarray(x, np.int64)
    return (x + (1 << (s - 1)) - 1 + ((x >> s) & 1)) >> s

def clamp8(x): return np.clip(x, -128, 127)

def trace_thread(sh, x, y, bufs, cbs, tg=64):
    th = Thread(sh, (x, y, 0), (x // tg, y, 0), (x % tg, 0, 0), bufs, cbs)
    th.run(); return th

def extract(th, x, y, C=16, E=32, row=15392, px=16):
    """weights/biases of one ConvNeXt pass from the trace of the thread at output pixel (x, y)"""
    d = [t for t in th.trace if t[0] == 'dot4']
    init = {t[1]: t[2] for t in th.trace if t[0] == 'init'}
    order = []
    for t in d:
        if t[1] not in order: order.append(t[1])
    by = {c: [t for t in d if t[1] == c] for c in order}
    k3, k1, k2 = order[:C], order[C:C + E], order[C + E:C + E + C]
    acts = [t[3][2] for c in k3 for t in by[c] if t[3]]
    base = min(acts)                             # tap (0, 0), channel 0 of output pixel (x, y): padded (x, y)
    P_in = base - (y * row + x * px)
    W0 = np.zeros((C, 3, 3, C), np.int64)
    for f, c in enumerate(k3):
        for _, _, xv, xt, yv, yt in by[c]:
            rel = xt[2] - base; ky, kx, cg = rel // row, (rel % row) // px, (rel % px) // 4
            W0[f, ky, kx, 4 * cg:4 * cg + 4] = i8(yv)
    W1 = np.array([np.concatenate([i8(t[4]) for t in by[c]]) for c in k1], np.int64)
    W2 = np.array([np.concatenate([i8(t[4]) for t in by[c]]) for c in k2], np.int64)
    b0 = np.array([init[c] for c in k3]); b1 = np.array([init[c] for c in k1]); b2 = np.array([init[c] for c in k2])
    return dict(W0=W0, W1=W1, W2=W2, b0=b0, b1=b1, b2=b2, in_base=P_in)

def forward_pixel(P, xpad, x, y, s0=7, s1=8, e2=12, ex=6, eo=6):
    """xpad: int8 [H+2, W+2, C] (zero border); output pixel (x, y) -> int8[C]"""
    patch = xpad[y:y + 3, x:x + 3, :].astype(np.int64)
    a0 = P['b0'] + np.einsum('fyxc,yxc->f', P['W0'], patch)
    c0 = clamp8(rne_shift(a0, s0))
    a1 = P['b1'] + P['W1'] @ c0
    r1 = clamp8(rne_shift(np.maximum(a1, 0), s1))
    a2 = P['b2'] + P['W2'] @ r1
    res = xpad[y + 1, x + 1, :].astype(np.float32)
    v = (a2.astype(np.float32) * np.float32(2.0 ** -e2) + res * np.float32(2.0 ** -ex)) * np.float32(2.0 ** eo)
    return clamp8(np.round(v).astype(np.int64))

def params_from_ir(text):
    """shifts (s0 after the 3x3, s1 after expand) and float scales (acc, residual, out) of a ConvNeXt pass"""
    import re, collections
    defs = dict(re.findall(r'^\s*(%\d+) = (.*)$', text, re.M))
    sh = [int(n) for n in re.findall(r'ashr i32 %\d+, (\d+)', text)]
    s0, s1 = sh[0], sh[16]
    def cval(c):
        if c.startswith('0x'): return struct.unpack('>d', bytes.fromhex(c[2:]))[0]
        return float(c)
    kinds = {}
    for dst, a, c in re.findall(r'(%\d+) = fmul fast float (%\d+), ([-0-9.eExA-F+]+)', text):
        src = defs.get(a, '')
        if src.startswith('fadd'): kinds['out'] = cval(c)
        elif src.startswith('sitofp'):
            inner = defs.get(src.split()[2], '')
            kinds['res' if 'extractvalue %dx.types.fouri32' in inner else 'acc'] = cval(c)
    import math
    return dict(s0=s0, s1=s1, e2=-int(math.log2(kinds['acc'])), ex=-int(math.log2(kinds['res'])), eo=int(math.log2(kinds['out'])))

def forward_full(P, xpad, s0, s1, e2, ex, eo):
    """whole image: xpad int8 [H+2, W+2, C] -> int8 [H, W, C] (exact, vectorised)"""
    H, W = xpad.shape[0] - 2, xpad.shape[1] - 2; x = xpad.astype(np.int64)
    a0 = np.broadcast_to(P['b0'], (H, W, len(P['b0']))).copy()
    for ky in range(3):
        for kx in range(3):
            a0 += x[ky:ky + H, kx:kx + W, :] @ P['W0'][:, ky, kx, :].T
    c0 = clamp8(rne_shift(a0, s0))
    r1 = clamp8(rne_shift(np.maximum(c0 @ P['W1'].T + P['b1'], 0), s1))
    a2 = r1 @ P['W2'].T + P['b2']
    v = (a2.astype(np.float32) * np.float32(2.0 ** -e2) + x[1:-1, 1:-1].astype(np.float32) * np.float32(2.0 ** -ex)) * np.float32(2.0 ** eo)
    return clamp8(np.round(v).astype(np.int64)).astype(np.int8)
