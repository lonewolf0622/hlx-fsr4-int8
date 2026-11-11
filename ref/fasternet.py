# Copyright 2026 lonewolf0622
# SPDX-License-Identifier: GPL-2.0-or-later
"""fasternet.py: FSR 4.1.2 FasterNet pass (C = 32 or 64: pass4/5/10, pass7/8) - weights from a trace + exact numpy.

  h   = C/2 channels; 3x3 grouped conv (16 inputs per output) on the first half only
  c0  = clamp8(rne(b0 + conv3x3(x[:h]), s0)) ++ x[h:]            (C channels)
  r1  = clamp8(rne(max(b1 + W1 c0, 0), s1))                      (2C channels)
  out[:h] = clamp8(round_ne((acc2 * 2^-a + x[:h]  * 2^-ra) * 2^oa))
  out[h:] = clamp8(round_ne((acc2 * 2^-b + c0[h:] * 2^-rb) * 2^ob)),  acc2 = b2 + W2 r1
Shifts/exponents are fitted against the AMD shader (interpreter); they are powers of two, so a fit is exact or fails.
"""
import itertools, numpy as np
from convnext import i8, rne_shift, clamp8

def extract(th, x, y, C, W, H):
    """input/output tensors are stored as planes of 16 channels: offset = plane*(H+2)*(W+2)*16 + py*(W+2)*16 + px*16 + c"""
    h = C // 2; row = (W + 2) * 16; plane = row * (H + 2)
    d = [t for t in th.trace if t[0] == 'dot4']
    init = {t[1]: t[2] for t in th.trace if t[0] == 'init'}
    order = []
    for t in d:
        if t[1] not in order: order.append(t[1])
    by = {c: [t for t in d if t[1] == c] for c in order}
    k3, k1, k2 = order[:h], order[h:h + 2 * C], order[h + 2 * C:h + 3 * C]
    assert all(len(by[c]) == 36 for c in k3) and all(len(by[c]) == C // 4 for c in k1) and all(len(by[c]) == C // 2 for c in k2), \
        [len(by[c]) for c in order[:3]]
    acts = [t[3][2] for c in k3 for t in by[c] if t[3]]
    base = min(acts)                                   # plane 0, tap (0, 0) of output pixel (x, y)
    W0 = np.zeros((h, 3, 3, C), np.int64)
    for f, c in enumerate(k3):
        for _, _, xv, xt, yv, yt in by[c]:
            rel = xt[2] - base; pl, r2 = divmod(rel, plane); ky, r3 = divmod(r2, row); kx, ch = divmod(r3, 16)
            W0[f, ky, kx, 16 * pl + ch:16 * pl + ch + 4] = i8(yv)
    W1 = np.array([np.concatenate([i8(t[4]) for t in by[c]]) for c in k1], np.int64)
    W2 = np.array([np.concatenate([i8(t[4]) for t in by[c]]) for c in k2], np.int64)
    b = lambda ks: np.array([init[c] for c in ks])
    return dict(C=C, W0=W0, W1=W1, W2=W2, b0=b(k3), b1=b(k1), b2=b(k2), in_base=base - (y * row + x * 16))

def to_planes(xpad):
    """[H+2, W+2, C] int8 -> bytes in the plane-of-16 layout"""
    return b''.join(np.ascontiguousarray(xpad[:, :, p:p + 16]).tobytes() for p in range(0, xpad.shape[2], 16))

def stage(P, xpad, x, y, s0, s1):
    C, h = P['C'], P['C'] // 2
    patch = xpad[y:y + 3, x:x + 3, :].astype(np.int64)
    a0 = P['b0'] + np.einsum('fyxc,yxc->f', P['W0'], patch)
    c0 = np.concatenate([clamp8(rne_shift(a0, s0)), xpad[y + 1, x + 1, h:].astype(np.int64)])
    r1 = clamp8(rne_shift(np.maximum(P['b1'] + P['W1'] @ c0, 0), s1))
    a2 = P['b2'] + P['W2'] @ r1
    return a2, xpad[y + 1, x + 1, :h].astype(np.int64), c0[h:]

def out_half(a2, res, a, r, o):
    v = (a2.astype(np.float32) * np.float32(2.0 ** -a) + res.astype(np.float32) * np.float32(2.0 ** -r)) * np.float32(2.0 ** o)
    return clamp8(np.round(v).astype(np.int64))

def forward_pixel(P, xpad, x, y, Q):
    a2, ra, rb = stage(P, xpad, x, y, Q['s0'], Q['s1']); h = P['C'] // 2
    return np.concatenate([out_half(a2[:h], ra, *Q['first']), out_half(a2[h:], rb, *Q['second'])])

def fit(P, xpad, samples):
    """samples: [((x, y), int8[C] from the AMD shader)] -> shifts/exponents reproducing all of them"""
    h = P['C'] // 2
    for s0, s1 in itertools.product(range(4, 12), repeat=2):
        st = [(stage(P, xpad, x, y, s0, s1), got) for (x, y), got in samples]
        Q = dict(s0=s0, s1=s1)
        for part, sl in (('first', slice(0, h)), ('second', slice(h, None))):
            for a, r, o in itertools.product(range(6, 18), range(2, 10), range(2, 10)):
                if all(np.array_equal(out_half(a2[sl], ra if part == 'first' else rb, a, r, o), got[sl]) for (a2, ra, rb), got in st):
                    Q[part] = (a, r, o); break
            else: break
        else: return Q
    return None

def forward_full(P, xpad, Q):
    """whole image exact: xpad int8 [H+2, W+2, C] -> int8 [H, W, C]"""
    C, h = P['C'], P['C'] // 2; H, W = xpad.shape[0] - 2, xpad.shape[1] - 2; x = xpad.astype(np.int64)
    a0 = np.broadcast_to(P['b0'], (H, W, h)).copy()
    for ky in range(3):
        for kx in range(3):
            a0 += x[ky:ky + H, kx:kx + W, :] @ P['W0'][:, ky, kx, :].T
    c0 = np.concatenate([clamp8(rne_shift(a0, Q['s0'])), x[1:-1, 1:-1, h:]], axis=2)
    r1 = clamp8(rne_shift(np.maximum(c0 @ P['W1'].T + P['b1'], 0), Q['s1']))
    a2 = r1 @ P['W2'].T + P['b2']
    return np.concatenate([out_half(a2[..., :h], x[1:-1, 1:-1, :h], *Q['first']), out_half(a2[..., h:], c0[..., h:], *Q['second'])], axis=2).astype(np.int8)
