# Copyright 2026 lonewolf0622
# SPDX-License-Identifier: GPL-2.0-or-later
"""ct2d.py: FSR 4.1.2 FNB_CT2D_ADD pass (pass9: C=64 -> 32, pass11: C=32 -> 16) - FasterNet block at low resolution,
2x2 stride-2 transposed conv to 2x2 output pixels, plus a skip input, quantised.

  fnb   = FasterNet(x)                                    (C ch, int8; params fitted like fasternet.py)
  acc   = bt + sum_c Wt[pos, o, c] * fnb[c]               (pos = 0..3: (0,0) (1,0) (0,1) (1,1))
  out   = clamp8(round_ne((acc * 2^-a + skip * 2^-r) * 2^o))   per group of 16 output channels
"""
import itertools, struct, numpy as np
from convnext import i8, clamp8
import fasternet as fn

def split(th, C):
    d = [t for t in th.trace if t[0] == 'dot4']
    init = {t[1]: (t[2], t[3]) for t in th.trace if t[0] == 'init'}
    order = []
    for t in d:
        if t[1] not in order: order.append(t[1])
    by = {c: [t for t in d if t[1] == c] for c in order}
    nf = C // 2 + 2 * C + C
    return d, init, order[:nf], order[nf:], by

def extract_ct(th, C, O):
    """transposed-conv weights Wt[4, O, C], biases, and the FNB output this thread fed it (int8[C])"""
    d, init, fnbc, ctc, by = split(th, C)
    assert len(ctc) == 4 * O and all(len(by[c]) == C // 4 for c in ctc), (len(ctc), [len(by[c]) for c in ctc[:2]])
    Wt = np.zeros((4, O, C), np.int64); bt = np.zeros((4, O), np.int64)
    for k, c in enumerate(ctc):
        pos, o = divmod(k, O)
        Wt[pos, o] = np.concatenate([i8(t[4]) for t in by[c]])        # weight: second operand (from the buffer)
        bt[pos, o] = init[c][0]
    fnb = np.concatenate([i8(t[2]) for t in by[ctc[0]]])              # activations of the first chain = FNB output
    return dict(Wt=Wt, bt=bt, fnb=fnb, bias_src=init[ctc[0]][1])

def ct_out(acc, skip, a, r, o):
    v = (acc.astype(np.float32) * np.float32(2.0 ** -a) + skip.astype(np.float32) * np.float32(2.0 ** -r)) * np.float32(2.0 ** o)
    return clamp8(np.round(v).astype(np.int64))
