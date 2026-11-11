# Copyright 2026 lonewolf0622
# SPDX-License-Identifier: GPL-2.0-or-later
"""gen_fpi8.py PASS.pkl OUTDIR: FasterNet block, one px per lane with all channels (C = 32: passes 4 / 5 / 10, + the fused 2x2 transposed
conv + skip of pass 11), int8 dot products with the weights as constants in the code (no weight loads, no workgroup memory, no barriers).
AMD's numbers exactly: int32 sums, power-of-two requants (the residual requant in integers where AMD's float path is exact).
Writes fpi8.comp, fpi8w.bin (biases only, unused), fpi8in.bin, fpi8ref.npy."""
import os as _os
ROOT = _os.environ.get('HLX_ROOT') or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys, os, pickle, struct, numpy as np
sys.path.insert(0, ROOT + '/ref')
import fasternet as fn
D = pickle.load(open(sys.argv[1], 'rb')); od = sys.argv[2]
P, Q, C, W, H = D['P'], D['Q'], D['C'], D['W'], D['H']
h, E = C // 2, 2 * C; CD, HD, ED = C // 4, h // 4, E // 4; NPL = C // 16
s0, s1 = Q['s0'], Q['s1']; CT = 'Wt' in D
pk = lambda v: struct.unpack('<I', np.asarray(v, np.int8).tobytes())[0]
dw = lambda v: [pk(v[4 * i:4 * i + 4]) for i in range(len(v) // 4)]
hx = lambda w: '0x%08xu' % w
ROW, PL = (W + 2) * 16, (W + 2) * 16 * (H + 2); inbytes = PL * NPL
L = []; e = L.append
def rq(acc, res, q, name):   # out = clamp8(round((acc * 2^-a + res * 2^-r) * 2^o)), integer form when the float path is exact
    a, r, o = q
    if a >= r and a > o and max(21 - a, 8 - r) - min(-a, -r) <= 24: return ' int %s = clamp(RS(%s + ((%s) << %d), %d), -128, 127);' % (name, acc, res, a - r, a - o)
    return ' int %s = clamp(int(roundEven((float(%s) * %r + float(%s) * %r) * %r)), -128, 127);' % (name, acc, 2.0 ** -a, res, 2.0 ** -r, 2.0 ** o)
# 3x3 grouped (each output: the 16 channels of its plane), taps v[plane][t][d]
for o in range(h):
    pl = o // 16; Wd = [dw(P['W0'][o, t // 3, t % 3, 16 * pl:16 * pl + 16]) for t in range(9)]
    e('  { int a = %d;' % int(P['b0'][o]) + ''.join(' a += SD(v%d[%d][%d], %s);' % (pl, t, d, hx(Wd[t][d])) for t in range(9) for d in range(4)) +
      ' c0[%d] |= (uint(clamp(RS(a, %d), -128, 127)) & 0xffu) << %du; }' % (o // 4, s0, 8 * (o % 4)))
# c0 second half = x
e('  ' + ' '.join('c0[%d] = xc[%d];' % (HD + i, HD + i) for i in range(CD - HD)))
for o in range(E):
    Wd = dw(P['W1'][o])
    e('  { int b = %d;' % int(P['b1'][o]) + ''.join(' b += SD(c0[%d], %s);' % (d, hx(Wd[d])) for d in range(CD)) +
      ' r1[%d] |= uint(clamp(RS(b, %d), 0, 127)) << %du; }' % (o // 4, s1, 8 * (o % 4)))
for o in range(C):
    Wd = dw(P['W2'][o]); q = Q['first'] if o < h else Q['second']
    e('  { int a = %d;' % int(P['b2'][o]) + ''.join(' a += SD(r1[%d], %s);' % (d, hx(Wd[d])) for d in range(ED)) +
      ' int r = int(xc[%d] << %du) >> 24;' % (o // 4, 24 - 8 * (o % 4)) + rq('a', 'r', q, 'q') + ' ow[%d] |= (uint(q) & 0xffu) << %du; }' % (o // 4, 8 * (o % 4)))
if CT:
    O, Wt, bt, cq = D['O'], D['Wt'], D['bt'], D['cq']; assert all(c == cq[0] for c in cq); OD = O // 4
    e('  uint x = xg + l; if (x >= W) return;')
    for pos in range(4):
        dx, dy = pos & 1, pos >> 1
        e('  { uint ox = 2u * x + %du, oy = 2u * y + %du, sb = (skipBase + (oy + 1u) * OROW + (ox + 1u) * 16u) >> 2, ob2 = (outBase + (oy + 1u) * OROW + (ox + 1u) * 16u) >> 2; uint tw[%d] = { %s };' % (dx, dy, OD, ', '.join(['0u'] * OD)))
        for pl in range(O // 16):
            e('    { uint so = sb; uvec4 sk = uvec4(xin[so], xin[so + 1u], xin[so + 2u], xin[so + 3u]);')
            for n in range(16 * pl, 16 * pl + 16):
                Wd = dw(Wt[pos, n])
                e('      { int a = %d;' % int(bt[pos, n]) + ''.join(' a += SD(ow[%d], %s);' % (d, hx(Wd[d])) for d in range(CD)) +
                  ' int r = int(sk[%d] << %du) >> 24;' % ((n % 16) // 4, 24 - 8 * (n % 4)) + rq('a', 'r', cq[0], 'q') + ' tw[%d] |= (uint(q) & 0xffu) << %du; }' % (n // 4, 8 * (n % 4)))
            e('    }')
        e('    ' + ' '.join('xout[ob2 + %du * (OPL >> 2) + %du] = tw[%d];' % (n4 // 4, n4 % 4, n4) for n4 in range(OD)) + ' }')
    assert O // 16 == 1   # (skip planes > 1: offset sb by the plane, not needed for pass 11)
    TAIL = ''
    CTH = '#define OROW %du\n#define OPL %du' % ((2 * W + 2) * 16, (2 * W + 2) * 16 * (2 * H + 2))
else:
    e('  uint x = xg + l; if (x >= W) return;')
    e('  ' + ' '.join('xout[((outBase + %du * PL + (y + 1u) * ROW + (x + 1u) * 16u) >> 2) + %du] = ow[%d];' % (i // 4, i % 4, i) for i in range(CD)))
    CTH = ''
BODY = '\n'.join(L)
G = f'''#version 450
#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_integer_dot_product : require
layout(local_size_x = 64) in;
layout(set = 0, binding = 0) readonly buffer In {{ uint xin[]; }};
layout(set = 0, binding = 1, std140) uniform Wt {{ uvec4 wt[4]; }};
layout(set = 0, binding = 2) writeonly buffer Out {{ uint xout[]; }};
layout(push_constant) uniform Pc {{ uint W, H, inBase, outBase, wBase, skipBase; }};
{CTH}
#define ROW {ROW}u
#define PL {PL}u
#define SD(a, b) dotPacked4x8EXT(int(a), int(b))   // signed x signed
int RS(int v, int s) {{ return (v + (1 << (s - 1)) - 1 + ((v >> s) & 1)) >> s; }}   // all int: HLSL turns int + uint arithmetic into uint (logical shift)
void main() {{
  uint xg = gl_WorkGroupID.x * 64u, y = gl_WorkGroupID.y, l = gl_LocalInvocationIndex, xr = min(xg + l, W - 1u);
''' + ''.join(f'''  uvec4 v{pl}[9]; [[unroll]] for (uint t = 0u; t < 9u; t++) {{ uint i = (inBase + {pl}u * PL + (y + t / 3u) * ROW + (xr + t % 3u) * 16u) >> 2; v{pl}[t] = uvec4(xin[i], xin[i + 1u], xin[i + 2u], xin[i + 3u]); }}
''' for pl in sorted(set(o // 16 for o in range(h)))) + f'''  uint xc[{CD}];''' + ''.join(f''' {{ uint i = (inBase + {pl}u * PL + (y + 1u) * ROW + (xr + 1u) * 16u) >> 2; {' '.join(f'xc[{4 * pl + d}] = xin[i + {d}u];' for d in range(4))} }}''' for pl in range(NPL)) + f'''
  uint c0[{CD}] = {{ {', '.join(['0u'] * CD)} }}; uint r1[{ED}] = {{ {', '.join(['0u'] * ED)} }}; uint ow[{CD}] = {{ {', '.join(['0u'] * CD)} }};
{BODY}
}}
'''
open(od + '/fpi8.comp', 'w').write(G)
open(od + '/fpi8w.bin', 'wb').write(np.zeros(16, np.uint32).tobytes())
rng = np.random.default_rng(3)
xpad = np.zeros((H + 2, W + 2, C), np.int8); xpad[1:-1, 1:-1] = rng.integers(-60, 60, (H, W, C))
raw = fn.to_planes(xpad); assert len(raw) == inbytes
fo = fn.forward_full(P, xpad, Q)
if not CT:
    open(od + '/fpi8in.bin', 'wb').write(raw); np.save(od + '/fpi8ref.npy', fo); skipb = 0; outb = inbytes
else:
    OW, OH = 2 * W, 2 * H
    spad = np.zeros((OH + 2, OW + 2, O), np.int8); spad[1:-1, 1:-1] = rng.integers(-60, 60, (OH, OW, O))
    open(od + '/fpi8in.bin', 'wb').write(raw + fn.to_planes(spad)); skipb = len(raw); outb = (OW + 2) * 16 * (OH + 2) * (O // 16)
    ref = np.zeros((OH, OW, O), np.int8); fo64 = fo.astype(np.int64)
    for pos in range(4):
        dx, dy = pos & 1, pos >> 1; acc = fo64 @ Wt[pos].T + bt[pos]
        for g, (a, r, oq) in enumerate(cq):
            sl = slice(16 * g, 16 * g + 16)
            v = (acc[..., sl].astype(np.float32) * np.float32(2.0 ** -a) + spad[1 + dy:1 + dy + OH:2, 1 + dx:1 + dx + OW:2, sl].astype(np.float32) * np.float32(2.0 ** -r)) * np.float32(2.0 ** oq)
            ref[dy::2, dx::2, sl] = np.clip(np.round(v), -128, 127)
    np.save(od + '/fpi8ref.npy', ref)
print('ok', (W + 63) // 64, H, outb, skipb, 'words', 16)
