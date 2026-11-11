# Copyright 2026 lonewolf0622
# SPDX-License-Identifier: GPL-2.0-or-later
"""gen_fni8.py PASS.pkl OUTDIR: FasterNet block (C = 32 or 64; + fused 2x2 transposed conv + skip for passes 9 / 11) on int8 dot products,
for GPUs that have them (RDNA 2/3 and later). AMD's int8 numbers exactly: int32 sums, power-of-two requants. Workgroup = 32 px of a row,
4 waves; each wave works its own output group (wave-uniform weights from the constant buffer), one px per lane; activations stay packed
4 x int8 per dword from load to store (AMD's plane-of-16 layout).
Writes fni8.comp, fni8w.bin, fni8in.bin, fni8ref.npy (self test: random input, exact CPU reference)."""
import os as _os
ROOT = _os.environ.get('HLX_ROOT') or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys, os, pickle, struct, numpy as np
sys.path.insert(0, ROOT + '/ref')
import fasternet as fn
D = pickle.load(open(sys.argv[1], 'rb')); od = sys.argv[2]
P, Q, C, W, H = D['P'], D['Q'], D['C'], D['W'], D['H']; assert C in (32, 64)
h, E = C // 2, 2 * C; CD, HD, ED = C // 4, h // 4, E // 4          # dwords per px: input, c0 first half, hidden
s0, s1 = Q['s0'], Q['s1']; QF, QS = Q['first'], Q['second']
CT = 'Wt' in D
pk = lambda v: struct.unpack('<I', np.asarray(v, np.int8).tobytes())[0]
def packn(v): v = np.asarray(v); assert v.min() >= -128 and v.max() <= 127 and len(v) % 4 == 0; return [pk(v[4 * i:4 * i + 4]) for i in range(len(v) // 4)]
words = []
B0 = len(words) // 4                      # W0[o][t]: one uvec4 = the 16 inputs of output o's plane at tap t
for o in range(h):
    for t in range(9): words += packn(P['W0'][o, t // 3, t % 3, 16 * (o // 16):16 * (o // 16) + 16])
B1 = len(words) // 4                      # W1[o]: C/16 uvec4
for o in range(E): words += packn(P['W1'][o])
B2 = len(words) // 4                      # W2[o]: E/16 uvec4
for o in range(C): words += packn(P['W2'][o])
BT = len(words) // 4
if CT:
    O, Wt, bt, cq = D['O'], D['Wt'], D['bt'], D['cq']; assert O in (16, 32) and all(c == cq[0] for c in cq)
    for p in range(4):
        for n in range(O): words += packn(Wt[p, n])
BB = len(words)                           # biases, int32 (dword index)
bias = list(P['b0']) + list(P['b1']) + list(P['b2']) + ([int(v) for p in range(4) for v in bt[p]] if CT else [])
words += [int(v) & 0xFFFFFFFF for v in bias]
while len(words) % 4: words.append(0)
assert len(words) * 4 <= 65536, len(words) * 4
b0i, b1i, b2i, bti = BB, BB + h, BB + h + E, BB + h + E + C
ROW, PL = (W + 2) * 16, (W + 2) * 16 * (H + 2); inbytes = PL * (C // 16)
f = lambda e: repr(float(2.0 ** e))
NO3, NOE, NOC = h // 4, E // 4, C // 4    # outputs per wave: 3x3, expand, contract
def requant(acc, res, q):   # -> an int expression: round((acc * 2^-a + res * 2^-r) * 2^o) (integer form when the float path is exact)
    a, r, o = q
    if a >= r and a > o and max(21 - a, 8 - r) - min(-a, -r) <= 24: return f'RS({acc} + (({res}) << {a - r}), {a - o})'
    return f'int(roundEven((float({acc}) * {f(-a)} + float({res}) * {f(-r)}) * {f(o)}))'
# block output: wave w has outputs NOC*w .. NOC*w+NOC-1 = dwords (NOC/4)*w + k; first half [0, h) requant QF, second QS (residual = x in both)
BLOCK = f'''  uint ow[{NOC // 4}];
  [[unroll]] for (uint k = 0u; k < {NOC // 4}u; k++) {{ uint dw = {NOC // 4}u * w + k, word = 0u, xr = XC(px, dw); bool fh = dw < {h // 4}u;
    [[unroll]] for (uint j = 0u; j < 4u; j++) {{ int r = int(xr << (24u - 8u * j)) >> 24; int a = c[4u * k + j];
      int q = fh ? {requant('a', 'r', QF)} : {requant('a', 'r', QS)};
      word |= (uint(clamp(q, -128, 127)) & 0xffu) << (8u * j); }}
    ow[k] = word; }}
'''
if CT:
    ca, cr, co = cq[0]; OD = O // 4
    TAIL = BLOCK + f'''  barrier();   // every wave has finished reading the hidden values
  [[unroll]] for (uint k = 0u; k < {NOC // 4}u; k++) HB(px, {NOC // 4}u * w + k) = ow[k];
  barrier();
  // transposed conv 2x2: wave w = output position, {O} channels
  {{ uint fo[{CD}]; [[unroll]] for (uint q = 0u; q < {CD}u; q++) fo[q] = HB(px, q);
    uint x = xg + px; if (x >= W) return;
    uint ox = 2u * x + (w & 1u), oy = 2u * y + (w >> 1), sb = (skipBase + (oy + 1u) * OROW + (ox + 1u) * 16u) >> 2, ob2 = (outBase + (oy + 1u) * OROW + (ox + 1u) * 16u) >> 2;
    [[unroll]] for (uint pl = 0u; pl < {O // 16}u; pl++) [[unroll]] for (uint k = 0u; k < 4u; k++) {{ uint word = 0u, sw = xin[sb + pl * (OPL >> 2) + k];
      [[unroll]] for (uint j = 0u; j < 4u; j++) {{ uint n = pl * 16u + 4u * k + j; int a = BI({bti}u + w * {O}u + n);
        uint wi = {BT}u + (w * {O}u + n) * {CD // 4}u;
        [[unroll]] for (uint q = 0u; q < {CD // 4}u; q++) {{ uvec4 ww = wt[wi + q]; [[unroll]] for (uint e = 0u; e < 4u; e++) a += SD(fo[4u * q + e], ww[e]); }}
        int sk = int(sw << (24u - 8u * j)) >> 24;
        int qq = {requant('a', 'sk', (ca, cr, co))};
        word |= (uint(clamp(qq, -128, 127)) & 0xffu) << (8u * j); }}
      xout[ob2 + pl * (OPL >> 2) + k] = word; }} }}
'''
    CTH = '#define OROW %du\n#define OPL %du' % ((2 * W + 2) * 16, (2 * W + 2) * 16 * (2 * H + 2))
else:
    TAIL = BLOCK + f'''  uint x = xg + px; if (x >= W) return;
  [[unroll]] for (uint k = 0u; k < {NOC // 4}u; k++) {{ uint dw = {NOC // 4}u * w + k; xout[((outBase + (dw >> 2u) * PL + (y + 1u) * ROW + (x + 1u) * 16u) >> 2) + (dw & 3u)] = ow[k]; }}
'''
    CTH = ''
G = f'''#version 450
#extension GL_EXT_control_flow_attributes : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_EXT_integer_dot_product : require
layout(local_size_x = 128) in;
layout(set = 0, binding = 0) readonly buffer In {{ uint xin[]; }};
layout(set = 0, binding = 1, std140) uniform Wt {{ uvec4 wt[{len(words) // 4}]; }};
layout(set = 0, binding = 2) writeonly buffer Out {{ uint xout[]; }};
layout(push_constant) uniform Pc {{ uint W, H, inBase, outBase, wBase, skipBase; }};
{CTH}
#define ROW {ROW}u
#define PL {PL}u
// input rows 0..2 x 34 px x {CD} dwords, c0 (first half; the second half is x) 32 px x {HD} dwords, hidden 32 px x {ED} dwords
shared uint IN[3 * 34 * {CD + 1}];
shared uint C0[32 * {HD + 1}];
shared uint HID[32 * {ED + 1}];
#define I3(r, p, d) IN[((r) * 34u + (p)) * {CD + 1}u + (d)]
#define XC(p, d) I3(1u, (p) + 1u, d)
#define HB(p, d) HID[(p) * {ED + 1}u + (d)]
int RS(int v, int s) {{ return (v + (1 << (s - 1)) - 1 + ((v >> s) & 1)) >> s; }}   // all int: HLSL turns int + uint arithmetic into uint (logical shift)
uint P8(int a, int b, int c, int d) {{ return (uint(clamp(a, -128, 127)) & 0xffu) | ((uint(clamp(b, -128, 127)) & 0xffu) << 8u) | ((uint(clamp(c, -128, 127)) & 0xffu) << 16u) | ((uint(clamp(d, -128, 127)) & 0xffu) << 24u); }}
#define SD(a, b) dotPacked4x8EXT(int(a), int(b))   // signed x signed
int BI(uint i) {{ return int(wt[i >> 2u][i & 3u]); }}
void main() {{
  uint xg = gl_WorkGroupID.x * 32u, y = gl_WorkGroupID.y, l = gl_LocalInvocationIndex;
  for (uint e = l; e < 3u * 34u * {C // 16}u; e += 128u) {{
    uint pl = e / 102u, r = (e / 34u) % 3u, px = e % 34u, gx = min(xg + px, W + 1u);
    uint i = (inBase + pl * PL + (y + r) * ROW + gx * 16u) >> 2;
    [[unroll]] for (uint d = 0u; d < 4u; d++) I3(r, px, pl * 4u + d) = xin[i + d]; }}
  barrier();
  uint w = gl_SubgroupID, px = l & 31u;
  // 3x3 grouped: outputs {NO3}w..{NO3}w+{NO3 - 1}, each from the 16 channels of its plane
  {{ uint pl3 = ({NO3}u * w) >> 4u; uint t9[36]; [[unroll]] for (uint t = 0u; t < 9u; t++) [[unroll]] for (uint d = 0u; d < 4u; d++) t9[4u * t + d] = I3(t / 3u, px + t % 3u, pl3 * 4u + d);
    int a[{NO3}];
    [[unroll]] for (uint j = 0u; j < {NO3}u; j++) {{ uint o = {NO3}u * w + j; a[j] = BI({b0i}u + o);
      [[unroll]] for (uint t = 0u; t < 9u; t++) {{ uvec4 ww = wt[{B0}u + o * 9u + t]; [[unroll]] for (uint d = 0u; d < 4u; d++) a[j] += SD(t9[4u * t + d], ww[d]); }} }}
    [[unroll]] for (uint k = 0u; k < {NO3 // 4}u; k++) C0[px * {HD + 1}u + {NO3 // 4}u * w + k] = P8(RS(a[4u * k], {s0}), RS(a[4u * k + 1u], {s0}), RS(a[4u * k + 2u], {s0}), RS(a[4u * k + 3u], {s0})); }}
  barrier();
  // expand {C} -> {E} + ReLU: outputs {NOE}w..
  {{ uint c0[{CD}]; [[unroll]] for (uint d = 0u; d < {HD}u; d++) c0[d] = C0[px * {HD + 1}u + d]; [[unroll]] for (uint d = 0u; d < {HD}u; d++) c0[{HD}u + d] = XC(px, {HD}u + d);
    [[dont_unroll]] for (uint g = 0u; g < {NOE // 4}u; g++) {{ int b[4];
      [[unroll]] for (uint j = 0u; j < 4u; j++) {{ uint o = {NOE}u * w + 4u * g + j; b[j] = BI({b1i}u + o);
        [[unroll]] for (uint q = 0u; q < {CD // 4}u; q++) {{ uvec4 ww = wt[{B1}u + o * {CD // 4}u + q]; [[unroll]] for (uint d = 0u; d < 4u; d++) b[j] += SD(c0[4u * q + d], ww[d]); }}
        b[j] = clamp(RS(b[j], {s1}), 0, 127); }}
      HB(px, {NOE // 4}u * w + g) = uint(b[0]) | (uint(b[1]) << 8u) | (uint(b[2]) << 16u) | (uint(b[3]) << 24u); }} }}
  barrier();
  // contract {E} -> {C}: outputs {NOC}w..
  int c[{NOC}];
  {{ uint r1[{ED}]; [[unroll]] for (uint d = 0u; d < {ED}u; d++) r1[d] = HB(px, d);
    [[unroll]] for (uint j = 0u; j < {NOC}u; j++) {{ uint o = {NOC}u * w + j; c[j] = BI({b2i}u + o);
      [[unroll]] for (uint q = 0u; q < {ED // 4}u; q++) {{ uvec4 ww = wt[{B2}u + o * {ED // 4}u + q]; [[unroll]] for (uint d = 0u; d < 4u; d++) c[j] += SD(r1[4u * q + d], ww[d]); }} }} }}
{TAIL}}}
'''
open(od + '/fni8.comp', 'w').write(G)
open(od + '/fni8w.bin', 'wb').write(np.array(words, np.uint32).tobytes())
rng = np.random.default_rng(3)
xpad = np.zeros((H + 2, W + 2, C), np.int8); xpad[1:-1, 1:-1] = rng.integers(-60, 60, (H, W, C))
raw = fn.to_planes(xpad); assert len(raw) == inbytes
fo = fn.forward_full(P, xpad, Q)
if not CT:
    open(od + '/fni8in.bin', 'wb').write(raw); np.save(od + '/fni8ref.npy', fo); skipb = 0; outb = inbytes
else:
    OW, OH = 2 * W, 2 * H
    spad = np.zeros((OH + 2, OW + 2, O), np.int8); spad[1:-1, 1:-1] = rng.integers(-60, 60, (OH, OW, O))
    open(od + '/fni8in.bin', 'wb').write(raw + fn.to_planes(spad)); skipb = len(raw); outb = (OW + 2) * 16 * (OH + 2) * (O // 16)
    ref = np.zeros((OH, OW, O), np.int8); fo64 = fo.astype(np.int64)
    for pos in range(4):
        dx, dy = pos & 1, pos >> 1; acc = fo64 @ Wt[pos].T + bt[pos]
        for g, (a, r, oq) in enumerate(cq):
            sl = slice(16 * g, 16 * g + 16)
            v = (acc[..., sl].astype(np.float32) * np.float32(2.0 ** -a) + spad[1 + dy:1 + dy + OH:2, 1 + dx:1 + dx + OW:2, sl].astype(np.float32) * np.float32(2.0 ** -r)) * np.float32(2.0 ** oq)
            ref[dy::2, dx::2, sl] = np.clip(np.round(v), -128, 127)
    np.save(od + '/fni8ref.npy', ref)
print('ok', (W + 31) // 32, H, outb, skipb, 'words', len(words))
