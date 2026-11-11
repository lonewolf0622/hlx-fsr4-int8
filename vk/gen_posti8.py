# Copyright 2026 lonewolf0622
# SPDX-License-Identifier: GPL-2.0-or-later
"""gen_posti8.py OUT.hlsl : own postpass for HLX FSR 4 Full's INT8 mode (D3D12, AMD's bindings, loaded in place of AMD's postpass under its
hash): last ConvNeXt block + 2x2 transposed conv on int8 dot products (weights as constants), then the reconstruction. AMD's numbers
exactly: the float steps follow AMD's operation order. Own structure: the tone-mapped render-resolution colour of the group's area is
computed once into workgroup memory (each render pixel's log2 once, not once per output tap); 16 x 16 half-resolution px per group.
Env: POSTSH (AMD's shader index, for the weights), ROW (scratch row bytes: 15392 1080p family, 30752 4K), AE (1: auto exposure; 0: exposure texture t6, copied to u9 for the next frame), NL (1: non-linear colour: raw colour taps,
no tone map, no output conversion), BLOB (the model's initializer blob: Ultra Performance 0x1658460)."""
import os as _os
ROOT = _os.environ.get('HLX_ROOT') or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import os, re, sys, numpy as np
sys.path.insert(0, ROOT + '/ref')
import post
from convnext import i8
out = sys.argv[1]; ROW = int(os.environ.get('ROW', '15392')); AE = int(os.environ.get('AE', '1')); NL = int(os.environ.get('NL', '0')); TILE = int(os.environ.get('TILE', '1'))
sh, bufs, cbs, c = post.make()
E = post.extract_post(post.run(sh, bufs, cbs, 50, 40)); P = E['P']
ir = open(ROOT + '/work/dis/%04d.dxil.txt' % post.SH).read()
shifts = [int(n) for n in re.findall(r'ashr i32 %\d+, (\d+)', ir)]; s0, s1 = shifts[0], shifts[16]
import struct
pk = lambda v: struct.unpack('<I', np.asarray(v, np.int8).tobytes())[0]
dw = lambda v: [pk(v[4 * i:4 * i + 4]) for i in range(len(v) // 4)]
hx = lambda w: '0x%08xu' % w
K = '0.0007044409285299479961395263671875'
L = []; e = L.append
# block: 3x3 (taps v[t], zero outside the half-resolution image), expand + ReLU, contract, residual (acc * 2^-10 + x / 16) * 16 in integers
IL = int(os.environ.get('IL', '8'))   # interleave IL outputs' dot products at a time (independent neighbours: no dependency stalls on RDNA 3); 0: one chain per output
def stage(nout, bias, nin, reg, wts, finish):
    if not IL:
        for o in range(nout): e('  { int a = %d;' % bias[o] + ''.join(' a += SD(%s, %s);' % (reg(d), hx(wts(o)[d])) for d in range(nin)) + ' ' + finish(o, 'a') + ' }')
        return
    for g0 in range(0, nout, IL):   # groups of IL outputs interleaved (fewer live accumulators than all at once)
        og = range(g0, min(g0 + IL, nout))
        e('  { int ' + ', '.join('a%d = %d' % (o, bias[o]) for o in og) + ';')
        for d in range(nin): e('   ' + ' '.join('a%d += SD(%s, %s);' % (o, reg(d), hx(wts(o)[d])) for o in og))
        e('   ' + ' '.join(finish(o, 'a%d' % o) for o in og) + ' }')
W0 = [[w for t in range(9) for w in dw(P['W0'][o, t // 3, t % 3])] for o in range(16)]
stage(16, [int(v) for v in P['b0']], 36, lambda d: 'v[%d][%d]' % (d // 4, d % 4), lambda o: W0[o],
      lambda o, a: 'c0[%d] |= (uint(clamp(RS(%s, %d), -128, 127)) & 0xffu) << %du;' % (o // 4, a, s0, 8 * (o % 4)))
stage(32, [int(v) for v in P['b1']], 4, lambda d: 'c0[%d]' % d, lambda o: dw(P['W1'][o]),
      lambda o, a: 'r1[%d] |= uint(clamp(RS(%s, %d), 0, 127)) << %du;' % (o // 4, a, s1, 8 * (o % 4)))
stage(16, [int(v) for v in P['b2']], 8, lambda d: 'r1[%d]' % d, lambda o: dw(P['W2'][o]),
      lambda o, a: '{ int r = int(v[4][%d] << %du) >> 24; int q = RS(%s + (r << 6), 6); bo[%d] |= (uint(clamp(q, -128, 127)) & 0xffu) << %du; }' % (o // 4, 24 - 8 * (o % 4), a, o // 4, 8 * (o % 4)))
BLOCK = '\n'.join(L); L = []; e = L.append
Wct, bct = E['Wct'], E['bct']
LOOP = int(os.environ.get('LOOP', '0'))   # sub-pixels as a real loop (weights from a small constant table): one sub-pixel's values live at a time
def ct(row):
    if LOOP: k = row % 8; return '(CTB[s * 8u + %du] + SD(bo[0], CTW[(s * 8u + %du) * 4u]) + SD(bo[1], CTW[(s * 8u + %du) * 4u + 1u]) + SD(bo[2], CTW[(s * 8u + %du) * 4u + 2u]) + SD(bo[3], CTW[(s * 8u + %du) * 4u + 3u]))' % ((k,) * 5)
    return '(%d' % int(bct[row]) + ''.join(' + SD(bo[%d], %s)' % (d, hx(dw(Wct[row])[d])) for d in range(4)) + ')'
CTT = 'static const int CTB[32] = { %s };\nstatic const uint CTW[128] = { %s };' % (', '.join(str(int(v)) for v in bct[:32]), ', '.join(hx(w) for r in range(32) for w in dw(Wct[r])))
# kk is precise: the compiler reassociates the product under fast math otherwise (AMD's order differs)
TAPS = [(0, -1), (-1, -1), (1, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]   # AMD's summation order (tap coordinates go to float as unsigned, as AMD's: -1 at the image edge is 2^32 - 1, weight ~0)
nm = lambda dx, dy: '%s%s' % ('mpo'[dx + 1], 'mpo'[dy + 1])
for s in (range(1) if LOOP else range(4)):
    sx, sy, R0 = s & 1, s >> 1, 8 * s
    e('  [loop] for (uint s = 0u; s < 4u; s++) { uint X = 2u * lx + (s & 1u), Y = 2u * ly + (s >> 1u);' if LOOP else '  { uint X = 2u * lx + %du, Y = 2u * ly + %du;' % (sx, sy))
    e('    Rec[uint2(X, Y)] = float16_t4(' + ', '.join('(float16_t)(1.0 / (exp2(float(%s) * (-%s)) + 1.0))' % (ct(R0 + k), K) for k in range(4)) + ');')
    e('    float ac = float(%s); float e1 = exp2(ac * %s), e2 = exp2(ac * (-%s));' % (ct(R0 + 4), K, K))
    e('    float fx = 2.0 / (exp2(float(%s) * (-%s)) + 1.0); float fy = 2.0 / (exp2(float(%s) * (-%s)) + 1.0); float fx2 = fx * fx, fy2 = fy * fy;' % (ct(R0 + 5), K, ct(R0 + 6), K))
    e('    float lrx = (m[1].z + (-0.5)) + (m[1].x * (float(X) + 0.5)), lry = (m[1].w + (-0.5)) + (m[1].y * (float(Y) + 0.5));')
    e('    int lix = int(round(lrx)), liy = int(round(lry)); precise float kk = ((((e1 - e2) / (e2 + e1)) * 2.0) * fx) * fy;')
    for dy in (-1, 0, 1):
        e('    float dy%s = (float(uint(liy + %d)) - lry) * m[0].w; float A%s = dy%s * kk, B%s = (dy%s * dy%s) * fy2;' % ((('mpo'[dy + 1],) + (dy,) + ('mpo'[dy + 1],) * 5)))
    for dx in (-1, 0, 1):
        e('    float dx%s = (float(uint(lix + %d)) - lrx) * m[0].z; float C%s = dx%s * fx2;' % ('mpo'[dx + 1], dx, 'mpo'[dx + 1], 'mpo'[dx + 1]))
    for i, (dx, dy) in enumerate(TAPS):   # tap by tap, accumulated in AMD's order (fewer live values)
        a, b = 'mpo'[dx + 1], 'mpo'[dy + 1]; n = a + b
        e('    float w%s = exp2((((A%s + C%s) * dx%s) + B%s) * (-3.2654933929443359375)); float3 t%s = TM(lix + %d, liy + %d);' % (n, b, a, a, b, n, dx, dy))
        if i == 0: e('    float sx = t%s.x * w%s, sy = t%s.y * w%s, sz = t%s.z * w%s; float ws = w%s;' % ((n,) * 7))
        elif i < 8: e('    sx = sx + (t%s.x * w%s); sy = sy + (t%s.y * w%s); sz = sz + (t%s.z * w%s); ws = ws + w%s;' % ((n,) * 7))
        else: e('    ws = ws + w%s;' % n)
    e('    float bl = 1.0 / (exp2(float(%s) * (-%s)) + 1.0); float ib = 1.0 - bl; float4 rp = Rp.Load(int3(X, Y, 0));' % (ct(R0 + 7), K))
    e('    float h0 = (rp.x * bl) + (ib * ((sx + (too.x * woo)) / ws)), h1 = (rp.y * bl) + (ib * ((sy + (too.y * woo)) / ws)), h2 = (rp.z * bl) + (ib * ((sz + (too.z * woo)) / ws));')
    e('    Hist[uint2(X, Y)] = float4(h0, h1, h2, h0);')
    e('    float o0 = h0, o1 = h1, o2 = h2;')
    if not NL: e('    if (asuint(m[5].w) == 0u) { o0 = max((exp2(h0 * 12.28870391845703125) + (-1.0)) * 0.006666666828095912933349609375, 0.0) / ex; o1 = max((exp2(h1 * 12.28870391845703125) + (-1.0)) * 0.006666666828095912933349609375, 0.0) / ex; o2 = max((exp2(h2 * 12.28870391845703125) + (-1.0)) * 0.006666666828095912933349609375, 0.0) / ex; }')
    e('    float q0 = min(max(o0, 0.0), 64000.0); Out[uint2(X, Y)] = float4(q0, min(max(o1, 0.0), 64000.0), min(max(o2, 0.0), 64000.0), q0); }')
TAIL = '\n'.join(L)
G = f'''// own postpass (HLX FSR 4 Full INT8 mode), generated by gen_posti8.py
cbuffer C : register(b1) {{ float4 m[7]; }};
Texture2D<float4> Color : register(t3); Texture2D<float4> Rp : register(t9); {'Texture2D<float4> AExp : register(t17);' if AE else 'Texture2D<float4> Expo : register(t6); RWTexture2D<float4> ExpOut : register(u9);'}
RWByteAddressBuffer Scratch : register(u13); RWTexture2D<float4> Hist : register(u1); RWTexture2D<float4> Out : register(u2); RWTexture2D<float16_t4> Rec : register(u6);   // (16-bit store, as AMD's: the 8-bit recurrent texture converts from it)
int SD(uint a, uint b) {{ return dot4add_i8packed(a, b, 0); }}
int RS(int v, int s) {{ return (v + (1 << (s - 1)) - 1 + ((v >> s) & 1)) >> s; }}
float H16(float v) {{ return float((float16_t)v); }}   // round to nearest even (f32tof16 can become a round-toward-zero pack)
groupshared float T0[1296], T1[1296], T2[1296];   // 36 x 36: the taps of 32 px span at most ceil(31 r) + 5 render px (r = render / output <= 1); 15.5 KB, 4 groups per CU
{CTT if LOOP else ''}
static int ox, oy; static float e150;
float3 TMd(int x, int y) {{ uint cx = uint(min(int(uint(max(x, 0))), int(asuint(m[4].w) - 1u))), cy = uint(min(int(uint(max(y, 0))), int(asuint(m[5].x) - 1u)));
  float4 c = Color.Load(int3(cx, cy, 0)); {'if (true) return c.xyz;' if NL else ''} return float3(max(log2((c.x * e150) + 1.0) * 0.08137547969818115234375, 0.0), max(log2((c.y * e150) + 1.0) * 0.08137547969818115234375, 0.0), max(log2((c.z * e150) + 1.0) * 0.08137547969818115234375, 0.0)); }}
float3 TM(int x, int y) {{ {'uint k = uint(clamp(y - oy, 0, 35)) * 36u + uint(clamp(x - ox, 0, 35)); return float3(T0[k], T1[k], T2[k]);' if TILE else 'return TMd(x, y);'} }}   // (render <= output size and jitter within a pixel: every tap is inside the 36 x 36 tile)
[numthreads(256, 1, 1)] void main(uint3 g : SV_GroupID, uint l : SV_GroupIndex) {{
  uint W2 = uint(int(asuint(m[4].x)) / 2), H2 = uint(int(asuint(m[4].y)) / 2);
{'  float ex = AExp.Load(int3(0, 0, 0)).x / m[5].y;' if AE else '  float e0 = Expo.Load(int3(0, 0, 0)).x; if ((g.x | g.y | l) == 0u) ExpOut[uint2(1, 0)] = ((e0 == 0.0) ? 1.0 : e0).xxxx; float ex = ((e0 == 0.0) ? 1.0 : e0) / m[5].y;'} e150 = ex * 150.0;
  ox = int(floor((m[1].z + (-0.5)) + (m[1].x * (float(g.x * 32u) + 0.5)))) - 2; oy = int(floor((m[1].w + (-0.5)) + (m[1].y * (float(g.y * 32u) + 0.5)))) - 2;
  uint lx = g.x * 16u + (l & 15u), ly = g.y * 16u + (l >> 4); bool valid = lx < W2 && ly < H2;   // (the block first, then the tile: their registers are not live together)
  uint bo[4] = {{ 0u, 0u, 0u, 0u }};
  if (valid) {{
  uint4 v[9];
  [unroll] for (uint t = 0; t < 9; t++) {{ int tx = int(lx) + int(t % 3u) - 1, ty = int(ly) + int(t / 3u) - 1;
    v[t] = (tx >= 0 && tx < int(W2) && ty >= 0 && ty < int(H2)) ? Scratch.Load4({ROW}u * uint(ty + 1) + 16u * uint(tx + 1)) : uint4(0, 0, 0, 0); }}
  uint4 xc = Scratch.Load4({ROW}u * (ly + 1u) + 16u * (lx + 1u)); v[4] = xc;
  uint c0[4] = {{ 0u, 0u, 0u, 0u }}; uint r1[8] = {{ 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u }};
{BLOCK}
  }}
{"  { uint sx = min(uint(ceil(m[1].x * 31.0)) + 5u, 36u), sy = min(uint(ceil(m[1].y * 31.0)) + 5u, 36u), n = sx * sy;   // (the area this group's taps reach: 26 x 26 at 1.5x)\n    for (uint i = l; i < n; i += 256u) { uint ty = i / sx, tx = i - ty * sx; float3 t = TMd(ox + int(tx), oy + int(ty)); uint k = ty * 36u + tx; T0[k] = t.x; T1[k] = t.y; T2[k] = t.z; } }" if TILE else ""}
  GroupMemoryBarrierWithGroupSync();
  if (!valid) return;
{TAIL}
}}
'''
open(out, 'w').write(G); print(out, 'ROW', ROW, 's0', s0, 's1', s1, 'ct biases', list(bct[:8]))
