# Copyright 2026 lonewolf0622
# SPDX-License-Identifier: GPL-2.0-or-later
"""gen_cnxi8.py SHADER_INDEX OUTDIR: ConvNeXt block (16 ch; passes 1 / 2 / 12) on int8 dot products, AMD's numbers exactly.
Workgroup = 64 px of a row (2 waves), one px per lane, all 16 channels per lane. IMM=1 (default): weights and biases are constants in
the code (straight-line dot products with immediate operands, no weight loads); IMM=0: from the constant buffer (wave-uniform).
The 3 input rows staged once in workgroup memory; packed int8 from load to store. Weights traced from AMD's shader (BLOB env: the
model's initializer blob, as gen_cnx2f). Writes cnxi8.comp, cnxi8w.bin, cnxi8in.bin, cnxi8ref.npy."""
import os as _os
ROOT = _os.environ.get('HLX_ROOT') or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys, os, struct, numpy as np
sys.path.insert(0, ROOT + '/ref')
from dxil_interp import Shader
from convnext import blob, extract, trace_thread, params_from_ir, forward_full
k, od = int(sys.argv[1]), sys.argv[2]; IMM = int(os.environ.get('IMM', '1')); NOLDS = int(os.environ.get('NOLDS', '1'))
W, H, C, row = 960, 540, 16, 15392
text = open(ROOT + '/work/dis/%04d.dxil.txt' % k).read(); sh = Shader(text)
scratch = bytearray(160 << 20); cbuf = bytearray(4096)
for r in range(32): struct.pack_into('<II', cbuf, 16 * r, W, H)
bufs = {r: (blob() if r[0] == 't' else scratch) for r in set(sh.res.values()) if r[0] in 'tu'}
cbs = {r: bytes(cbuf) for r in set(sh.res.values()) if r[0] == 'b'}
P = extract(trace_thread(sh, 100, 100, bufs, cbs), 100, 100); Q = params_from_ir(text)
pk = lambda v: struct.unpack('<I', np.asarray(v, np.int8).tobytes())[0]
p16 = lambda v: [pk(v[4 * i:4 * i + 4]) for i in range(len(v) // 4)]
W0 = [[p16(P['W0'][o, t // 3, t % 3]) for t in range(9)] for o in range(16)]     # [o][t] -> 4 dwords
W1 = [p16(P['W1'][o]) for o in range(32)]                                        # [o] -> 4 dwords
W2 = [p16(P['W2'][o]) for o in range(16)]                                        # [o] -> 8 dwords
words = [d for o in W0 for t in o for d in t] + [d for o in W1 for d in o] + [d for o in W2 for d in o]
B1, B2 = 16 * 9, 16 * 9 + 32
BB = len(words); words += [int(v) & 0xFFFFFFFF for v in list(P['b0']) + list(P['b1']) + list(P['b2'])]
while len(words) % 4: words.append(0)
assert len(words) * 4 <= 65536
f = lambda e: repr(float(2.0 ** e))
L = []; e = L.append
e2, ex, eo = Q['e2'], Q['ex'], Q['eo']; INTF = e2 >= ex and e2 > eo and max(20 - e2, 8 - ex) - min(-e2, -ex) <= 24   # |acc| < 2^20, |x| < 2^8: float sum exact
def fin(o):   # block output o: round((acc * 2^-e2 + x * 2^-ex) * 2^eo), exactly
    r = 'int(v[4][%d] << %du) >> 24' % (o // 4, 24 - 8 * (o % 4))
    if INTF: return ' int t = a + ((%s) << %d); int q = RS(t, %d);' % (r, e2 - ex, e2 - eo)
    return ' float vv = (float(a) * %s + float(%s) * %s) * %s; int q = int(roundEven(vv));' % (f(-e2), r, f(-ex), f(eo))
if IMM:
    hx = lambda w: '0x%08xu' % w
    for o in range(16):
        e('  { int a = %d;' % int(P['b0'][o]) + ''.join(' a += SD(v[%d][%d], %s);' % (t, d, hx(W0[o][t][d])) for t in range(9) for d in range(4)) +
          ' c0[%d] |= (uint(clamp(RS(a, %d), -128, 127)) & 0xffu) << %du; }' % (o // 4, Q['s0'], 8 * (o % 4)))
    for o in range(32):
        e('  { int b = %d;' % int(P['b1'][o]) + ''.join(' b += SD(c0[%d], %s);' % (d, hx(W1[o][d])) for d in range(4)) +
          ' r1[%d] |= uint(clamp(RS(b, %d), 0, 127)) << %du; }' % (o // 4, Q['s1'], 8 * (o % 4)))
    for o in range(16):
        e('  { int a = %d;' % int(P['b2'][o]) + ''.join(' a += SD(r1[%d], %s);' % (d, hx(W2[o][d])) for d in range(8)) +
          fin(o) + ' ow[%d] |= (uint(clamp(q, -128, 127)) & 0xffu) << %du; }' % (o // 4, 8 * (o % 4)))
else:
    for o in range(16):
        e('  { uint o = %du; int a = BI(%du + o); [[unroll]] for (uint t = 0u; t < 9u; t++) { uvec4 ww = wt[o * 9u + t]; [[unroll]] for (uint d = 0u; d < 4u; d++) a += SD(v[t][d], ww[d]); }' % (o, BB) +
          ' c0[%d] |= (uint(clamp(RS(a, %d), -128, 127)) & 0xffu) << %du; }' % (o // 4, Q['s0'], 8 * (o % 4)))
    for o in range(32):
        e('  { uint o = %du; uvec4 ww = wt[%du + o]; int b = BI(%du + o); [[unroll]] for (uint d = 0u; d < 4u; d++) b += SD(c0[d], ww[d]);' % (o, B1, BB + 16) +
          ' r1[%d] |= (uint(clamp(RS(max(b, 0), %d), -128, 127)) & 0xffu) << %du; }' % (o // 4, Q['s1'], 8 * (o % 4)))
    for o in range(16):
        e('  { uint o = %du; int a = BI(%du + o); [[unroll]] for (uint q = 0u; q < 2u; q++) { uvec4 ww = wt[%du + 2u * o + q]; [[unroll]] for (uint d = 0u; d < 4u; d++) a += SD(r1[4u * q + d], ww[d]); }' % (o, BB + 48, B2) +
          ' int r = int(v[4][%d] << %du) >> 24; float vv = (float(a) * %s + float(r) * %s) * %s;' % (o // 4, 24 - 8 * (o % 4), f(-Q['e2']), f(-Q['ex']), f(Q['eo'])) +
          ' ow[%d] |= (uint(clamp(int(roundEven(vv)), -128, 127)) & 0xffu) << %du; }' % (o // 4, 8 * (o % 4)))
BODY = '\n'.join(L)
G = f'''#version 450
#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_integer_dot_product : require
layout(local_size_x = 64) in;
layout(set = 0, binding = 0) readonly buffer In {{ uint xin[]; }};
layout(set = 0, binding = 1, std140) uniform Wt {{ uvec4 wt[{len(words) // 4}]; }};
layout(set = 0, binding = 2) writeonly buffer Out {{ uint xout[]; }};
layout(push_constant) uniform Pc {{ uint W, H, inBase, outBase, rowW; }};
#define SD(a, b) dotPacked4x8EXT(int(a), int(b))   // signed x signed
int BI(uint i) {{ return int(wt[i >> 2u][i & 3u]); }}
int RS(int v, int s) {{ return (v + (1 << (s - 1)) - 1 + ((v >> s) & 1)) >> s; }}   // all int: HLSL turns int + uint arithmetic into uint (logical shift)
shared uvec4 T[3 * 66];
void main() {{
  uint xg = gl_WorkGroupID.x * 64u, y = gl_WorkGroupID.y, l = gl_LocalInvocationIndex;
{'' if not NOLDS else '/*'}  [[unroll]] for (uint r = 0u; r < 3u; r++) [[unroll]] for (uint h = 0u; h < 2u; h++) {{ uint px = l + 64u * h; if (px < 66u) {{ uint gx = min(xg + px, W + 1u);
    uint i = (inBase + (y + r) * rowW + gx * 16u) >> 2; T[r * 66u + px] = uvec4(xin[i], xin[i + 1u], xin[i + 2u], xin[i + 3u]); }} }}
{'*/' if NOLDS else ''}
  {'' if NOLDS else 'barrier();'}
  uint x = xg + l; if (x >= W) return;
  uvec4 v[9];
{'/*' if NOLDS else ''}  [[unroll]] for (uint t = 0u; t < 9u; t++) v[t] = T[(t / 3u) * 66u + l + t % 3u];{'*/' if NOLDS else ''}
{'' if NOLDS else '/*'}  [[unroll]] for (uint t = 0u; t < 9u; t++) {{ uint i = (inBase + (y + t / 3u) * rowW + (x + t % 3u) * 16u) >> 2; v[t] = uvec4(xin[i], xin[i + 1u], xin[i + 2u], xin[i + 3u]); }}{'' if NOLDS else '*/'}
  uint c0[4] = {{ 0u, 0u, 0u, 0u }}; uint r1[8] = {{ 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u }}; uint ow[4] = {{ 0u, 0u, 0u, 0u }};
{BODY}
  uint ob = (outBase + (y + 1u) * rowW + (x + 1u) * 16u) >> 2;
  [[unroll]] for (uint k = 0u; k < 4u; k++) xout[ob + k] = ow[k];
}}
'''
open(od + '/cnxi8.comp', 'w').write(G)
open(od + '/cnxi8w.bin', 'wb').write(np.array(words, np.uint32).tobytes())
rng = np.random.default_rng(5)
xpad = np.zeros((H + 2, W + 2, C), np.int8); xpad[1:-1, 1:-1] = rng.integers(-60, 60, (H, W, C))
open(od + '/cnxi8in.bin', 'wb').write(xpad.tobytes())
np.save(od + '/cnxi8ref.npy', forward_full(P, xpad, **Q))
print('ok', (W + 63) // 64, H, (W + 2) * 16 * (H + 2), 0, 'words', len(words))
