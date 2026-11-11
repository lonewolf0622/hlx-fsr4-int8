# Copyright 2026 lonewolf0622
# SPDX-License-Identifier: GPL-2.0-or-later
"""gen_dsi8.py PASS.pkl OUTDIR: 2x2 stride-2 downscale conv C -> 2C (passes 3 / 6) on int8 dot products, AMD's numbers exactly
(int32 sum, round(acc * 2^-ex) per 16-channel group). Workgroup = 32 output px of a row, 4 waves; wave w = output channels
(O/4) w .. (wave-uniform weights), one px per lane; the input patches staged once in workgroup memory, packed int8 throughout.
Writes dsi8.comp, dsi8w.bin, dsi8in.bin, dsi8ref.npy."""
import sys, pickle, struct, numpy as np
D = pickle.load(open(sys.argv[1], 'rb')); od = sys.argv[2]
Wd, b, ex, C, IW, IH = D['Wd'], D['b'], D['ex'], D['C'], D['IW'], D['IH']
O, OW, OH = 2 * C, IW // 2, IH // 2; CD, NO = C // 4, O // 4
pk = lambda v: struct.unpack('<I', np.asarray(v, np.int8).tobytes())[0]
words = []
for o in range(O):                         # W[o]: taps (ky, kx) x C inputs = C dwords = C/4 uvec4
    for t in range(4): words += [pk(Wd[o, t >> 1, t & 1, 4 * i:4 * i + 4]) for i in range(C // 4)]
BB = len(words); words += [int(v) & 0xFFFFFFFF for v in b]
while len(words) % 4: words.append(0)
assert len(words) * 4 <= 65536
IROW, IPL = (IW + 2) * 16, (IW + 2) * 16 * (IH + 2); OROW, OPL = (OW + 2) * 16, (OW + 2) * 16 * (OH + 2)
exs = sorted(set(ex)); assert len(ex) == O // 16
G = f'''#version 450
#extension GL_EXT_control_flow_attributes : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_EXT_integer_dot_product : require
layout(local_size_x = 128) in;
layout(set = 0, binding = 0) readonly buffer In {{ uint xin[]; }};
layout(set = 0, binding = 1, std140) uniform Wt {{ uvec4 wt[{len(words) // 4}]; }};
layout(set = 0, binding = 2) writeonly buffer Out {{ uint xout[]; }};
layout(push_constant) uniform Pc {{ uint W, H, inBase, outBase; }};
#define SD(a, b) dotPacked4x8EXT(int(a), int(b))   // signed x signed
int BI(uint i) {{ return int(wt[i >> 2u][i & 3u]); }}
shared uint P4[32 * {4 * CD + 1}];   // per output px: taps (0,0) (0,1) (1,0) (1,1) x {C} channels
void main() {{
  uint xg = gl_WorkGroupID.x * 32u, y = gl_WorkGroupID.y, l = gl_LocalInvocationIndex;
  for (uint e = l; e < 32u * 4u * {C // 16}u; e += 128u) {{   // one 16-byte load per (px, tap, plane)
    uint px = e & 31u, t = (e >> 5) & 3u, pl = e >> 7, x = min(xg + px, W - 1u);
    uint i = (inBase + pl * {IPL}u + (2u * y + (t >> 1) + 1u) * {IROW}u + (2u * x + (t & 1u) + 1u) * 16u) >> 2;
    [[unroll]] for (uint d = 0u; d < 4u; d++) P4[px * {4 * CD + 1}u + t * {CD}u + pl * 4u + d] = xin[i + d]; }}
  barrier();
  uint w = gl_SubgroupID, px = l & 31u, x = xg + px;
  uint v[{4 * CD}]; [[unroll]] for (uint d = 0u; d < {4 * CD}u; d++) v[d] = P4[px * {4 * CD + 1}u + d];
  if (x >= W) return;
  [[unroll]] for (uint k = 0u; k < {NO // 4}u; k++) {{ uint word = 0u;
    [[unroll]] for (uint j = 0u; j < 4u; j++) {{ uint o = {NO}u * w + 4u * k + j; int a = BI({BB}u + o);
      [[unroll]] for (uint q = 0u; q < {CD}u; q++) {{ uvec4 ww = wt[o * {CD}u + q]; [[unroll]] for (uint e = 0u; e < 4u; e++) a += SD(v[4u * q + e], ww[e]); }}
      float s = {' : '.join(f'(o >> 4u) == {g}u ? {float(2.0 ** -e)!r}' for g, e in enumerate(ex[:-1])) + (' : ' if len(ex) > 1 else '') + repr(float(2.0 ** -ex[-1]))};
      word |= (uint(clamp(int(roundEven(float(a) * s)), -128, 127)) & 0xffu) << (8u * j); }}
    uint dw = {NO // 4}u * w + k;
    xout[((outBase + (dw >> 2u) * {OPL}u + (y + 1u) * {OROW}u + (x + 1u) * 16u) >> 2) + (dw & 3u)] = word; }}
}}
'''
open(od + '/dsi8.comp', 'w').write(G)
open(od + '/dsi8w.bin', 'wb').write(np.array(words, np.uint32).tobytes())
rng = np.random.default_rng(4)
xpad = np.zeros((IH + 2, IW + 2, C), np.int8); xpad[1:-1, 1:-1] = rng.integers(-60, 60, (IH, IW, C))
open(od + '/dsi8in.bin', 'wb').write(b''.join(np.ascontiguousarray(xpad[:, :, p:p + 16]).tobytes() for p in range(0, C, 16)))
x = xpad[1:-1, 1:-1].astype(np.int64)
acc = np.broadcast_to(b, (OH, OW, O)).copy()
for ky in range(2):
    for kx in range(2): acc += x[ky::2, kx::2][:OH, :OW] @ Wd[:, ky, kx, :].T
sc = np.repeat([2.0 ** -e for e in ex], 16)
ref = np.clip(np.round(acc.astype(np.float32) * sc.astype(np.float32)), -128, 127).astype(np.int8)
np.save(od + '/dsi8ref.npy', ref)
print('ok', (OW + 31) // 32, OH, OPL * (O // 16), 0, 'words', len(words))
