# Copyright 2026 lonewolf0622
# SPDX-License-Identifier: GPL-2.0-or-later
"""extract.py AMD_DLL WORK: AMD's FSR 4.1.2 shaders (DXBC containers, in file order) -> WORK/sh/NNNN.dxil, and their disassembly
(dxc -dumpbin) -> WORK/dis/NNNN.dxil.txt. AMD_DLL: amd_fidelityfx_upscaler_dx12.dll from AMD's FidelityFX SDK 2.4.0
(Kits/FidelityFX/signedbin, FSR 4.1.2.4555, sha256 aad4fccc...). DXC: the dxc binary (environment)."""
import hashlib, os, struct, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
dll, work = sys.argv[1], sys.argv[2]
d = open(dll, 'rb').read()
if hashlib.sha256(d).hexdigest()[:16] != 'aad4fccccb29084a':
    sys.exit('not the FidelityFX SDK 2.4.0 amd_fidelityfx_upscaler_dx12.dll (FSR 4.1.2.4555)')
os.makedirs(work + '/sh', exist_ok=True); os.makedirs(work + '/dis', exist_ok=True)
i, n = d.find(b'DXBC'), 0
while i >= 0:
    size = struct.unpack_from('<I', d, i + 24)[0]
    open('%s/sh/%04d.dxil' % (work, n), 'wb').write(d[i:i + size]); n += 1
    i = d.find(b'DXBC', i + size)
dxc = os.environ.get('DXC', 'dxc')
def dis(k):
    t = subprocess.run([dxc, '-dumpbin', '%s/sh/%04d.dxil' % (work, k)], capture_output=True, text=True).stdout
    open('%s/dis/%04d.dxil.txt' % (work, k), 'w').write(t)
with ThreadPoolExecutor(os.cpu_count() or 4) as pool:
    list(pool.map(dis, range(n)))
print(n, 'shaders')
