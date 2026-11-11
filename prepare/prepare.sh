#!/bin/bash
# Copyright 2026 lonewolf0622
# SPDX-License-Identifier: GPL-2.0-or-later
# prepare.sh : everything the kernel build reads, from AMD's DLL: work/sh + work/dis (extract.py), the pass offsets and the weight
# tables of passes 3-11 for every layout family. AMD_FSR_DLL = amd_fidelityfx_upscaler_dx12.dll (FidelityFX SDK 2.4.0), DXC = dxc.
set -e
R=$(cd "$(dirname "$0")/.." && pwd); export HLX_ROOT=$R
: "${AMD_FSR_DLL:?set AMD_FSR_DLL to amd_fidelityfx_upscaler_dx12.dll from the AMD FidelityFX SDK 2.4.0}"; export AMD_FSR_DLL
python3 $R/prepare/extract.py "$AMD_FSR_DLL" $R/work
cp "$AMD_FSR_DLL" $R/work/amd_fidelityfx_upscaler_dx12.dll   # the weights are read from it (ref/convnext.py)
cd $R/ref
for fam in std1080 std4k up1080 up4k; do python3 offsets.py $fam > /dev/null; done
for fam in std1080 up1080; do
  export BLOB=$(python3 $R/prepare/plan.py $fam blob)
  python3 $R/prepare/plan.py $fam | while read -r kind k a b c d out; do
    mkdir -p "$(dirname "$out")"
    case $kind in
      ds) python3 prep_ds.py $k $a $b $c "$out" > /dev/null;;
      ct) python3 prep_ct.py $k $a $b $c $d "$out" > /dev/null;;
      fnb) python3 prep_fnb.py $k $a $b $c "$out" > /dev/null;;
    esac
    echo "$fam $(basename "$out" .pkl)"
  done
done
python3 $R/prepare/derive4k.py std4k std1080
python3 $R/prepare/derive4k.py up4k up1080
