#!/bin/bash
# Copyright 2026 lonewolf0622
# SPDX-License-Identifier: GPL-2.0-or-later
# build.sh OUTDIR : HLX FSR 4 Full's INT8 kernels for AMD RDNA 2 / RDNA 3 (and other GPUs with int8 dot products), as DXIL:
#   i8_<family><pass>.dxil + i8_<family><pass>w.bin  the 12 network passes for the 1080p and 4K layout families, standard and
#                                                    Ultra Performance models; i8repl.txt maps each to AMD's shader (sha256) it replaces
#   post/<AMD sha>.dxil                              the postpass, every variant AMD's runtime uses
#   pre/<AMD sha>.dxil                               the prepass, every variant without non-linear colour
# Each is bit-exact with the AMD shader it replaces. Run prepare/prepare.sh first. DXC = dxc 1.9 (default: dxc on PATH).
# PROXY=1: timing forms only (int8 dot replaced by xor + add: wrong numbers, for timing on GPUs without int8 dot products).
set -e
O=$(realpath -m "$1"); [ -n "$1" ] || { echo "usage: build.sh OUTDIR"; exit 1; }
R=$(cd "$(dirname "$0")" && pwd); export HLX_ROOT=$R; W=$R/work; DXC=${DXC:-dxc}
[ -d $W/passes ] || { echo "run prepare/prepare.sh first"; exit 1; }
mkdir -p $O $O/post $O/pre $R/vk/k; cd $R/vk
: > $O/i8repl.txt
for fam in ":offsets.json:passes:1920:0:" "m_:offsets_m.json:passes_m:3840:0:" "u_:offsets_up.json:passes_up:1920:1:0x1658460" "um_:offsets_upm.json:passes_upm:3840:1:0x1658460"; do
  IFS=: read pf offs pk famw up blob <<< "$fam"
  for p in p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11 p12; do
    read kind k Wd Hd ib ob sb <<< $(python3 -c "import json;o=json.load(open('$W/passes/$offs'))['$p'];print(o['kind'],o.get('wshader',o['shader']),o['W'],o['H'],o['inBase'],o['outBase'],o['skipBase'] or 0)")
    sk=$(python3 -c "import json;print(json.load(open('$W/passes/$offs'))['$p']['shader'])")
    case $kind in
      cnx) BLOB=${blob:-0} python3 gen_cnxi8.py $k k > /dev/null; g=cnxi8; gx=$(( (Wd + 63) / 64 )); push="$Wd $Hd $ib $ob $(( (Wd + 2) * 16 ))";;
      ds)  python3 gen_dsi8.py $W/$pk/$p.pkl k > /dev/null; g=dsi8; gx=$(( (Wd + 31) / 32 )); push="$Wd $Hd $ib $ob";;
      *)   C=$(python3 -c "import pickle;print(pickle.load(open('$W/$pk/$p.pkl','rb'))['C'])")
           if [ $C = 32 ]; then python3 gen_fpi8.py $W/$pk/$p.pkl k > /dev/null; g=fpi8; gx=$(( (Wd + 63) / 64 ))
           else python3 gen_fni8.py $W/$pk/$p.pkl k > /dev/null; g=fni8; gx=$(( (Wd + 31) / 32 )); fi
           push="$Wd $Hd $ib $ob 0 $sb";;
    esac
    python3 glsl2hlsl.py k/$g.comp k/i8dx.hlsl > /dev/null
    [ -n "$PROXY" ] && { sed -i 's/{ return dot4add_i8packed(uint(a), uint(b), 0); }/{ return a ^ b; }/' k/i8dx.hlsl; grep -q "return a ^ b" k/i8dx.hlsl; }
    $DXC -T cs_6_6 -E main -HV 2021 -enable-16bit-types -O3 k/i8dx.hlsl -Fo $O/i8_$pf$p.dxil > /dev/null
    cp k/${g}w.bin $O/i8_$pf${p}w.bin
    sha=$(sha256sum $W/sh/$(printf %04d $sk).dxil | cut -c1-64)
    echo "$sha i8_$pf$p.dxil i8_$pf${p}w.bin $gx $Hd 1 $push F=$famw U=$up" >> $O/i8repl.txt
  done
done
# postpass: weights by model (standard: AMD's shader 411, Ultra Performance: 503 with its initializer blob)
grep '^post ' $R/data/variants.txt | while read -r _ sha fam model inv ae dmv jit nl; do
  row=$(case $fam in S) echo 15392;; M) echo 30752;; L) echo 61472;; esac)
  if [ $model = 0 ]; then psh=411; blob=0; else psh=503; blob=0x1658460; fi
  ROW=$row POSTSH=$psh BLOB=$blob AE=$ae NL=$nl python3 gen_posti8.py k/post.hlsl > /dev/null
  [ -n "$PROXY" ] && { sed -i 's/{ return dot4add_i8packed(a, b, 0); }/{ return int(a ^ b); }/' k/post.hlsl; grep -q "return int(a ^ b)" k/post.hlsl; }
  $DXC -T cs_6_6 -E main -HV 2021 -enable-16bit-types -O3 k/post.hlsl -Fo $O/post/$sha.dxil > /dev/null
done
# prepass: 16 x 8 pixel groups (the runtime dispatches twice AMD's group rows); variants with non-linear colour keep AMD's
grep '^pre ' $R/data/variants.txt | while read -r _ sha fam model inv ae dmv jit nl; do
  [ "$nl" = 0 ] || continue
  row=$(case $fam in S) echo 15392;; M) echo 30752;; L) echo 61472;; esac)
  (cd ../pre && ROW=$row INV=$inv AE=$ae DMV=$dmv JIT=$jit python3 genpre_px.py ../vk/k/pre.hlsl > /dev/null)
  $DXC -T cs_6_6 -E main -HV 2021 -O3 -enable-16bit-types k/pre.hlsl -Fo $O/pre/$sha.dxil > /dev/null
done
echo "$(wc -l < $O/i8repl.txt) network kernels, $(ls $O/post | wc -l) postpass, $(ls $O/pre | wc -l) prepass -> $O"
