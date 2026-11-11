# Copyright 2026 lonewolf0622
# SPDX-License-Identifier: GPL-2.0-or-later
"""plan.py FAMILY: the weight-table jobs of one layout family (passes 3-11), one line each: kind shader args... output;
plan.py FAMILY blob: the family's weight blob offset in AMD's DLL (Ultra Performance: its own InitializerBuffer)"""
import json, os, sys
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
f = json.load(open(R + '/data/families.json'))[sys.argv[1]]
if sys.argv[2:] == ['blob']: print(f['blob']); sys.exit()
for p, v in f['passes'].items():
    out = '%s/work/%s/%s.pkl' % (R, f['pkldir'], p)
    if v['kind'] == 'ds': print('ds', v['shader'], v['C'] // 2, 2 * v['W'], 2 * v['H'], '-', out)
    elif v['kind'] == 'ct': print('ct', v['shader'], v['C'], v['C'] // 2, v['W'], v['H'], out)
    elif v['kind'] == 'fnb': print('fnb', v['shader'], v['C'], v['W'], v['H'], '-', out)
