# Copyright 2026 lonewolf0622
# SPDX-License-Identifier: GPL-2.0-or-later
"""derive4k.py FAMILY_4K FAMILY_1080: the 4K layout family's weight tables are the 1080p family's (same weights; the 4K shaders only
lay the tensors out wider) with the 4K sizes"""
import json, os, pickle, sys
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = json.load(open(R + '/data/families.json')); big, small = F[sys.argv[1]], F[sys.argv[2]]
os.makedirs('%s/work/%s' % (R, big['pkldir']), exist_ok=True)
for p, v in big['passes'].items():
    if v['kind'] == 'cnx': continue
    d = pickle.load(open('%s/work/%s/%s.pkl' % (R, small['pkldir'], p), 'rb'))
    if v['kind'] == 'ds': d['IW'], d['IH'] = 2 * v['W'], 2 * v['H']
    else: d['W'], d['H'] = v['W'], v['H']
    pickle.dump(d, open('%s/work/%s/%s.pkl' % (R, big['pkldir'], p), 'wb'))
print(sys.argv[1], 'from', sys.argv[2])
