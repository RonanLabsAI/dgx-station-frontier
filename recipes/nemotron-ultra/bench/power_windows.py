#!/usr/bin/env python3
# Copyright 2026 RonanLabs. SPDX-License-Identifier: Apache-2.0
"""power_windows.py <benchdir> -- per bench window (windows.tsv) mean / max power.draw per GPU from power-left.csv
(rank-0 Station) and power-right.csv (second Station; copy it next to the left one first). Prints a TSV:
window, host, gpu, mean W, max W, samples."""
import csv
import sys
from datetime import datetime

D = sys.argv[1]
win = {}
for line in open(f'{D}/windows.tsv'):
    lab, ev, ts = line.rstrip('\n').split('\t')
    win.setdefault(lab, {})[ev] = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
print('window\thost\tgpu\tmean_W\tmax_W\tn')
for host in ('left', 'right'):
    try:
        rows = list(csv.reader(open(f'{D}/power-{host}.csv')))[1:]
    except FileNotFoundError:
        continue
    pts = []
    for r in rows:
        try:
            pts.append((datetime.strptime(r[0].strip()[:19], '%Y/%m/%d %H:%M:%S'), r[2].strip(), float(r[3].split()[0])))
        except (ValueError, IndexError):
            pass
    for lab, w in win.items():
        if 'start' not in w or 'end' not in w:
            continue
        for g in sorted({p[1] for p in pts}):
            v = [p[2] for p in pts if p[1] == g and w['start'] <= p[0] <= w['end']]
            if v:
                print(f"{lab}\t{host}\t{g}\t{sum(v)/len(v):.0f}\t{max(v):.0f}\t{len(v)}")
