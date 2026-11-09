# -*- coding: utf-8 -*-
"""对照实验结果汇总分析。"""
import pandas as pd

df = pd.read_csv('output/s5_e3_parts/s5_e3_controls.csv')
tmp = pd.read_csv('output/s5_e3_controls_tmp.csv')
df = pd.concat([df, tmp], ignore_index=True)
df = df.drop_duplicates(subset=['control', 'r', 'lam', 'seed'], keep='first')
df.to_csv('output/s5_e3_parts/s5_e3_controls.csv', index=False)
print('total rows:', len(df))
print()

main = pd.read_csv('output/s5_e3_parts/s5_e3.csv')
for name, g in df.groupby('control'):
    print(f'== {name} (n={len(g)}) ==')
    print('  chosen_seq median=%.0f mean=%.1f | in[35,45]: %d/%d | ==40: %d' % (
        g.chosen_seq.median(), g.chosen_seq.mean(),
        ((g.chosen_seq >= 35) & (g.chosen_seq <= 45)).sum(), len(g), (g.chosen_seq == 40).sum()))
    print('  chosen_hidden counts:', dict(g.chosen_hidden.value_counts().sort_index()))
    print('  init_seq_med median=%s  init_hid_med median=%s' % (g.init_seq_med.median(), g.init_hid_med.median()))
print()

print('== 主实验 std-init 对照 (lam=0.1) ==')
m = main[main.lam == 0.1]
for r in [0.2, 0.6, 0.95]:
    g = m[m.r == r]
    print('r=%.2f: seq median=%.0f mean=%.1f | in[35,45]: %d/%d | ==40: %d | hidden %s' % (
        r, g.chosen_seq.median(), g.chosen_seq.mean(),
        ((g.chosen_seq >= 35) & (g.chosen_seq <= 45)).sum(), len(g), (g.chosen_seq == 40).sum(),
        dict(g.chosen_hidden.value_counts().sort_index())))
