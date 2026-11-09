# -*- coding: utf-8 -*-
"""生成 V3.1 修订所需的全部规范数字（单一事实来源）。

从运行级归档重算 S1/S2/S3/B2 的所有文稿数字，两种聚合口径并存：
  med_of_mean: 单元格(25 runs)均值 -> 跨 12 股中位数（现文稿 Table 3 口径）
  med_of_med : 单元格中位数       -> 跨 12 股中位数（稳健中位口径）
输出: output/v31_canonical_numbers.json
"""
import glob
import json

import numpy as np
import pandas as pd
from scipy import stats


STOCKS = [333, 2415, 300274, 300308, 300750, 600031, 600036,
          600276, 600519, 600887, 600900, 688981]


def norm_stock(s):
    return pd.to_numeric(s, errors='coerce').astype('Int64')


s1 = pd.concat([pd.read_csv(f) for f in glob.glob('output/s1v14_parts/*.csv')],
               ignore_index=True)
s1['stock'] = norm_stock(s1['stock'])

# 规范 Persistence 分母：必须取实验运行中与 LSTM 共用同一测试窗的实测记录值
# （run_experiment.py 中 prepare() 同一管线），而非用任意窗口重算。
pp = s1[s1['model'] == 'Persistence'][['stock', 'horizon', 'fold', 'RMSE']] \
    .rename(columns={'RMSE': 'pers'})


def ratio_of(df):
    df = df.copy()
    df['stock'] = norm_stock(df['stock'])
    df = df.merge(pp, on=['stock', 'horizon', 'fold'], how='left')
    assert df['pers'].notna().all(), 'persistence 配对缺失'
    df['ratio'] = df['RMSE'] / df['pers']
    return df


price = ratio_of(s1[s1['model'] == 'APSO-LSTM'])
adam = ratio_of(pd.concat([pd.read_csv(f) for f in glob.glob('output/adam_parts/*.csv')],
                          ignore_index=True))
s3 = ratio_of(pd.concat([pd.read_csv(f) for f in glob.glob('output/s3_parts/*.csv')],
                        ignore_index=True))
b1 = pd.read_csv('output/b1_cells.csv')
ar = b1[b1['feature'] == 'arima']

R = {}
g = price.groupby(['stock', 'horizon'])['ratio'].agg(['mean', 'median']).reset_index()
R['S1_curve_med_of_mean'] = g.groupby('horizon')['mean'].median().round(4).to_dict()
R['S1_curve_med_of_med'] = g.groupby('horizon')['median'].median().round(4).to_dict()
R['below1_40d_med'] = int((g[g['horizon'] == '+40d']['median'] < 1).sum())
R['below1_40d_mean'] = int((g[g['horizon'] == '+40d']['mean'] < 1).sum())
gg = g[g['horizon'] == '+40d'].set_index('stock')
R['closest_mean_40d'] = {'stock': int(gg['mean'].idxmin()),
                         'value': round(float(gg['mean'].min()), 4)}


def medof(df, agg):
    d = df[df['horizon'].isin(['+1d', '+5d', '+20d'])]
    t = d.groupby(['stock', 'horizon'])['ratio'].agg(agg).unstack()
    return {k: round(float(v), 4) for k, v in t.median().items()}


for nm, df in (('price', price), ('adam', adam),
               ('tech', s3[s3['model'] == 'APSO-LSTM-tech']),
               ('sent', s3[s3['model'] == 'APSO-LSTM-sent'])):
    R[f'T3_{nm}_medofmean'] = medof(df, 'mean')
    R[f'T3_{nm}_medofmed'] = medof(df, 'median')

ar2 = ar.set_index(['stock', 'horizon'])['ratio'].unstack()
R['T3_arima_stockmean'] = {h: round(float(ar2[h].mean()), 4)
                           for h in ['+1d', '+5d', '+20d']}
R['T3_arima_stockmedian'] = {h: round(float(ar2[h].median()), 4)
                             for h in ['+1d', '+5d', '+20d']}

base = price[price['horizon'].isin(['+1d', '+5d', '+20d'])] \
    .groupby(['stock', 'horizon'])['ratio'].mean()
for feat in ('tech', 'sent'):
    t = s3[s3['model'] == f'APSO-LSTM-{feat}'] \
        .groupby(['stock', 'horizon'])['ratio'].mean()
    w = (t < base.reindex(t.index)).groupby('horizon').sum()
    R[f'{feat}_beats_price_cells'] = {k: int(v) for k, v in w.items()}

# V3.2 复核 c2：推断单位统一为股（n=12）。每 (feature × horizon) 的
# 股级配对 Wilcoxon 族（2×3=6 项，Bonferroni 阈值 0.05/6≈0.0083）。
R['S3_family'] = {}
for feat in ('tech', 'sent'):
    for h in ('+1d', '+5d', '+20d'):
        a = s3[(s3['model'] == f'APSO-LSTM-{feat}') & (s3['horizon'] == h)] \
            .groupby('stock')['ratio'].median()
        b = price[price['horizon'] == h].groupby('stock')['ratio'].median()
        d = (a - b).dropna()
        R['S3_family'][f'{feat} vs price {h}'] = {
            'wins_feat': int((d < 0).sum()), 'n': int(len(d)),
            'median_ratio_feat': round(float(a.median()), 4),
            'p': round(float(stats.wilcoxon(d).pvalue), 6)}

s2 = pd.read_csv('output/s2_results.csv')
s2 = s2.dropna(subset=['model', 'horizon']).copy()
s2['stock'] = s2['stock'].astype(int)
s2 = ratio_of(s2)
fx = ratio_of(s1[s1['horizon'].isin(['+5d', '+10d'])
                 & (s1['model'] == 'APSO-LSTM')])
fx['model'] = 'Fixed-APSO'
allm = pd.concat([s2[['stock', 'horizon', 'fold', 'seed', 'model', 'ratio']],
                  fx[['stock', 'horizon', 'fold', 'seed', 'model', 'ratio']]],
                 ignore_index=True)
for agg in ('mean', 'median'):
    t = allm.groupby(['stock', 'horizon', 'model'])['ratio'].agg(agg) \
        .groupby(['horizon', 'model']).median().unstack()
    R[f'S2_med_of_{agg}'] = {h: {m: round(float(t.loc[h, m]), 4)
                                 for m in t.columns} for h in t.index}
cell = allm.groupby(['stock', 'horizon', 'model'])['ratio'].median().unstack()
# V3.2 复核 c2：推断单位统一为股（n=12/horizon），分 horizon 成族，
# 不再用 24 个 股×horizon 格做单次检验（伪重复）。
R['S2_pairs'] = {}
for a, b in [('Fixed-APSO', 'Staged-PSO'), ('Fixed-APSO', 'Unified-PSO'),
             ('Staged-PSO', 'Unified-PSO')]:
    R['S2_pairs'][f'{a} vs {b}'] = {}
    for h in ('+5d', '+10d'):
        ch = cell.xs(h, level='horizon')
        d = (ch[a] - ch[b]).dropna()
        R['S2_pairs'][f'{a} vs {b}'][h] = {
            'wins_a': int((d < 0).sum()), 'n': int(len(d)),
            'median_diff': round(float(d.median()), 4),
            'p': round(float(stats.wilcoxon(d).pvalue), 6)}
# 架构选择分布（V3.2 复核 c0 修复后，结构候选真实可达）
import re as _re
def _parse_arch(a):
    m = _re.match(r's(\d+),h(\d+)', str(a))
    return int(m.group(1)), int(m.group(2))
_s2raw = pd.read_csv('output/s2_results.csv')
_seqs, _hids = zip(*_s2raw['arch'].map(_parse_arch))
_s2raw = _s2raw.assign(seq=_seqs, hid=_hids)
R['S2_arch'] = {}
for _m in ('Unified-PSO', 'Staged-PSO'):
    _g = _s2raw[_s2raw['model'] == _m]
    R['S2_arch'][_m] = {
        'hidden_dist': {int(k): int(v) for k, v in
                        _g['hid'].value_counts().sort_index().items()},
        'seq_median': round(float(_g['seq'].median()), 1),
        'seq_min': int(_g['seq'].min()), 'seq_max': int(_g['seq'].max())}

json.dump(R, open('output/v31_canonical_numbers.json', 'w'), indent=1,
          ensure_ascii=False)
print(json.dumps(R, indent=1, ensure_ascii=False))
