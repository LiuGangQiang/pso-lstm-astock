# -*- coding: utf-8 -*-
"""E1 Block permutation test for dispersion comparison (Item 6 of the editing checklist / Review c10).

Design: The same seed shares the initialization and 600 subsampling streams between λ=0 and the maximum test λ,
The samples are paired rather than independent. For the four panels (Hengrui/China Merchants Bank × full sample/400-day window),
with seed as the block, exchange the λ labels within the block, and recalculate the Levene statistic consistent with the main text
(absolute deviation of the mean, two groups) to obtain the calibrated permutation p-value.

Null hypothesis: The two λ labels within the block are interchangeable (stronger than the conservative null hypothesis of "equal dispersion only").
The statistic is consistent with the nominal Levene in the main text, making it easy to directly replace the criteria.
B = 200,000 Monte Carlo permutations, rng seed 0."""
import numpy as np
import pandas as pd

B = 200_000
RNG = np.random.default_rng(0)

full = pd.read_csv('output/s5_e1_parts/s5_e1.csv')
win = pd.read_csv('output/s5_e1_parts/s5_e1_win400.csv')

HR, CMB = 600276, 600036


def levene_W(x, y):
    """Two-group Brown–Forsythe statistic (deviations from group medians),
    Consistent with scipy levene(center='median') (nominal p-value standard in the main text)."""
    zx = np.abs(x - np.median(x))
    zy = np.abs(y - np.median(y))
    n1, n2 = len(zx), len(zy)
    zbar1, zbar2 = zx.mean(), zy.mean()
    zbar = (zx.sum() + zy.sum()) / (n1 + n2)
    ssb = n1 * (zbar1 - zbar) ** 2 + n2 * (zbar2 - zbar) ** 2
    ssw = ((zx - zbar1) ** 2).sum() + ((zy - zbar2) ** 2).sum()
    dfb, dfw = 1, n1 + n2 - 2
    return (ssb / dfb) / (ssw / dfw)


def block_permutation(a, b, B=B, rng=RNG):
    """Block-wise permutation: Each seed pair exchanges labels with a probability of 1/2, and Levene's W is recalculated."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    W_obs = levene_W(a, b)
    n = len(a)
    cnt = 0
    for _ in range(B):
        swap = rng.random(n) < 0.5
        pa = np.where(swap, b, a)
        pb = np.where(swap, a, b)
        if levene_W(pa, pb) >= W_obs:
            cnt += 1
    p = (cnt + 1) / (B + 1)
    return W_obs, p


def panel(df, stock, lams, label, win_size=None):
    d = df[df.stock == stock]
    if win_size is not None:
        d = d[d.win == win_size]
    d0 = d[d.lam == lams[0]].set_index('seed')['ratio']
    d1 = d[d.lam == lams[1]].set_index('seed')['ratio']
    common = d0.index.intersection(d1.index)
    a = d0.loc[common].values
    b = d1.loc[common].values
    W_obs, p_perm = block_permutation(a, b)
    # Nominal (treating paired samples as independent groups) p-value for comparison
    from scipy import stats
    W_nom, p_nom = stats.levene(a, b, center='median')
    sd0, sd1 = a.std(ddof=1), b.std(ddof=1)
    mean_diff = b.mean() - a.mean()
    print(f'{label}: n={len(common)} seeds, λ {lams[0]} vs {lams[1]}')
    print(f'  SD {sd0:.4f} -> {sd1:.4f} (Δmean {mean_diff:+.4f})')
    print(f'  nominal Levene W={W_nom:.4f} p={p_nom:.4f} | '
          f'block-permutation p={p_perm:.4f} (B={B})')
    return dict(panel=label, n=len(common), lam_low=lams[0], lam_high=lams[1],
                sd_low=sd0, sd_high=sd1, mean_diff=mean_diff,
                W_nominal=W_nom, p_nominal=p_nom, p_blockperm=p_perm, B=B)


rows = [
    panel(full, HR, (0.0, 0.5), 'Hengrui full-length (λ=0 vs 0.5)'),
    panel(win, HR, (0.0, 0.1), 'Hengrui 400-day (λ=0 vs 0.1)', win_size=400),
    panel(full, CMB, (0.0, 0.5), 'CMB full-length (λ=0 vs 0.5)'),
    panel(win, CMB, (0.0, 0.5), 'CMB 400-day (λ=0 vs 0.5)', win_size=400),
]

out = pd.DataFrame(rows)
out.to_csv('output/s5_e1_blockperm.csv', index=False)
print()
print(out[['panel', 'p_nominal', 'p_blockperm']].to_string(index=False))
print('saved -> output/s5_e1_blockperm.csv')
