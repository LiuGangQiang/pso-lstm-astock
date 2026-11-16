# -*- coding: utf-8 -*-
"""M4 / S4: Statistical test of volatility groupings (Proposal Document V1.1, Chapter 5, Section 5.4, H4).

Question: Does the advantage of APSO-LSTM over Persistence (ratio RMSE_APSO / RMSE_Persist)
Does it change systematically with the volatility level (low/medium/high)? (H4: There are significant differences between the levels;
The initial hypothesis was that "the learning model is relatively superior at the high volatility level" – in S1, +1d is weakly supported, and at +20d, the three levels coincide.)

Data:
- Price three-level features share the APSO-LSTM and Persistence results of s1v14
  (output/s1v14_parts/*.csv, containing h{1,3,5,10,20,40}, taking {1,5,20} to align with S3);
- The Persistence of tech/sent is the same as that of price (Persistence only uses the closing price),
  Therefore, the three-layer comparison of the sub-features directly uses s3_parts + s1v14 Persistence.

Statistical design:
- Unit = (target, horizon, feature set); each unit is first run 25 times with 5-fold × 5 seeds.The mean of the "same fold, same seed" ratio is taken (aggregation at the ratio level, maintaining pairing);
- Inter-level comparison: Kruskal-Wallis H test (3 levels × 4 stocks per tier, non-normal small sample);
  Post-hoc pairwise Mann-Whitney U (Bonferroni correction ×3);
- "Whether it is superior to Persistence" within the hierarchy: Perform a one-sample Wilcoxon test on the 25 paired odds ratios (median vs. 1);
- Effect size: median odds ratio across levels, proportion of odds ratios < 1;
- A50 sub-sample robustness: Of the 12 stocks, those belonging to the CSI A50 (see A50 constants for the list,
  The above tests are rerun separately, based on the list after the December 2024 rebalancing).

Output: output/s4_group_test.csv (Summary table of test results)."""
import glob
import os

import numpy as np
import pandas as pd
from scipy import stats

TIERS = ['low_volatility', 'medium_volatility', 'high_volatility']
HORIZONS = ["+1d", "+5d", "+20d"]
FEATURES = ["price", "tech", "sent"]

# 12 stocks that are part of the CSI A50 (2024-12 post-rebalancing list; non-A50 stocks are not listed)
A50 = {"600900", "600036", "600519", "600887", "000333", "600276",
       "688981", "300750", "300308"}
# Note: Sany Heavy Industry (600031), Hikvision (002415), and Sungrow Power Supply (300274) are not on the list for this period.


def load_price_rows():
    """APSO-LSTM and Persistence (h{1,5,20}) for s1v14."""
    frames = []
    for p in glob.glob(os.path.join("output", "s1v14_parts", "*.csv")):
        df = pd.read_csv(p, dtype={"stock": str})
        frames.append(df[df.horizon.isin(HORIZONS)])
    return pd.concat(frames, ignore_index=True)


def load_feature_rows():
    """APSO-LSTM-tech / -sent for s3."""
    frames = []
    for p in glob.glob(os.path.join("output", "s3_parts", "*.csv")):
        frames.append(pd.read_csv(p, dtype={"stock": str}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def paired_ratios(apso: pd.DataFrame, persist: pd.DataFrame, feature: str):
    """Compute paired APSO/Persistence RMSE ratios by (stock, horizon, fold, seed),
    then average the 25 ratios for each (stock, horizon) and return a long-format table.

    Persistence is seed-independent (run_m1_baseline records it only at seed 0),
    so broadcast it across the seeds present for APSO to pair all 5 folds × 5 seeds."""
    seeds = sorted(apso.seed.unique())
    persist = persist[persist.model == "Persistence"]
    frames = []
    for s in seeds:
        p = persist.copy()
        p["seed"] = s
        frames.append(p)
    persist = pd.concat(frames, ignore_index=True)
    m = apso.merge(persist, on=["stock", "horizon", "fold", "seed"],
                   suffixes=("_a", "_p"))
    m = m[m.model_a.str.startswith("APSO-LSTM")]
    m["ratio"] = m.RMSE_a / m.RMSE_p
    g = m.groupby(["stock", "name_a", "tier_a", "horizon"]).ratio.mean().reset_index()
    g = g.rename(columns={"name_a": "name", "tier_a": "tier"})
    g["feature"] = feature
    return g


def kruskal_and_posthoc(groups: dict):
    """groups: {tier: [ratios]} → (KW H, p), and pairwise MWU (Bonferroni)."""
    samples = [np.asarray(groups[t]) for t in TIERS if len(groups.get(t, []))]
    if len(samples) < 2:
        return None, []
    H, p_kw = stats.kruskal(*samples)
    post = []
    for i in range(len(TIERS)):
        for j in range(i + 1, len(TIERS)):
            a, b = np.asarray(groups.get(TIERS[i], [])), np.asarray(groups.get(TIERS[j], []))
            if len(a) and len(b):
                u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
                post.append((f"{TIERS[i]} vs {TIERS[j]}", u, min(p * 3, 1.0)))
    return (H, p_kw), post


def wilcoxon_vs_1(ratios):
    r = np.asarray(ratios)
    r = r[r != 1]
    if len(r) < 6:
        return None, None
    try:
        w, p = stats.wilcoxon(r - 1)
    except ValueError:
        return None, None
    return w, p


def run_tests(cell: pd.DataFrame, label: str, rows: list):
    """Cell: The (stock, tier, ratio) table of a certain (feature, horizon, sub-sample)."""
    groups = {t: cell[cell.tier == t].ratio.values for t in TIERS}
    (H, p_kw), post = kruskal_and_posthoc(groups)
    med = {t: (float(np.median(groups[t])) if len(groups[t]) else np.nan) for t in TIERS}
    rows.append(dict(analysis=label, test='Kruskal-Wallis (Hierarchical)', H=round(H, 4) if H else None,
                     p=p_kw, low_volatility_median=med['low_volatility'], medium_volatility_median=med['medium_volatility'],
                     high_volatility_median=med['high_volatility'], details="; ".join(
                         f"{t}: n={len(groups[t])}, median={med[t]:.3f}" if len(groups[t]) else f"{t}: empty"
                         for t in TIERS)))
    for t in TIERS:
        w, p = wilcoxon_vs_1(groups[t])
        rows.append(dict(analysis=label, test=f"Wilcoxon (median=1, {t})", W=w, p=p,
                         low_volatility_median=med['low_volatility'], medium_volatility_median=med['medium_volatility'], high_volatility_median=med['high_volatility']))
    for name, u, p in post:
        rows.append(dict(analysis=label, test=f"MWU Bonferroni({name})", U=u, p=p,
                         low_volatility_median=med['low_volatility'], medium_volatility_median=med['medium_volatility'], high_volatility_median=med['high_volatility']))


def main():
    price = load_price_rows()
    feat = load_feature_rows()
    persist = price[price.model == "Persistence"]

    cells = []
    cells.append(paired_ratios(price[price.model == "APSO-LSTM"], persist, "price"))
    for fset in ("tech", "sent"):
        if len(feat):
            cells.append(paired_ratios(feat[feat.model == f"APSO-LSTM-{fset}"], persist, fset))
    allcells = pd.concat(cells, ignore_index=True)
    allcells.to_csv(os.path.join("output", "s4_cells.csv"), index=False,
                    encoding="utf-8-sig")

    rows = []
    for feature in FEATURES:
        for h in HORIZONS:
            cell = allcells[(allcells.feature == feature) & (allcells.horizon == h)]
            if not len(cell):
                continue
            run_tests(cell, f"{feature} {h} full sample", rows)
            a50cell = cell[cell.stock.isin(A50)]
            if len(a50cell) >= 6:
                run_tests(a50cell, f"{feature} {h} A50 subsample", rows)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join("output", "s4_group_test.csv"), index=False,
               encoding="utf-8-sig")
    print(f"Cells: {len(allcells)}; test rows: {len(out)}")
    print(out[out.test.str.startswith("Kruskal")].to_string(index=False))


if __name__ == "__main__":
    main()
