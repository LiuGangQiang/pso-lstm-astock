# -*- coding: utf-8 -*-
"""Figures for manuscript V2 with English labels (matplotlib has no CJK glyphs in this environment)."""
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = "output/paper_figs"
import os
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

# ---- Fig 1: S1 ratio curve (corrected pairing) ----
g = pd.read_csv("output/s1_cells_corrected.csv", dtype={"stock": str})
order = ["+1d", "+3d", "+5d", "+10d", "+20d", "+40d"]
piv = g.pivot_table(index="stock", columns="horizon", values="ratio")[order]
x = np.arange(len(order))
fig, ax = plt.subplots(figsize=(7.2, 4.2))
for stock, row in piv.iterrows():
    ax.plot(x, row.values, color="0.75", lw=1, zorder=1)
med = piv.median()
ax.plot(x, med.values, color="#1F4E79", lw=2.6, marker="o", zorder=3, label="Median across 12 stocks")
ax.axhline(1.0, color="#C00000", ls="--", lw=1.4, zorder=2)
ax.text(0.05, 1.04, "Parity with Persistence", color="#C00000", fontsize=9)
ax.set_xticks(x, order)
ax.set_xlabel("Forecast horizon")
ax.set_ylabel("RMSE ratio vs Persistence")
ax.set_ylim(0.8, 4.4)
ax.legend(frameon=False, loc="upper right")
ax.set_title("S1: median RMSE ratio vs Persistence by horizon")
fig.tight_layout()
fig.savefig(f"{OUT}/fig1_ratio_curve.png", dpi=200)
plt.close(fig)

# ---- Fig 2: feature ablation (S3) ----
# V3.2 review c14/c23: aggregate as the mean over fold×seed within each stock, then the median across stocks;
# Consistent with Table 3 in the main text; price comes from s1, tech/sent comes from s3.
raw = pd.concat([pd.read_csv(p, dtype={"stock": str}) for p in glob.glob("output/s1v14_parts/*.csv")], ignore_index=True)
s3raw = pd.concat([pd.read_csv(p, dtype={"stock": str}) for p in glob.glob("output/s3_parts/*.csv")], ignore_index=True)
for _d in (raw, s3raw):
    _d["stock"] = _d["stock"].str.zfill(6)
    _d["horizon"] = _d["horizon"].astype(str)
pe = raw[(raw.model == "Persistence")]
pb = pd.concat([pe.assign(seed=s) for s in range(5)], ignore_index=True)


def stock_med_ratio(df, agg="mean"):
    """Aggregate fold×seed values within each stock, then take the median across stocks; return a series indexed by horizon."""
    m = df.merge(pb[["stock", "horizon", "fold", "seed", "RMSE"]],
                 on=["stock", "horizon", "fold", "seed"], suffixes=("", "_per"))
    m["r"] = m.RMSE / m.RMSE_per
    per_stock = m.groupby(["stock", "horizon"]).r.agg(agg)
    return per_stock.groupby("horizon").median()


feats = [("Price only (1 feat)", stock_med_ratio(raw[(raw.model == "APSO-LSTM")]).reindex(["+1d", "+5d", "+20d"]), "#1F4E79"),
         ("Tech (+4 feats)", stock_med_ratio(s3raw[s3raw.model == "APSO-LSTM-tech"]).reindex(["+1d", "+5d", "+20d"]), "#2E75B6"),
         ("Sent (+3 feats)", stock_med_ratio(s3raw[s3raw.model == "APSO-LSTM-sent"]).reindex(["+1d", "+5d", "+20d"]), "#9DC3E6")]
fig, ax = plt.subplots(figsize=(7.2, 4.0))
w = 0.26
x = np.arange(3)
for i, (name, s, c) in enumerate(feats):
    ax.bar(x + (i - 1) * w, s.values, width=w, color=c, label=name)
ax.set_xticks(x, ["+1d", "+5d", "+20d"])
ax.set_xlabel("Forecast horizon")
ax.set_ylabel("Median RMSE ratio vs Persistence")
ax.axhline(1.0, color="#C00000", ls="--", lw=1.2)
ax.legend(frameon=False)
# V3.5 (Review c20): The original title has the opposite literal meaning – the feature has never improved accuracy (0/12 stocks, average caliber).
ax.set_title("Feature extensions improve accuracy in 0/12 stocks (S3)",
             fontsize=12.5)
fig.tight_layout()
fig.savefig(f"{OUT}/fig2_features.png", dpi=200)
plt.close(fig)

# ---- Fig 3: encoding comparison (S2) ----
# V3.2 Review c0/c14: s2_results.csv is the rerun result after decoding correction;
# Aggregate as the median over fold×seed within each stock, then the median across stocks.
s2 = pd.read_csv("output/s2_results.csv", dtype={"stock": str})
s2["stock"] = s2["stock"].str.zfill(6)
s2["horizon"] = s2["horizon"].astype(str)

encs = [("Fixed architecture (100×50)", stock_med_ratio(raw[(raw.model == "APSO-LSTM") & (raw.horizon.isin(["+5d", "+10d"]))], agg="median").reindex(["+5d", "+10d"]), "#1F4E79"),
        ("Unified-PSO (15×20)", stock_med_ratio(s2[s2.model == "Unified-PSO"], agg="median").reindex(["+5d", "+10d"]), "#2E75B6"),
        ("Staged-PSO (12×10 / 10×8)", stock_med_ratio(s2[s2.model == "Staged-PSO"], agg="median").reindex(["+5d", "+10d"]), "#9DC3E6")]
fig, ax = plt.subplots(figsize=(7.2, 4.0))
x = np.arange(2)
w = 0.26
for i, (name, s, c) in enumerate(encs):
    ax.bar(x + (i - 1) * w, s.values, width=w, color=c, label=name)
ax.set_xticks(x, ["+5d", "+10d"])
ax.set_xlabel("Forecast horizon")
ax.set_ylabel("Median RMSE ratio vs Persistence")
ax.axhline(1.0, color="#C00000", ls="--", lw=1.2)
# V3.5 (Review c21): Move the legend below the plot area to avoid overlapping with the +5d bar.
ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.16))
ax.set_title("Joint-encoding search trails the fixed architecture (S2)")
fig.tight_layout()
fig.savefig(f"{OUT}/fig3_encoding.png", dpi=200)
plt.close(fig)

# ---- Fig 4: S5-E1 penalty share & dispersion, 2 stocks x 2 window regimes ----
e1_main = pd.read_csv("output/s5_e1_parts/s5_e1.csv")
e1_win = pd.read_csv("output/s5_e1_parts/s5_e1_win400.csv")
panels = [  # (data, row, col)
    (e1_main[e1_main.stock == 600276], 0, 0),
    (e1_main[e1_main.stock == 600036], 0, 1),
    (e1_win[e1_win.stock == 600276], 1, 0),
    (e1_win[e1_win.stock == 600036], 1, 1),
]
COLS = ["Hengrui (600276)", "China Merchants Bank (600036)"]
ROWS = ["full-length training segment", "400-day short window"]
fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.2))
for df, r, c in panels:
    ax = axes[r, c]
    g = df.groupby("lam").agg(share=("penalty_share", "median"),
                              disp=("ratio", "std"),
                              n=("ratio", "count")).reset_index().sort_values("lam")
    xs = np.arange(len(g))
    ax.plot(xs, g["share"] * 100, color="#1F4E79", lw=2.2, marker="o")
    ax.set_ylim(-5, 105)
    dense = len(g) > 4
    ax.set_xticks(xs, [f"{v:g}" for v in g["lam"]],
                  rotation=35 if dense else 0, ha="right" if dense else "center")
    ax.set_xlabel("\u03bb")
    if c == 0:
        ax.set_ylabel("Penalty share (%)")  # V3.5 c22: Shorten the left axis title
    ax.grid(axis="y", color="0.9", lw=0.6)
    ax2 = ax.twinx()
    ax2.plot(xs, g["disp"], color="#C00000", lw=1.8, marker="s", ls="--")
    lo, hi = float(g["disp"].min()), float(g["disp"].max())
    span = hi - lo
    small = span < 0.02
    # Increase the amount of white space when the range is small, so that the 3 tick marks are sufficiently separated (to eliminate overlapping labels);
    # Discard the relative lower limit of hi for small spans, otherwise 0.06·hi will flatten the scale distribution.
    pad = max(span * (1.5 if small else 0.3), 1e-9) if small else max(span * 0.3, hi * 0.06, 1e-9)
    ax2.set_ylim(lo - pad, hi + pad)
    nd = 4 if small else 3
    ticks = np.linspace(lo, hi, 3)
    ax2.set_yticks(ticks)
    ax2.set_yticklabels([f"{v:.{nd}f}" for v in ticks], fontsize=8.5)
    if c == 1:
        ax2.set_ylabel("Cross-seed SD of RMSE ratio", color="#C00000", fontsize=9)
    if r == 0:
        ax.set_title(COLS[c], fontsize=10.5)
for r in range(2):
    fig.text(0.055, [0.72, 0.30][r], ROWS[r], rotation=90, va="center", fontsize=10)
handles = [plt.Line2D([], [], color="#1F4E79", lw=2.2, marker="o", label="Penalty share (left axis)"),
           plt.Line2D([], [], color="#C00000", lw=1.8, marker="s", ls="--", label="Cross-seed RMSE-ratio SD (right axis)")]
fig.legend(handles=handles, frameon=False, fontsize=9, loc="upper center",
           ncol=2, bbox_to_anchor=(0.55, 0.925))
# V3.7 (Review c10): Neutral title, describes only the endpoint panel, does not imply equivalence
fig.suptitle("S5-E1: penalty share versus \u03bb (endpoints of four stock \u00d7 window panels)",
             fontsize=11)
fig.text(0.55, 0.005, "\u03bb positions on the x-axis are categorical (levels are not evenly spaced)",
         ha="center", fontsize=8, color="0.35")
# V3.5 (Review c22): Increase line spacing to prevent the panel title from overlapping with the x-axis label in the figure above.
fig.tight_layout(rect=[0.09, 0.03, 1, 0.95], h_pad=3.0)
fig.savefig(f"{OUT}/fig4_penalty_panels.png", dpi=200)
plt.close(fig)
print("figs written to", OUT)
