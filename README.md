# PSO-LSTM A-Share Study — Data and Code Archive

Data and code underlying the manuscript:

> *A Systematic Evaluation of PSO-LSTM Hybrid Models for A-Share Stock Price Prediction:
> Architecture Search, Feature Extensions, and Search-Budget Effects*

## Contents

- `*.py` — experiment runner and analysis scripts (36 scripts), including the runner chain
  `s1_driver.py` … `s5_e3_budget_curve.py`, analysis scripts (`b1_stats_v35.py`,
  `analyze_controls.py`, …) and the figure script `paper_figs.py`.
- `output/` — run-level CSV outputs (7,110 archived run records) and the unified run manifest
  `output/run_manifest_v37.csv` (experiment × generating script × files × row counts).
- `output/archive_v2_unreachable/` — superseded pre-review result CSVs, retained for verification
  of the corrections documented in the manuscript revision notes.
- `data/` — post-adjusted daily bars (Eastmoney interface, 2015–2025) for the 12 study stocks.

## Reproduction

Software environment: Python 3.12.16, PyTorch 2.15.0 (CPU), pandas 3.0.2, numpy 2.4.4,
scipy 1.18.1. On Windows, run each experiment as a standalone module
(`python s1_driver.py`, …); runner scripts support chunked execution with resume.
Analysis scripts regenerate every statistic reported in the manuscript from the archived CSVs.

## License

MIT License. See `LICENSE`.
