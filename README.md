# Particle Swarm Optimization–Long Short-Term Memory Hybrid Models for A-Share Stock Price Prediction

## Description

This archive contains the Python code, input data snapshots, run-level experimental outputs, analysis scripts, and figure-generation code supporting the manuscript:

> *A Systematic Evaluation of Particle Swarm Optimization–Long Short-Term Memory Hybrid Models for A-Share Stock Price Prediction: Architecture Search, Feature Extensions, and Search-Budget Effects*

The study evaluates fixed-architecture and joint architecture–weight variants of particle swarm optimization (PSO) combined with long short-term memory (LSTM) networks. It includes Persistence, autoregressive integrated moving average (ARIMA), and equally protocolled Adam-only controls; anchored walk-forward evaluation; multiple random seeds; feature ablations; search-budget controls; and synthetic mechanism probes. The archive is intended to make both the reported negative results and the positive mechanism findings auditable. It is a research artifact, not investment advice or a live trading system.

Archive version: 1.0.1 (English documentation revision, 2026-11-15). Numerical results and model configurations are unchanged from version 1.0.0; code comments, documentation, human-readable labels, and progress messages have been translated into English.

## Dataset Information

### Source and coverage

- Provider: Eastmoney historical quote interface (`https://push2his.eastmoney.com/api/qt/stock/kline/get`). No access token is required.
- Instrument type: mainland China A-share stocks.
- Frequency and adjustment: daily bars (`klt=101`), post-adjusted prices (`fqt=2`).
- Requested range: 2015-01-01 through 2025-12-31. Series begin later for stocks listed after 2015, notably CATL (300750) and SMIC (688981).
- CSV schema: `date, open, close, high, low, volume, amount`. Dates are ISO `YYYY-MM-DD`; prices are post-adjusted; `volume` and `amount` retain the provider's units.
- Data snapshots are included, so the archived analyses do not require network access. If a requested CSV is absent, the fetch functions download and cache it. Researchers re-fetching data should record the retrieval date because a provider may revise historical adjustments.

### Study sample

The 12 study stocks are stratified into four stocks per historical volatility tier:

| Tier | Stock codes | English names |
|---|---|---|
| Low volatility | 600900, 600036, 600519, 600887 | China Yangtze Power; China Merchants Bank; Kweichow Moutai; Yili Group |
| Medium volatility | 000333, 600276, 600031, 002415 | Midea Group; Hengrui Pharmaceuticals; SANY Heavy Industry; Hikvision |
| High volatility | 688981, 300750, 300274, 300308 | SMIC; CATL; Sungrow Power Supply; Zhongji Innolight |

`data/` contains the 12 study-stock CSVs. `data_pool/` contains the complete 32-stock candidate pool used by `select_stocks.py`, including all 12 study series. The primary walk-forward runners read from `data_pool/`; the early end-to-end prototype `run_experiment.py` reads from `data/`.

### Preprocessing

Rows are sorted by date, converted to numeric values, and filtered to positive closing prices. No missing-value imputation is applied. Sliding windows use observations through day *t* to predict the close at *t+h−1*. For feature experiments, leading rows with indicator-derived missing values are removed before sequence construction. All features are causal and use data available at or before *t*. Min–max scaling is fitted independently within each training fold and only applied to the corresponding test fold. Predictions and targets are inverse-transformed to the original post-adjusted price scale before metric calculation. Samples without a complete *h*-day label are excluded at each test-window boundary.

## Code Information

The archive contains 38 Python scripts at the repository root:

- `pso_lstm.py`: LSTM regressor, PSO/adaptive PSO, architecture decoders, unified and staged joint encodings, sequence construction, and metrics.
- `data_fetch.py`, `select_stocks.py`: data download/cache, candidate pool, and volatility-based stock selection.
- `run_m1_baseline.py`, `s1_driver.py`: S1 fixed-architecture adaptive PSO–LSTM and Persistence experiments with five anchored walk-forward folds and five seeds.
- `run_s2.py`, `s2_driver.py`, `s2_rerun.py`, `merge_s2_run.py`: S2 unified versus staged joint-encoding experiments and merging of run-level outputs.
- `run_s3.py`, `s3_driver.py`, `s3_master.py`: S3 price-only, technical-feature, and sentiment/capital-proxy feature ablations.
- `run_arima_baseline.py`, `arima_all.py`: aligned rolling-origin ARIMA(p,1,0) baseline with training-fold AIC order selection.
- `run_adam_baseline.py`, `adam_all.py`: Adam-only control using the same architecture, folds, seeds, and fine-tuning protocol without PSO pre-search.
- `gen_b1_cells.py`, `gen_v31_numbers.py`, `b1_stats_v35.py`, `s4_analyze.py`: paired ratios, canonical descriptive values, stock-level inference, and volatility-tier analyses.
- `s5_e1.py`, `s5_e1_cmb400.py`, `s5_e1_blockperm.py`: weight-penalty probes and seed-blocked permutation tests.
- `s5_e3.py`, `s5_e3_fill.py`, `s5_e3_controls.py`, `run_controls_pool.py`, `run_multiseed.py`: synthetic signal-strength experiments and initialization, random-search, no-signal, memory-length, and generator-seed controls.
- `s5_e3_probe.py`, `s5_e3_probes.py`, `s5_e3_budget_curve.py`, `run_budget_curve.py`, `s5_e3_diag.py`, `h2_diagnose.py`, `s5_theory_calib.py`: mechanism, budget, reachability, and numerical-calibration probes.
- `run_experiment.py`: early end-to-end prototype; it is not the source of the final walk-forward tables.
- `paper_figs.py`: regenerates manuscript figures from the archived run-level CSVs.

`output/run_manifest_v38.csv` maps each experiment to its generating scripts, archived files, and row counts. The manifest covers 7,630 run-level records. `output/archive_v2_unreachable/` retains superseded pre-correction files solely for auditability; these files are not inputs to the final reported results. Temporary `tmp_*.csv` files are incremental checkpoints and should not be appended to canonical outputs a second time.

## Requirements

The verified environment was:

- Windows 11 64-bit, AMD64; Intel64 Family 6 Model 170 processor; 22 logical CPUs.
- Python 3.12.16.
- PyTorch 2.15.0+cpu (CUDA was not used).
- NumPy 2.4.4, pandas 3.0.2, SciPy 1.18.1, and Matplotlib 3.10.9.

Exact package entries are provided in `requirements.txt`. The ARIMA baseline is implemented with NumPy and does not require `statsmodels`. PyTorch is restricted to one thread in the primary runners for reproducibility. Full scans are computationally expensive; analysis of the archived outputs is much faster and does not require retraining.

## Usage Instructions

Run commands from the extracted archive root so that relative `data/`, `data_pool/`, and `output/` paths resolve correctly.

### 1. Create an environment and install dependencies

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. Reproduce analyses from the archived results

No network access or model retraining is needed for these commands:

```bash
python gen_b1_cells.py
python gen_v31_numbers.py
python b1_stats_v35.py
python s4_analyze.py
python analyze_controls.py
python s5_e1_blockperm.py
python paper_figs.py
```

Generated summaries are written under `output/`; figures are written to `output/paper_figs/`. Running these commands overwrites or refreshes the corresponding derived files, so copy the archive first if you need to preserve the delivered derived outputs byte-for-byte.

### 3. Run a small model smoke test

```bash
python run_m1_baseline.py --stocks 600036 --horizons 5 --folds 0 --seeds 0 --apso-particles 15 --apso-iters 20 --tag smoke
```

Results are appended to `output/smoke_parts/600036.csv`. This small run uses the prototype search budget, not the production S1 budget.

### 4. Re-run the principal experiments

The following entry points support chunked or incremental execution. Existing completed combinations in the relevant `output/*_parts/` directory are skipped.

```bash
python s1_driver.py --start 0 --count 8
python s2_driver.py --start 0 --count 8
python s3_driver.py --start 0 --count 8
python arima_all.py
python adam_all.py
python s5_e1.py
python s5_e1_cmb400.py
python s5_e3.py
python s5_e3_fill.py
python run_controls_pool.py
python run_multiseed.py
python s5_e3_probes.py
python run_budget_curve.py
python s5_theory_calib.py
```

For S1, the driver defaults encode the production configuration: 100 particles, 50 iterations, weight penalty 0.1, fitness subsample 600, five folds, and seeds 0–4. Use `--help` on individual runners for stock, horizon, fold, seed, and budget options. On Windows, run drivers from the archive root and avoid moving scripts between directories because multiprocessing uses the spawn mechanism.

## Methodology

The primary evaluation uses five anchored walk-forward folds. Fold *k* trains on the first `(50 + 10k)%` of a stock's chronological series and tests on the following 10%, for `k = 0, …, 4`. The LSTM has one recurrent hidden layer and a linear output layer. Adaptive PSO evolves network initial weights before Adam fine-tuning. Joint variants additionally search sequence length and hidden width. The synthetic experiments implant a known decaying-memory signal and vary its variance share to test when architecture search responds to structure.

The main error metric is root mean squared error (RMSE) on the inverse-scaled price scale. RMSE ratios to Persistence provide paired, scale-normalized comparisons across stocks. Supporting metrics include mean squared error, mean absolute percentage error, mean arctangent absolute percentage error, symmetric mean absolute percentage error, Theil's U, normalized RMSE, and percentage similarity. Inferential analyses use stock-level aggregation rather than treating repeated fold–seed runs as independent stocks; details and multiple-testing controls are given in the manuscript and analysis scripts.

## Citations

If you use this archive, cite the associated PeerJ Computer Science manuscript once its publication details are available. Until then, cite this repository/archive as:

> Liu, GangQiang (2026). *Particle Swarm Optimization–Long Short-Term Memory Hybrid Models for A-Share Stock Price Prediction: Data and Code Archive* (Version 1.0.1). GitHub: https://github.com/LiuGangQiang/pso-lstm-astock

The manuscript's theoretical and implementation background is based on the references cited in the article; no separate third-party Python package is used for ARIMA.

## License and Contribution Guidelines

The original code is released under the MIT License; see `LICENSE`. Third-party Eastmoney market data remain subject to the provider's terms and are included solely to support verification of this research. The MIT License does not override third-party data rights.

For contributions, open an issue or pull request at `https://github.com/LiuGangQiang/pso-lstm-astock`. Please use English in code, comments, and documentation; preserve chronological data separation; add a small reproducible test for any change; and do not commit credentials, proprietary datasets, or investment recommendations. Changes affecting numerical results should include the generating command, software environment, random seeds, and a row-count/manifest update.
