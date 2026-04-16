# Daily Risk Monitoring (DRM) — SPX Analysis

A Python-based framework that replicates the daily risk monitoring
calculations employed in fund management. Given a historical
price series for the SPX total-return index, it computes Value at Risk,
runs a full backtesting suite, and produces a formatted Excel report
with embedded charts.

---

## Project structure

```
Project_April_2026/
│
├── load_data.py          Step 1 — load and clean the SPX CSV
├── rolling_window.py     Step 2 — rolling 3-year window + outlier filter
├── var.py                Step 3 — HS-VaR and MC-VaR calculation
├── exceptions.py         Step 4 — exception flags and backtesting
├── stress_test.py        Step 5 — stress test grid (current + stressed vols)
├── greeks.py             Black-Scholes Greeks calculator
├── export.py             Step 6 — Excel workbook export
│
├── spxtr_level_data.csv  Input: SPX total-return index daily OHLC
├── requirements.txt      Python dependencies
│
└── DRM_Analysis_Output.xlsx   Generated output workbook
```

---

## Running the analysis

Each step can be run independently for inspection, or the full pipeline
can be executed in one command via the export script.

**Run individual steps:**
```
python load_data.py          # verify data quality
python rolling_window.py     # inspect the 3-year window
python var.py                # compute today's VaR
python exceptions.py         # run the full backtest
python stress_test.py        # run the stress test grid
python greeks.py             # verify Greeks on a sample position
```

**Run the full pipeline and export to Excel:**
```
python export.py
```
---

## Configurable parameters

All steps prompt the user interactively at runtime. Press Enter to
accept the default value (matching the original DRM workbook settings).

| Parameter | Default | Description |
|---|---|---|
| Lookback window | 3 years | Historical window for VaR estimation |
| Outlier cutoff | ±3.5% | Returns beyond this excluded from MC pool |
| MC draws | 944 | Number of Monte Carlo resamples |
| VaR horizon | 20 days | Scaling horizon (UCITS regulatory standard) |
| Confidence level | 99% | VaR confidence level |
| Portfolio NAV | $10,000,000 | Used for stress test dollar P&L |
| Beta | 1.0 | Portfolio sensitivity to the index |
| Vol multiplier | 2.0× | Volatility scaling for stressed vol grid |

---

## Output workbook structure

| Sheet | Contents |
|---|---|
| README | Methodology notes and column definitions |
| SPX Returns | Full daily return history (9,200+ rows) |
| VaR Summary | Rolling backtest — one row per trading day |
| VaR Latest | T0 / T1 snapshot for the most recent date |
| Stress Test | ±40% scenario grid, both vol assumptions |
| Return Stats | Distribution statistics, 3-year vs full history |
| VaR Chart | Rolling 20-day VaR time series (1992–2026) |
| Distributions | Return histograms — window, MC, full history, filtered pool |

---

## Key financial concepts

**Daily return**  
`(Close_t / Close_{t-1}) - 1` — percentage change in index level day over day.

**Historical Simulation VaR**  
Sort the past 3 years of daily returns and read off the 1st percentile.
No distributional assumptions. The threshold below which losses fall on
only 1% of days.

**Monte Carlo VaR**  
Resample 944 returns from the filtered historical pool (outliers removed)
and read the 1st percentile of the simulated distribution.

**Exception**  
A day where the actual return was worse than the VaR predicted.
Expected rate: ~1% of days at 99% confidence.

**Square-root-of-time rule**  
`VaR_T = VaR_1day × √T` — scales 1-day VaR to a T-day horizon under
the assumption of independent daily returns.

**Stress test**  
Deterministic P&L calculation under a fixed market shock. Complements
VaR by testing scenarios beyond the historical distribution.

---

## Extending to options portfolios

The `greeks.py` module computes Black-Scholes Greeks for individual
option positions. To use them in the stress test, pass an
`option_positions` list to `compute_stress_test()`:

```python
positions = [
    {
        "K": 15000.0,       # strike price
        "T_days": 30,       # trading days to expiry
        "r": 0.045,         # risk-free rate
        "sigma": 0.20,      # implied volatility
        "option_type": "call",
        "quantity": -10,    # negative = short position
        "spc": 100,         # shares per contract
        "q": 0.0,           # dividend yield
    }
]
result = compute_stress_test(spx, option_positions=positions)
```

Greeks are summed across all positions. In Grid 2 (stressed vols),
`sigma` is scaled by `vol_multiplier` — producing the nonlinear
P&L profile visible in the curved red line of the DRM stress test chart.


