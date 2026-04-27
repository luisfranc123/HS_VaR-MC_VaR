"""
======================================================
Exception Flags and Backtesting
======================================================
What this file does:
Runs the VaR calculation across every trading day in the dataset 
and checks whether each day's actual return breached the VaR prediction. 
Then evaluates whether the overall exception rate us statistically acceptable (Binomial test).

Outputs: 
    1. complete day-by-day backtesting table
    2. summary report with exception rates and traffic light ratings
    3. the worst exceptions (days where losses most esceeded the VaR)

Relies on:
    load_data.py
    rolling_window.py
    var.py
"""
import numpy as np
import pandas as pd
from scipy import stats
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_data import load_spx
from rolling_window  import describe_window, prompt_parameters, default_lookback_years, \
                            default_outlier_cutoff, default_MC_draws
from var import compute_var, scale_var, default_horizon_days, default_confidence

csv_filename = "spxtr_level_data.csv"
#======================================================
# Section 1 — The Exception Check (single day)
#======================================================
def check_exception(actual_return: float, var_1day: float) -> int:
    """
    Check whether a single day's return breached the VaR threshold.

    Parameters:
        - actual_return: the realized daily return
        - var_1day: the VaR threshold for that day

    Returns: 
        1 if exception (breached occurred), 0 if not. 
    """
    return int(actual_return < var_1day)

#======================================================
# Section 2 — Rolling Backtest
#======================================================
def run_backtest(spx: pd.DataFrame, 
                lookback_years: int = default_lookback_years, 
                outlier_cutoff: float = default_outlier_cutoff, 
                mc_draws: int = default_MC_draws, 
                horizon_days: int = default_horizon_days, 
                confidence: float = default_confidence) -> pd.DataFrame:
    """
    Run the full rolling backtest across every available trading day.

    For each day:
        1. extract the trailing lookback_year window of returns
        2. compute HS-VaR and MC-VaR for that day
        3. compare the actual return against each VaR threshold
        4. record the result

    Parameters: 
        - spx: full SPX DataFrame from Step 1
        - lookback_years: window length in years
        - outlier_cutoff: outlier threshold for MC pool
        - mc_draws: number of MC samples
        - horizon_days: VaR scaling horizon
        - confidence: VaR confidence level
    
    Returns: 
    pd.DataFrame with one row per testable trading day, columns:
        - date — the trading date
        - actual_return — realized SPX daily return
        - hs_var_1day — Historical Simulation 1-day VaR
        - mc_var_1day — Monte Carlo 1-day VaR
        - hs_var_scaled — HS-VaR scaled to horizon_days
        - mc_var_scaled — MC-VaR scaled to horizon_days
        - exception_hs — 1 if actual_return breached hs_var_1day, else 0
        - exception_mc — 1 if actual_return breached mc_var_1day, else 0
        - excess_hs — how much the return exceeded HS VaR (negative = bad)
        - excess_mc — how much the return exceeded MC VaR (negative = bad)
    """
    
    # find the first date where a full lookback window exists
    first_date = spx["date"].min() + pd.DateOffset(years = lookback_years)
    testable = spx[spx["date"] >= first_date].reset_index(drop = True)

    n_days = len(testable)
    print(f"\n Running backtest across {n_days:,} trading days...")
    print(f"({testable['date'].min().date()} -> {testable['date'].max().date()})")
    print(f"This computes VaR for every single day")

    rows = []

    for i, row in testable.iterrows():
        date = row["date"]
        actual_return = row["daily_return"]

        # progress indicator
        if i % 500 == 0:
            pct = (i - testable.index[0])/n_days*100
            print(f"Progress: {pct:>5.1f}% ({date.date()})", end = "\r")

        #------------------------------------------------------
        # Build the window for this date
        #------------------------------------------------------  
        window_data = describe_window(
            spx,
            date,
            lookback_years = lookback_years,
            outlier_cutoff = outlier_cutoff,
            mc_draws = mc_draws,
        )

        # skip if the  window is too small 
        if window_data["n_full_window"] < 100:
            continue

        #------------------------------------------------------
        # Compute VaR for this date
        #------------------------------------------------------ 
        var_results = compute_var(
            window_data, 
            confidence = confidence, 
            horizon_days = horizon_days, 
            seed = 123
        )
        
        hs_var_1day = var_results["hs_var_1day"]
        mc_var_1day = var_results["mc_var_1day"]
        hs_var_scaled = var_results["hs_var_scaled"]
        mc_var_scaled = var_results["mc_var_scaled"]

        #------------------------------------------------------
        # Check exceptions
        #------------------------------------------------------ 
        exc_hs = check_exception(actual_return, hs_var_1day)
        exc_mc = check_exception(actual_return, mc_var_1day)

        # Excess: how far the actual return went beyond the VaR threshold.
        excess_hs = (actual_return - hs_var_1day) if exc_hs else 0.0
        excess_mc = (actual_return - mc_var_1day) if exc_mc else 0.0

        rows.append({
            "date": date,
            "actual_return": actual_return,
            "hs_var_1day": hs_var_1day,
            "mc_var_1day": mc_var_1day,
            "hs_var_scaled": hs_var_scaled,
            "mc_var_scaled": mc_var_scaled,
            "exception_hs": exc_hs,
            "exception_mc": exc_mc,
            "excess_hs": excess_hs,
            "excess_mc": excess_mc,
        })
    
    print(f"Progress: 100% — done.\n")
    return pd.DataFrame(rows)

#======================================================
# Section 3 — Exception Statistics
#======================================================
def compute_exception_stats(bt: pd.DataFrame, 
                            confidence: float = default_confidence) -> dict:
    """
    Compute summary statistics for the backtest results. 

    Parameters: 
        - bt: backtest DataFrame from run_backtest()
        - confidence: VaR confidence level used 
    
    Returns: 
        dict with counts, rates, traffic light ratings, and p-values
    """
    n_days = len(bt)
    expected_rate = 1 - confidence
    expected_count = n_days*expected_rate

    # --- Exception counts and rates ---
    n_exc_hs = int(bt["exception_hs"].sum())
    n_exc_mc = int(bt["exception_mc"].sum())
    rate_hs = n_exc_hs/n_days
    rate_mc = n_exc_mc/n_days

    # --- Traffic light assessment ---
    # based on the Basel framework scaled to our purposes
    def traffic_light(rate: float, expected: float) -> str:
        if rate <= expected*1.5:
            return "Green"
        elif rate <= expected*3.0:
            return "Yellow"
        else:
            return "Red"
    
    light_hs = traffic_light(rate_hs, expected_rate)
    light_mc = traffic_light(rate_mc, expected_rate)

    # --- Binomial p-value ---
    # scipy.stats.binomtest(k, n, p).
    #   k = observed exceptions
    #   n = total days
    #   p = expected probability per day (0.01 for 99% VaR)
    # alternative = "greater" tests whether the observed rate is
    # significantly higher than expected.

    pval_hs = stats.binomtest(n_exc_hs, n_days, expected_rate, 
                              alternative = 'greater').pvalue
    pval_mc = stats.binomtest(n_exc_mc, n_days, expected_rate, 
                              alternative = 'greater').pvalue
    
    # --- Worst Exceptions ---
    # the days where the actual return most exceeded the VaR threshold.
    worst_hs = (bt[bt["exception_hs"] == 1]
                .nsmallest(10, "excess_hs"))[["date", "actual_return", "hs_var_1day", "excess_hs"]]
    worst_mc = (bt[bt["exception_mc"] == 1]
                .nsmallest(10, "excess_mc"))[["date", "actual_return", "mc_var_1day", "excess_mc"]]
    
    # --- Annual exception counts ---
    # group exceptions by calendar year to see if failures cluster in time. 
    bt_copy = bt.copy()
    bt_copy["year"] = bt_copy["date"].dt.year
    annual = (bt_copy.groupby("year")
              .agg(n_days = ("actual_return", "count"), 
                   exc_hs = ("exception_hs", "sum"), 
                   exc_mc = ("exception_mc", "sum"))
                   .reset_index())
    annual["rate_hs"] = annual["exc_hs"]/annual["n_days"]
    annual["rate_mc"] = annual["exc_mc"]/annual["n_days"]

    return {
        # totals
        "n_days": n_days, 
        "confidence": confidence, 
        "expected_rate": expected_rate, 
        "expected_count": expected_count, 
        # HS results
        "n_exc_hs": n_exc_hs, 
        "rate_hs": rate_hs, 
        "light_hs": light_hs, 
        "pval_hs": pval_hs, 
        "worst_hs": worst_hs,
        # MC results
        "n_exc_mc": n_exc_mc, 
        "rate_mc": rate_mc, 
        "light_mc": light_mc, 
        "pval_mc": pval_mc, 
        "worst_mc": worst_mc,
        # Annual breakdown
        "annual": annual, 
    }

#======================================================
# Section 4 — Print Report
#======================================================
def print_backtest_report(s: dict) -> None:
    """
    print the full backtesting report.
    """
# traffic light symbols
    symbols = {"Green": "Green [Ok]", "Yellow": "Yellow [!]", "Red": "Red [!!]"}


    print("\n" + "=" * 65)
    print("BACKTESTING REPORT")
    print("=" * 65)
    print(f"\n  SETUP")
    print(f"  {'Total days tested':<35} {s['n_days']:>8,}")
    print(f"  {'Confidence level':<35} {s['confidence']:>8.0%}")
    print(f"  {'Expected exception rate':<35} {s['expected_rate']:>8.2%}")
    print(f"  {'Expected exception count':<35} {s['expected_count']:>8.1f}")

    print(f"\n  RESULTS SUMMARY")
    print(f"  {'Method':<28} {'Exceptions':>10} {'Rate':>8} "
            f"{'vs Expected':>12} {'Rating':>15}")
    print(f"  {'-'*63}")

    for label, n, rate, light in [
            ("Historical Simulation", s["n_exc_hs"], s["rate_hs"], s["light_hs"]),
            ("Monte Carlo",           s["n_exc_mc"], s["rate_mc"], s["light_mc"]),
        ]:
            diff = rate - s["expected_rate"]
            sign = "+" if diff >= 0 else ""
            print(f"  {label:<28} {n:>10,} {rate:>8.3%} "
                f"  {sign}{diff:.3%}    {symbols[light]:>15}")

    print(f"\n  STATISTICAL TEST  (Binomial — is exception rate acceptable?)")
    print(f"  Null hypothesis: true exception rate = {s['expected_rate']:.1%}")
    print(f"  A p-value < 0.05 means the model is likely miscalibrated.\n")
    print(f"  {'Method':<28} {'p-value':>10}  {'Conclusion'}")
    print(f"  {'-'*63}")

    for label, pval in [("Historical Simulation", s["pval_hs"]),
                         ("Monte Carlo",           s["pval_mc"])]:
        conclusion = "Model likely miscalibrated" if pval < 0.05 \
                     else "Within acceptable range"
        print(f"  {label:<28} {pval:>10.4f}   {conclusion}")

    print(f"\n  WORST EXCEPTIONS — HS-VaR  (days actual loss most exceeded VaR)")
    print(f"  {'Date':<14} {'Actual Return':>14} {'VaR Threshold':>14} "
          f"{'Excess':>10}")
    print(f"  {'-'*55}")
    for _, row in s["worst_hs"].iterrows():
        print(f"  {str(row['date'].date()):<14} "
              f"{row['actual_return']:>13.4%}  "
              f"{row['hs_var_1day']:>13.4%}  "
              f"{row['excess_hs']:>9.4%}")

    print(f"\n  ANNUAL EXCEPTION BREAKDOWN  "
          f"(clustering here is a warning sign)")
    print(f"  {'Year':<8} {'Days':>6} {'HS Exc':>8} {'HS Rate':>9} "
          f"{'MC Exc':>8} {'MC Rate':>9}")
    print(f"  {'-'*55}")
    for _, row in s["annual"].iterrows():
        # Flag years where the exception rate was notably high
        flag = " <--" if row["rate_hs"] > s["expected_rate"] * 3 else ""
        print(f"  {int(row['year']):<8} {int(row['n_days']):>6} "
              f"{int(row['exc_hs']):>8} {row['rate_hs']:>8.2%}  "
              f"{int(row['exc_mc']):>8} {row['rate_mc']:>8.2%}{flag}")

#======================================================
# User Input
#======================================================
def prompt_all_parameters() -> tuple:
    """
    Prompt for all configurable parameters. 
    """
    lookback_years, outlier_cutoff, mc_draws = prompt_parameters()
    
    # Horizon
    raw = input(f"  VaR horizon in trading days  "
                f"[default = {default_horizon_days}] : ").strip()
    horizon_days = default_horizon_days
    if raw != "":
        try:
            v = int(raw)
            horizon_days = v if 1 <= v <= 252 else default_horizon_days
        except ValueError:
            pass

    # Confidence
    raw = input(f"  Confidence level (0.90–0.99) "
                f"[default = {default_confidence}]  : ").strip()
    confidence = default_confidence
    if raw != "":
        try:
            v = float(raw)
            confidence = v if 0.90 <= v <= 0.99 else confidence
        except ValueError:
            pass

    print(f"\n  Parameters confirmed:")
    print(f"Lookback: {lookback_years} year(s)")
    print(f"Outlier cut: ±{outlier_cutoff:.1%}")
    print(f"MC draws: {mc_draws:,}")
    print(f"Horizon: {horizon_days} days")
    print(f"Confidence: {confidence:.0%}\n")

    return lookback_years, outlier_cutoff, mc_draws, horizon_days, confidence

#======================================================
# Entry Point
#======================================================
if __name__ == "__main__":
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, csv_filename)

    spx = load_spx(csv_path)
    print(f"\n SPX data loaded. Most recent date: {spx['date'].max().date()}")
    
    (lookback_years, outlier_cutoff, mc_draws, 
     horizon_days, confidence) = prompt_all_parameters()
    
    # Run rolling backtest
    bt = run_backtest(
        spx, 
        lookback_years = lookback_years,
        outlier_cutoff = outlier_cutoff,
        mc_draws = mc_draws,
        horizon_days = horizon_days,
        confidence = confidence
    )
    
    # Compute statistics
    stats_dict = compute_exception_stats(bt, confidence)

    # print report
    print_backtest_report(stats_dict)




