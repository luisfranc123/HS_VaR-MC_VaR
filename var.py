"""
======================================================
Vakue at risk (VaR) Calculation
======================================================
What this file does:
Computes the 1-day and 20-day 99% VaR using two methods:
    A. Historical SImulation (HS-VaR) — uses the full returns window .
    B. Monte Carlo Simulation (MC-VaR) — resamples from the filtered pool

Relies on: 
    load_data.py — load_spx()
    rolling_window.py — describe_window()
"""

import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_data import load_spx
from rolling_window  import describe_window, prompt_parameters, default_lookback_years, \
                            default_outlier_cutoff, default_MC_draws

csv_filename = "spxtr_level_data.csv"
default_confidence = 0.99
default_horizon_days = 20
random_seed = 123

#======================================================
# Method A — Historical Simulation VaR
#======================================================
def hs_var(full_window: pd.Series, 
          confidence: float = default_confidence, 
          horizon_days: int = default_horizon_days) -> dict:
    """
    Compute historical simulation VaR from the full return window.

    Parameters:
        - full_window: pd.Series of daily returns
        - confidence = VaR confidence level (default = 0.99)

    Returns:
        dict with keys:
            - var_1day — the 1-day VaR (e.g. -0.025)
            - var_20day — the 20-day VaR (VaR scaled by sqrt(20))
            - percentile — the significance level employed (1-confidence)
            - n_obs — number of observations used
            - method — label string
    """
    significance = (1 - confidence)*100

    #------------------------------------------------------
    # The Core Calculation
    #------------------------------------------------------   
    var_1day = float(np.percentile(full_window, significance))
    var_scaled = var_1day*np.sqrt(horizon_days)
    
    return {
     "method": "Historical Simulation", 
     "var_1day": var_1day, 
     "var_scaled": var_scaled, 
     "horizon_days": horizon_days, 
     "n_obs": len(full_window),
    }

#======================================================
# Method B — Monte Carlo Simulation VaR
#======================================================
def mc_var(filtered_pool: pd.Series, 
          mc_draws: int = default_MC_draws, 
          confidence: float = default_confidence, 
          horizon_days: int = default_horizon_days, 
          seed: int = random_seed) -> dict:
    """
    Compute Monte Carlo VaR by resampling from the filtered pool.

    Parameters: 
        - filtered_pool: pd.Series — outlier-filtered return pool
        - mc_draws: number of random samples to draw 
        - confidence: VaR confidence level 
        - horizon_days: time horizon for scaling 
        - seed: random seed for reproductibility

    Returns: 
         - dict with var_1day, var_scaled, the full simulated array, and diagnostics. 
    """
    rng = np.random.default_rng(seed)
    # pool: the filtered historical returns
    pool = filtered_pool.values
    # simulated: rng.choice(pool, size = n, replace = True)
    simulated = rng.choice(pool, size = mc_draws, replace = True)

    # Read the VaR from the simulated distribution
    significance = (1 - confidence)*100
    var_1day = float(np.percentile(simulated, significance))
    var_scaled = var_1day*np.sqrt(horizon_days)

    return {
        "method": "Monte Carlo Simulation",
        "var_1day": var_1day,
        "var_scaled": var_scaled,
        "horizon_days": horizon_days,
        "simulated": simulated,
        "n_pool": len(filtered_pool),
        "n_draws": mc_draws,
    }

#======================================================
# Combined Results
#======================================================  
def compute_var(window_data: dict, 
               confidence: float = default_confidence, 
               horizon_days: int = default_horizon_days, 
               seed: int = random_seed) -> dict:
    """
    Run both VaR methods and package the results together.

    Parameters:
    
        - window_data: dict from describe_window() 
        - confidence: VaR confidence level
        - horizon_days: time horizon in trading days
        - seed: random seed for MC
 
    Returns:
    
        dict with all results from both methods and a comparison.
    """
    # Historical simulation 
    hs = hs_var(
        full_window = window_data["full_window"], 
        confidence = confidence, 
        horizon_days = horizon_days, 
    )
    # Monte Carlo simulation
    mc = mc_var(
        filtered_pool = window_data["filtered_pool"], 
        mc_draws = window_data["mc_draws"], 
        confidence = confidence, 
        horizon_days = horizon_days, 
        seed = seed, 
    )

    diff_1day = mc["var_1day"] - hs["var_1day"]
    diff_scaled = mc["var_scaled"] - hs["var_scaled"]

    return {
        "as_of_date": window_data["as_of_date"],
        "confidence": confidence,
        "lookback_years": window_data["lookback_years"],
        "outlier_cutoff": window_data["outlier_cutoff"],
        "horizon_days": horizon_days,
        "hs_var_1day": hs["var_1day"],
        "hs_var_scaled": hs["var_scaled"],
        "hs_n_obs": hs["n_obs"],
        "mc_var_1day": mc["var_1day"],
        "mc_var_scaled": mc["var_scaled"],
        "mc_n_pool": mc["n_pool"],
        "mc_n_draws": mc["n_draws"],
        "mc_simulated": mc["simulated"],
        "diff_1day": diff_1day,
        "diff_scaled": diff_scaled,
        "diff_1day_bps": diff_1day*10_000,
        "diff_scaled_bps": diff_scaled*10_000,
    }

#======================================================
# Scale to arbitrary horizon
#======================================================  
def scale_var(var_1day: float, horizon_days: int) -> float:
    """
    Scale 1-day VaR to any horizon using the scale multiplication (np.sqrt(horizon_days))
    
    Parameters:
    
        - var_1day: 1-day VaR as a negative decimal (e.g. -0.025)
        - horizon_days: target horizon in trading days

    Returns: 
        float — scaled VaR 
    """
    return var_1day*np.sqrt(horizon_days)

#======================================================
# Print Report
#======================================================  
def print_var_report(r: dict) -> None:
    """
    Print the VaR report
    """
    h = r["horizon_days"]
    print("\n" + "=" * 65)
    print("STEP 3 — VaR RESULTS")
    print("=" * 65)
    print(f"\nCalculation date: {r['as_of_date'].date()}")
    print(f"Confidence level: {r['confidence']:.0%}")
    print(f"Lookback window: {r['lookback_years']} year(s)")
    print(f"Outlier cutoff: ± {r['outlier_cutoff']:.1%}")
    print(f"Horizon: {h} trading day(s)  "
          f"(√{h} = {np.sqrt(h):.4f})")
 
    # Warning for long horizons where sqrt(T) loses reliability
    if h > 30:
        print(f"\n [Note] Horizon > 30 days: the sqrt(T) scaling rule becomes")
        print(f"less reliable here. Treat the {h}-day VaR as an approximation.")
 
    print(f"\n  {'METHOD':<34}  {'1-Day VaR':>11}  {f'{h}-Day VaR':>11}")
    print(f"  {'-'*61}")
 
    print(f"  {'A. Historical Simulation':<34}"
          f"  {r['hs_var_1day']:>10.4%}"
          f"  {r['hs_var_scaled']:>10.4%}")
    print(f"     observations used : {r['hs_n_obs']:,}")
 
    print(f"\n  {'B. Monte Carlo Simulation':<34}"
          f"  {r['mc_var_1day']:>10.4%}"
          f"  {r['mc_var_scaled']:>10.4%}")
    print(f"     pool size  : {r['mc_n_pool']:,}  (full window minus outliers)")
    print(f"     draws      : {r['mc_n_draws']:,}")
 
    print(f"\nCOMPARISON  (MC minus HS)")
    print(f"{'1-Day  difference':<34}"
          f"{r['diff_1day']:>10.4%}"
          f"({r['diff_1day_bps']:>+.1f} bps)")
    print(f"{f'{h}-Day difference':<34}"
          f"{r['diff_scaled']:>10.4%}"
          f"({r['diff_scaled_bps']:>+.1f} bps)")
 
    if abs(r["diff_1day_bps"]) > 20:
        print(f"\n[Note] Difference > 20 bps — consider increasing MC draws.")
    else:
        print(f"\nMethods are consistent (difference within 20 bps). (OK)")
 
    print(f"\nINTERPRETATION")
    print(f"Based on the past {r['lookback_years']} year(s) of SPX returns:")
    print(f"- On {r['confidence']:.2%} of days, SPX should not lose more than "
          f"{abs(r['hs_var_1day']):.2%} in a single day (HS)")
    print(f" - On {r['confidence']:.2%} of days, SPX should not lose more than "
          f"{abs(r['mc_var_1day']):.2%} in a single day (MC)")
    print(f" - Over {h} trading days, the threshold becomes "
          f"{abs(r['hs_var_scaled']):.2%} (HS)")
 
    sim = r["mc_simulated"]
    print(f"\n  MC SIMULATED DISTRIBUTION ({r['mc_n_draws']:,} draws)")
    print(f"{'Range':<30} [{sim.min():.5f}, {sim.max():.5f}]")
    print(f"{'Mean':<30} {sim.mean():.7f}")
    print(f"{'Std deviation':<30} {sim.std():.7f}")
 
    n_tail    = max(10, int((1 - r['confidence'])*r['mc_n_draws']) + 1)
    worst_sim = np.sort(sim)[:n_tail]
    print(f"\nLEFT TAIL — {len(worst_sim)} worst simulated returns")
    print(f"(MC-VaR read from position ~{int((1 - r['confidence'])*r['mc_n_draws'])} of sorted array)")
    print(f"{'  '.join(f'{v:.5f}' for v in worst_sim)}")
 
#======================================================
# User Input
#======================================================  
def prompt_all_parameters() -> tuple:
    """
    Ask the user to enter the key parameters interactively. 
    Pressing enter without typing anything keeps the default inputs.
    """
    # First three parameters reuse the existing prompt from previous step. 
    lookback_years, outlier_cutoff, mc_draws = prompt_parameters()
    # Add the new horizon parameter
    raw = int(input(f"\nVaR horizon in trading days "
        f"[default = {default_horizon_days}]: ").strip())
    if raw == "":
        horizon_days = default_horizon_days
    else:
        try:
            horizon_days = raw
            if horizon_days < 1 or horizon_days > 252:
                print("Out of range (1-252). Using default.")
                horizon_days = default_horizon_days
        except ValueError:
            print("Not a valid integer. Using default.")
            horizon_days = default_horizon_days
    
    if horizon_days > 30:
        print(f"[Note] {horizon_days} days > 30: sqrt(T) scaling is an "
             f"approximation beyond this point.")
    print(f"  Horizon set to: {horizon_days} trading day(s)  "
          f"(√{horizon_days} = {np.sqrt(horizon_days):.4f})\n")

    raw = float(input(f"Confidence Interval (0.90 - 0.99) "
                     f"[default = {default_confidence}]: ").strip())
    if raw == "":
        confidence = default_confidence
    else:
        try:
            confidence = raw
            if confidence < 0.90 or confidence > 0.99:
                print("Out of range (0.90 - 0.99). Using default.")
                confidence = default_confidence
        except ValueError:
            print("Not a valid value Using default.")
            confidence = default_confidence
    
    return lookback_years, outlier_cutoff, mc_draws, horizon_days, confidence

#======================================================
# Entry Point
#====================================================== 
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, csv_filename)

    spx = load_spx(csv_path)
    latest_date = spx["date"].max()

    print(f"\nSPX data loaded. Most recent date: {latest_date.date()}")
    # Ask the user for input parameters
    lookback_years, outlier_cutoff, mc_draws, horizon_days, confidence = prompt_all_parameters()

    # Run the analysis with those parameters
    window_data = describe_window(
        spx, 
        latest_date, 
        lookback_years = lookback_years, 
        outlier_cutoff = outlier_cutoff, 
        mc_draws = mc_draws,         
    )

    results = compute_var(window_data, 
               confidence, 
               horizon_days = horizon_days
               )
    print_var_report(results)