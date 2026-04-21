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
# Method C — Fund-Level VaR 
#======================================================  
def fund_var(positions: list, 
             scenario_levels: "pd.Series", 
             base_spx: float, 
             confidence: float = default_confidence, 
             horizon_days: int = default_horizon_days) -> dict:
    """
    Compute fund-level VaR by repricing all option positions 
    under each simulated SPX scenario.

    HOW IT WORKS
    ------------
    For each of the N scenario SPX levels:
        1. Derive the scenario price for each position's underlying:
           SPX/SPXW/XSP: use scenario_spx directly
           SPY: use scenario_spx/spy_multiplier
        2. Reprice each option using Black-Scholes:
           S = scenario price, T = T_days - 1 (one day passes)
        3. P&L per position = (new_price - base_price) * qty * spc
        4. Sum P&L across positions = portfolio P&L in dollars
        5. Divide by portfolio gross value = fund return (%)
    Sort the N fund returns and read the 1st percentile = fund VaR.

    Parameters: 
        - positions: list of position dicts from read_positions.py
        - scenario_levels: pd.Series of simulated SPX levels.
        - base_spx: today's SPX closing level
        - confidence: VaR confidence level 
        - horizon_days: scaling horizon in trading days

    Retirns:
    dict with keys:
        var_1day — 1-day fund VaR 
        var_scaled — VaR scaled to horizon_days
        distribution — pd.Series of N fund returns 
        n_scenarios — number of scenarios run
        n_positions — number of positions repriced
        portfolio_value — gross portfolio value used as denominator
        spy_multiplier — SPX/SPY ratio derived from position data 
    """
    from greeks import bs_price

    # ratio = base_spx\spy_hedgeprice
    spy_prices = [p["S"] for p in positions if p["underlying"] == "spy"]
    spy_multiplier = base_spx/spy_prices[0] if spy_prices else 10.029

    # Portfolio gross value — denominator for fund return calculation.
    # Use DRM PosnDollars where available, else compute from inputs.
    net_posn = sum(
        p["drm_posn_dollars"] for p in positions
        if p["drm_posn_dollars"] is not None)
    # Fallback: if PosnDollars not available, estimate from inputs
    if net_posn == 0:
        net_posn = sum(p["S"]*p["quantity"]*p["spc"] for p in positions)
    portfolio_value = abs(net_posn)

    # --- Reprice every position under every scenario ---
    fund_returns = []

    for spx_level in scenario_levels:

        total_pl = 0.0

        for p in positions:

            # scenario underlying price
            if p["underlying"] in ("spx", "xsp"):
                S_new = float(spx_level)
            elif p["underlying"] == "spy":
                S_new = float(spx_level)/spy_multiplier
            else: 
                continue

            # Time remaining after one day passes
            T_new = p["T_days"] - 1.0

            if T_new <= 0:
                # Expired
                if p["option_type"] == "call":
                    new_price = max(S_new - p["K"], 0.0)
                else:
                    new_price = max(p["K"] - S_new, 0.0)
            else:
                try:
                    new_price = bs_price(
                        S = S_new,
                        K = p["K"],
                        T = T_new / 252.0,
                        r = p["r"],
                        sigma = p["sigma"],
                        option_type = p["option_type"],
                        q = p["q"],
                    )
                except Exception:
                    new_price = p["S"]
            
            # P&L = (new option price - base option price)*qty*spc
            base_price = p.get("base_price", p["S"])
            total_pl += (new_price - base_price)*p["quantity"]*p["spc"]
        
        fund_returns.append(
            total_pl/portfolio_value if portfolio_value != 0 else 0.0)
    
    distribution = pd.Series(sorted(fund_returns))
    significance = (1 - confidence)*100 
    var_1day = float(np.percentile(distribution, significance))
    var_scaled = var_1day*np.sqrt(horizon_days)

    return {
        "var_1day": var_1day,
        "var_scaled": var_scaled,
        "distribution": distribution,
        "n_scenarios": len(scenario_levels),
        "n_positions": len(positions),
        "portfolio_value": portfolio_value,
        "spy_multiplier": spy_multiplier,
    }

#======================================================
# Combined Results
#======================================================  
def compute_var(window_data: dict, 
               confidence: float = default_confidence, 
               horizon_days: int = default_horizon_days, 
               seed: int = random_seed, 
               positions: list = None, 
               scenario_levels: "pd.Series" = None, 
               base_spx: float = None) -> dict:
    """
    Run both VaR methods and package the results together.

    Parameters:
    
        - window_data: dict from describe_window() 
        - confidence: VaR confidence level
        - horizon_days: time horizon in trading days
        - seed: random seed for MC
        - positions: list of position dicts from read_positions.py
          When provided, fund_var() is computed. When None, 
          only HS and MC VaR are computed.
        - scenario_levels: pd.Series of simulated SPX levels
          Required when positions is provided.
        - base_spx: today's SPX level (required with positions)

 
    Returns:
    
        dict with HS, MC, and optionally fund VaR results.
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

    result = {
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
        "fund_var_1day": None,
        "fund_var_scaled": None,
        "fund_var_dist": None,
        "fund_var_ratio": None,
        "fund_n_scenarios": None,
        "fund_portfolio_value": None,
        "fund_spy_multiplier": None,
    }

    # Run fund VaR only when positions and scenario_levels are supplied
    if positions and scenario_levels is not None and base_spx is not None:
        print(" [Fund VaR] Repricing positions under each scenario...")
        fv = fund_var(
            positions = positions, 
            scenario_levels = scenario_levels, 
            base_spx = base_spx, 
            confidence = confidence, 
            horizon_days = horizon_days, 
        )
        result["fund_var_1day"] = fv["var_1day"]
        result["fund_var_scaled"] = fv["var_scaled"]
        result["fund_var_dist"] = fv["distribution"]
        result["fund_n_scenarios"] = fv["n_scenarios"]
        result["fund_portfolio_value"] = fv["portfolio_value"]
        result["fund_spy_multiplier"] = fv["spy_multiplier"]
        # VaR ratio: fund VaR/SPX HS-VaR
        if hs["var_1day"] != 0:
            result["fund_var_ratio"] = fv["var_1day"]/hs["var_1day"]
    
    return result
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


    # --- Fund VaR section — only shown when positions were provided ---
    if r.get("fund_var_1day") is not None:
        fv1  = r["fund_var_1day"]
        fvs  = r["fund_var_scaled"]
        rat  = r["fund_var_ratio"]
        dist = r["fund_var_dist"]

        print(f"\n" + "=" * 65)
        print(f"  FUND-LEVEL VaR  (position repricing)")
        print("=" * 65)
        print(f"  Scenarios run: {r['fund_n_scenarios']:,}")
        print(f"  Positions repriced: {r['fund_n_scenarios']} x "
              f"{len(r['fund_var_dist'])} fund returns")
        print(f"  Portfolio value: ${r['fund_portfolio_value']:>15,.0f}")
        print(f"  SPY multiplier: {r['fund_spy_multiplier']:.4f}")

        print(f"\n  {'METHOD':<30}  {'1-Day VaR':>11}  {f'{h}-Day VaR':>11}")
        print(f"  {'-'*57}")
        print(f"  {'Our fund VaR':<30}  {fv1:>10.4%}  {fvs:>10.4%}")
        print(f"  {'SPX HS-VaR (benchmark)':<30}  "
              f"{r['hs_var_1day']:>10.4%}  {r['hs_var_scaled']:>10.4%}")
        print(f"\n  VaR Ratio (fund / SPX)  : {rat:.4f}  "
              f"({'fund < index' if abs(fv1) < abs(r['hs_var_1day']) else 'fund > index'})")

        print(f"\n  FUND RETURN DISTRIBUTION ({len(dist):,} scenarios)")
        print(f"  {'Range':<30} [{dist.min():.5f},  {dist.max():.5f}]")
        print(f"  {'Mean':<30} {dist.mean():.7f}")
        print(f"  {'Std deviation':<30} {dist.std():.7f}")

        worst_fund = dist.values[:10]
        print(f"\n  LEFT TAIL — 10 worst fund returns")
        print(f"  {'  '.join(f'{v:.5f}' for v in worst_fund)}")

        print(f"\n  NOTE: Our scenarios are capped at ±{r['outlier_cutoff']:.1%}.")
        print(f"  The DRM system uses wider scenarios (up to ~±9%).")
        print(f"  To narrow the gap, increase the outlier cutoff when prompted.")

    print("\n" + "=" * 65)
    print("  VaR RESULTS")
    print("=" * 65)
    print(f"\n  Calculation date: {r['as_of_date'].date()}")
    print(f"  Confidence level: {r['confidence']:.0%}")
    print(f"  Lookback window: {r['lookback_years']} year(s)")
    print(f"  Outlier cutoff: ± {r['outlier_cutoff']:.1%}")
    print(f"  Horizon: {h} trading day(s)  (√{h} = {np.sqrt(h):.4f})")

    if h > 30:
        print(f"\n  [Note] Horizon > 30 days: sqrt(T) scaling becomes less")
        print(f"  reliable. Treat the {h}-day VaR as an approximation.")

    print(f"\n  {'METHOD':<34}  {'1-Day VaR':>11}  {f'{h}-Day VaR':>11}")
    print(f"  {'-'*61}")

    print(f"  {'A. Historical Simulation':<34}"
          f"  {r['hs_var_1day']:>10.4%}"
          f"  {r['hs_var_scaled']:>10.4%}")
    print(f"     observations used : {r['hs_n_obs']:,}")

    print(f"\n  {'B. Monte Carlo Simulation':<34}"
          f"  {r['mc_var_1day']:>10.4%}"
          f"  {r['mc_var_scaled']:>10.4%}")
    print(f"     pool size: {r['mc_n_pool']:,}  (full window minus outliers)")
    print(f"     draws: {r['mc_n_draws']:,}")

    print(f"\n  COMPARISON  (MC minus HS)")
    print(f"  {'1-Day  difference':<34}"
          f"  {r['diff_1day']:>10.4%}"
          f"  ({r['diff_1day_bps']:>+.1f} bps)")
    print(f"  {f'{h}-Day difference':<34}"
          f"  {r['diff_scaled']:>10.4%}"
          f"  ({r['diff_scaled_bps']:>+.1f} bps)")

    if abs(r["diff_1day_bps"]) > 20:
        print(f"\n  [Note] Difference > 20 bps — consider increasing MC draws.")
    else:
        print(f"\n  Methods are consistent (difference within 20 bps).  (OK)")

    print(f"\n  INTERPRETATION")
    print(f"  Based on the past {r['lookback_years']} year(s) of SPX returns:")
    print(f"  - On {r['confidence']:.0%} of days, loss will not exceed "
          f"{abs(r['hs_var_1day']):.2%} in a single day  (HS)")
    print(f"  - On {r['confidence']:.0%} of days, loss will not exceed "
          f"{abs(r['mc_var_1day']):.2%} in a single day  (MC)")
    print(f"  - Over {h} trading days, the threshold becomes "
          f"{abs(r['hs_var_scaled']):.2%}  (HS)")

    sim = r["mc_simulated"]
    print(f"\n  MC SIMULATED DISTRIBUTION  ({r['mc_n_draws']:,} draws)")
    print(f"  {'Range':<30} [{sim.min():.5f},  {sim.max():.5f}]")
    print(f"  {'Mean':<30} {sim.mean():.7f}")
    print(f"  {'Std deviation':<30} {sim.std():.7f}")

    # Correct left tail slice: significance% of mc_draws
    # e.g. 1% of 944 = 9.44 → show the 10 worst
    n_tail = max(10, int((1 - r["confidence"]) * r["mc_n_draws"]) + 1)
    worst_sim = np.sort(sim)[:n_tail]
    print(f"\n  LEFT TAIL — {n_tail} worst simulated returns")
    print(f"  (MC-VaR read from position ~{int((1-r['confidence'])*r['mc_n_draws'])} "
          f"of sorted array)")
    print(f"  {'  '.join(f'{v:.5f}' for v in worst_sim)}")

    print("\n" + "=" * 65)
    print("  VaR calculation complete.")
    print("=" * 65)
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
    raw = input(f"\nVaR horizon in trading days "
        f"[default = {default_horizon_days}]: ").strip()
    if raw == "":
        horizon_days = default_horizon_days
    else:
        try:
            horizon_days = int(raw)
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
    
    # --- Confidence ---
    raw = input(f"Confidence Interval (0.90 - 0.99) "
                     f"[default = {default_confidence}]: ").strip()
    if raw == "":
        confidence = default_confidence
    else:
        try:
            confidence = float(raw)
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