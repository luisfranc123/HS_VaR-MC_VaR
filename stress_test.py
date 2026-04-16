"""
======================================================
Stress Test
======================================================
What this file does:
Applies a grid of hypothetical market shocks to the current
assets' level and computes the resulting P&L for each scenario. 

Two grids:
    Grid 1 — current vols (RM01 - RM08): market move only
    Grid 2 — stressed vols (RM09 - RM18): market move + vol bump

Configurable parameters
    - shock steps — size of each step in the grid (default 10%)
    - max_shock — largest shock to test (default 40%)
    - portfolio_value — NAV used for dollar P&L (default $10,000,000)
    - beta — portfolio sensitivity to the index (default 1.0)
    - vol_multiplier — how much to scale vol in Grid 2 (default 2.0)

Relies on:
    load_data.py — load_spx()
"""
import numpy as np
import pandas as pd
from greeks import compute_greeks
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_data import load_spx

csv_filename = "spxtr_level_data.csv"

# --- Default parameters ---
default_shock_steps = 0.10
default_max_shock = 0.40
default_portfolio_value = 10_000_000.0
default_beta = 1.0
default_vol_multiplier = 2.0

#======================================================
# Section 1 — Build the scneario grid
#======================================================
def build_scenario_grid(max_shock: float = default_max_shock, 
                        shock_steps: float = default_shock_steps) -> list:
    """
    Build the list of shock scenarios matching the DRM RM-code convention.
    
    Parameters:
        - max_shock: largest absolute shock to include 
        - shock_steps: size of each step 
    
    Returns:
    List of dicts, each with keys:
        - rm_code — scenario label (e.g. "RM01", "RM02")
        - shock — the fractional move (e.g. -0.40, 0.0, +0.10)
    """
    # Build the negative shoks from most negative to zero, 
    # then the positive shocks from smallest to largest.
    n_steps = round(max_shock/shock_steps)
    actual_max = n_steps*shock_steps
    if abs(actual_max - max_shock) > 0.001:
        print(f"[Note] max_schock {max_shock:.0%} is not a clean multiple of "
              f"step size {shock_steps:.0%}. "
              f"Using {actual_max:.0%} as the actual maximum.")
    negative_shocks = [-round(shock_steps*i, 10) for i in range(n_steps, 0, -1)]
    positive_shocks = [round(shock_steps*i, 10) for i in range(1, n_steps + 1)]

    all_shocks = negative_shocks + [0.0] + positive_shocks

    # Assign RM-codes
    scenarios = []
    neg_counter = 1
    pos_counter = n_steps + 1
    
    for shock in all_shocks:
        if shock < 0:
            rm_code = f"RM{neg_counter:02d}"
            neg_counter += 1
        elif abs(shock) < 1e-10:
            rm_code = "RM00"
        else:
            rm_code = f"RM{pos_counter:02d}"
            pos_counter += 1
        scenarios.append({"rm_code": rm_code, "shock": shock})
    
    return scenarios

#======================================================
# Section 2 — Apply shocks
#======================================================
def apply_stress(spx_level: float, 
                 scenarios: list, 
                 portfolio_value: float = default_portfolio_value, 
                 beta: float = default_beta, 
                 vol_multiplier: float = 1.0, 
                 option_positions: list = None) -> pd.DataFrame:
    """
    Apply a list of shock scenarios to the current SPX level and 
    compute the resulting P&L for each scenario. 

    Parameters: 
        - spx_level: current SPX closing level
        - scenarios: list of dicts from build_scenario_grid()
        - portfolio_value: portfolio NAV in dollars
        - beta: portfolio sensitivity to the index
                        1.0  = moves 1-for-1 with SPX (pure index fund)
                        0.5  = moves half as much (hedged fund)
                        -1.0 = inverse fund (profits when SPX falls)
        - vol_multiplier: volatility scaling factor for Grid 2
                        1.0 = current vols (no vol change)
                        2.0 = stressed vols (vol doubled)
        - option_position: optional list of option position dicts for
                           non-index portfolios. Each dict must contain:
                           - K — strike price
                           - T_days — trading days to expiry
                           - r — risk-free rate
                           - sigma
    
    Returns: 
        pd.DataFrame with one row per scenario and columns:
            rm_code — scenario identifier
            shock_pct — the hypothetical index move
            spx_current — SPX level before the shock
            spx_stressed — SPX level after applying the shock
            vol_multiplier — vol scaling applied (1.0 or 2.0)
            pl_pct — portfolio P&L as a fraction
            pl_dollars — P&L in dollar terms
            position_value — portfolio value after the shock
            delta — dollar sensitivity to a 1-point SPX move
            gamma — change in delta per 1-point SPX move
            vega — sensitivity to a 1-vol-point change
            theta — daily time decay (dollars per day)
            rho — interest rate sensitivity
    """
    rows = []

    for s in scenarios:
        shock = s["shock"]
        spx_stressed = spx_level*(1 + shock)
        pl_pct = beta*shock
        pl_dollars = portfolio_value*pl_pct
        position_val = portfolio_value + pl_dollars

        # --- Greeks ---
        # Case A — Options portfolio: compute from Black-Scholes
        # Sigma isscaled by vol_multiplier for Grid 2 (stressed vols).
        if option_positions:
            delta = gamma = vega = theta = rho = 0.0
            for pos in option_positions:
                g = compute_greeks(
                    S = spx_stressed, 
                    K = pos["K"], 
                    T_days = pos["T_days"], 
                    r = pos["r"], 
                    sigma = pos["sigma"]*vol_multiplier, 
                    option_type = pos["option_type"], 
                    quantity = pos["quantity"], 
                    spc = pos.get("spc", 100.0), 
                    q = pos.get("q", 0.0), 
                )
                delta += g["delta_dollar"]
                gamma += g["gamma_dollar"]
                vega += g["vega_dollar"]
                theta += g["theta_dollar"]
                rho += g["rho_dollar"]
        # Case B — Pure index portfolio: Delta only, all others zero. 
        else:
            delta = (portfolio_value*beta)/spx_level if spx_level != 0 else 0.0
            gamma = 0.0 
            vega = 0.0 
            theta = 0.0
            rho = 0.0
        
        rows.append({
            "rm_code": s["rm_code"],
            "shock_pct": shock,
            "spx_current": spx_level,
            "spx_stressed": spx_stressed,
            "vol_multiplier": vol_multiplier,
            "pl_pct": pl_pct,
            "pl_dollars": pl_dollars,
            "position_value": position_val,
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "theta": theta,
            "rho": rho,
        })
    
    return pd.DataFrame(rows)

#======================================================
# Section 3 — Compute both grids
#======================================================
def compute_stress_test(spx: pd.DataFrame, 
                        as_of_date: pd.Timestamp = None, 
                        portfolio_value: float = default_portfolio_value, 
                        beta: float = default_beta, 
                        vol_multiplier: float = default_vol_multiplier, 
                        max_shock: float = default_max_shock, 
                        shock_steps: float = default_shock_steps, 
                        option_positions: list = None) -> dict:
    """
    Compute both stress test grids for a given date.

    Parameters: 
        - spx: full SPX DataFrame from load_spx()
        - as_of_date: valuation date (defaults to most recent date in spx)
        - portfolio_value: portfolio NAV in dollars
        - beta: portfolio sensitivity to the index
        - vol_multiplier: vol scaling for Grid 2 (default 2.0 = double vol)
        - max_shock: largest shock to test (default 0.40)
        - shock_steps: step size between scenarios (default 0.10)
    
    Returns: 
    dict with keys:
        as_of_date — the valuation date used
        spx_level — SPX closing level on that date
        portfolio_value — NAV used
        beta — beta used
        scenarios — the scenario list from build_scenario_grid()
        grid1 — DataFrame: current vols stress test
        grid2 — DataFrame: stressed vols stress test
    """
    # Use the most recent date if none specified
    if as_of_date is None:
        as_of_date = spx["date"].max()
    
    # Get the SPX closing level on or before as_of_date
    prior = spx[spx["date"] <= as_of_date]
    if prior.empty:
        raise ValueError(f"No SPX data on or before {as_of_date.date()}")
    
    spx_level = float(prior.iloc[-1]["close"])
    actual_date = prior.iloc[-1]["date"]

    # Build the scenario grid (same for both grids)
    scenarios = build_scenario_grid(max_shock, shock_steps)

    # Grid 1: current vols — vol_multiplier = 1.0 (no vol change)
    grid1 = apply_stress(
        spx_level = spx_level, 
        scenarios = scenarios, 
        portfolio_value = portfolio_value, 
        beta = beta, 
        vol_multiplier = 1.0, 
        option_positions = option_positions,  
    )

    # Grid 2: stressed vols — vol_multiplier = user-specified (default 2.0)
    # For a pure index this produces identical numbers to Grid 1.
    grid2 = apply_stress(
        spx_level = spx_level, 
        scenarios = scenarios, 
        portfolio_value = portfolio_value, 
        beta = beta, 
        vol_multiplier = vol_multiplier, 
        option_positions = option_positions,  
    )

    print(f"[Stress Test] as of {actual_date.date()}"
          f"| SPX: {spx_level:,.2f}" 
          f" | NAV: ${portfolio_value:,.0f}"
          f" | Beta: {beta}")
    
    return {
        "as_of_date": actual_date,
        "spx_level": spx_level,
        "portfolio_value": portfolio_value,
        "beta": beta,
        "vol_multiplier": vol_multiplier,
        "scenarios": scenarios,
        "grid1": grid1,
        "grid2": grid2,
    }

# ======================================================
# Section 4 — Print Report
# ======================================================
def print_stress_report(result: dict) -> None:
    """
    Print both stress test grids. 
    """
    def print_grid(grid: pd.DataFrame, title: str) -> None:
        """Print a single stress test grid."""
        print(f"\n{title}")
        print(f"{'-' * 85}")
        print(f"{'Scenario':<10} {'Shock':>7}  {'SPX Current':>12}"
              f"{'SPX Stressed':>13}  {'P&L %':>8}  "
              f"{'P&L ($)':>13}  {'Position ($)':>14}")
        print(f"{'-' * 85}")

        for _, row in grid.iterrows():
            shock = row["shock_pct"]
            sign = "-" if shock < 0 else ("+" if shock > 0 else " ")

            print(f"{row['rm_code']:<10}"
                  f"{sign}{abs(shock):>5.0%}  "
                  f"{row['spx_current']:>12,.2f}"
                  f"{row['spx_stressed']:>13,.2f}"
                  f"{row['pl_pct']:>+8.2%}"
                  f"{row['pl_dollars']:>+13,.0f}"
                  f"  {row['position_value']:>14,.0f}")
        print(f"{'-' * 85}")
    
    def print_greeks(grid: pd.DataFrame) -> None:
        """Print the Greeks table for the base case (RM 00 — zero move)."""
        base = grid[grid["rm_code"] == "RM00"].iloc[0]
        print(f"\nGreeks (base case RM00 — zero market move)")
        print(f"{'Delta  ($/SPX point)':<35} {base['delta']:>12,.2f}")
        print(f"{'Gamma  (delta change per SPX point)':<35} {base['gamma']:>12,.4f}")
        print(f"{'Vega   ($/vol point)':<35} {base['vega']:>12,.2f}")
        print(f"{'Theta  ($/day)':<35} {base['theta']:>12,.2f}")
        print(f"{'Rho    ($/1% rate move)':<35} {base['rho']:>12,.2f}")
        print(f"\nNote: Gamma, Vega, Theta, Rho are zero for a pure")
        print(f"index portfolio. They become non-zero when the portfolio")
        print(f"holds options — as seen in the EDGE, CAOS, ATTR funds.")
    
    # --- Header ---
    print("\n" + "=" * 85)
    print("STRESS TEST RESULTS")
    print("=" * 85)
    print(f"\nValuation date   : {result['as_of_date'].date()}")
    print(f"SPX level: {result['spx_level']:>12,.2f}")
    print(f"Portfolio NAV: ${result['portfolio_value']:>12,.0f}")
    print(f"Beta: {result['beta']:>12.2f}")
    print(f"Vol multiplier: {result['vol_multiplier']:>12.1f}*  "
          f"(Grid 2 only)")
    
    # --- Grid 1 ---
    print_grid(result["grid1"], "GRID 1 — CURRENT VOLS  (market move only)")

    # --- Grid 2 ---
    print_grid(result["grid2"], 
               f"GRID 2 — STRESSED VOLS "
               f"market move + {result['vol_multiplier']:.0f}*vol")
    
    # --- Greeks ---
    print_greeks(result["grid1"])

    # --- Interpretation ---
    print(f"\nINTERPRETATION")
    g1 = result["grid1"]

    worst = g1.loc[g1["pl_dollars"].idxmin()]
    best = g1.loc[g1["pl_dollars"].idxmax()]

    print(f"Worst scenario ({worst['rm_code']} {worst['shock_pct']:+.0%}): "
          f"P&L = ${worst['pl_dollars']:>+,.0f}  "
          f"({worst['pl_pct']:+.2%} of NAV)")
    print(f"Best  scenario ({best['rm_code']}  {best['shock_pct']:+.0%}): "
          f"P&L = ${best['pl_dollars']:>+,.0f}  "
          f"({best['pl_pct']:+.2%} of NAV)")
    print("\n" + "=" * 85)
    print("  Stress test complete.")
    print("=" * 85)


# ======================================================
# User Input
# ======================================================
def prompt_stress_parameters() -> tuple:
    """
    Prompt the user for stress test specific parameters. 
    Pressing enter keeps the default value.

    Returns: 
        (portfolio_value, beta, vol_multiplier, max_shock, shock_steps)
    """
    print("\n" + "=" * 65)
    print("STRESS TEST PARAMETERS")
    print("Press Enter to keep the default value.")
    print("=" * 65)

    # Portfolio value ---
    raw = input(f"\nPortfolio NAV in dollars  "
                f"[default = ${default_portfolio_value:,.0f}] : ").strip()
    portfolio_value = default_portfolio_value
    if raw != "":
        try:
            v = float(raw.replace(",", ""))
            portfolio_value = v if v > 0 else default_portfolio_value
            if v <= 0:
                print("Must be positive. Using default.")
        except ValueError:
            print("Not a valid number. Using default.")
    
    # --- Beta ---
    raw = input(f"Portfolio beta "
                f"[default = {default_beta}]: ").strip()
    beta = default_beta
    if raw != "":
        try:
            beta = float(raw)
        except ValueError:
            print("Not a valid number. using default.")
            beta = default_beta
    
    # --- Vol multiplier ---
    raw = input(f"Vol multiplier (Grid 2) "
                f"[default = {default_vol_multiplier}] : ").strip()
    vol_multiplier = default_vol_multiplier
    if raw != "":
        try:
            v = float(raw)
            vol_multiplier = v if v >= 1.0 else default_vol_multiplier
            if v < 1.0:
                print("Must be >= 1.0. Using default.")
        except ValueError:
            print("Not a valid number. Using default.")    
    
    # --- Max shock ---
    raw = input(f"Max shock (decimal)  "
                f"[default = {default_max_shock}] : ").strip()
    max_shock = default_max_shock
    if raw != "":
        try:
            v = float(raw)
            max_shock = v if 0.05 <= v <= 1.0 else default_max_shock
            if not (0.05 <= v <= 1.0):
                print("Out of range (0.05–1.0). Using default.")
        except ValueError:
            print("Not a valid number. Using default.")
    
    # --- Shock steps ---
    raw = input(f"Step size (decimal) "
                f"[default = {default_shock_steps}] : ").strip()
    shock_steps = default_shock_steps
    if raw != "":
        try:
            v = float(raw)
            shock_steps = v if 0.01 <= v <= 0.20 else default_shock_steps
            if not (0.01 <= v <= 0.20):
                print("Out of range (0.01–0.20). Using default.")
        except ValueError:
            print("Not a valid number. Using default.")
    
    print(f"\nParameters confirmed:")
    print(f"NAV: ${portfolio_value:>15,.0f}")
    print(f"Beta: {beta:>15.2f}")
    print(f"Vol multiplier: {vol_multiplier:>15.1f}×")
    print(f"Max shock: {max_shock:>15.0%}")
    print(f"Step size: {shock_steps:>15.0%}")

    return portfolio_value, beta, vol_multiplier, max_shock, shock_steps

# ======================================================
# Entry Point
# ======================================================
if __name__ == "__main__":

    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, csv_filename)

    spx = load_spx(csv_path)
    print(f"\nSPX data loaded. Most recent date: {spx['date'].max().date()}")

    (portfolio_value, beta,
     vol_multiplier, max_shock,
     shock_steps) = prompt_stress_parameters()
    
    result = compute_stress_test(
        spx,
        portfolio_value = portfolio_value,
        beta = beta,
        vol_multiplier = vol_multiplier,
        max_shock = max_shock,
        shock_steps = shock_steps,
    )
    print_stress_report(result)




        


