#======================================================
# The Rolling 3-year window and outlier filter
#======================================================
import pandas as pd
import numpy as np
import os
import sys

# Import 
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_data import load_spx

csv_filename = "spxtr_level_data.csv"

#------------------------------------------------------
# Configuration
#------------------------------------------------------
default_lookback_years = 3
default_outlier_cutoff = 0.035
default_MC_draws = 944

#------------------------------------------------------
# Function 1 - Extract the rolling window
#------------------------------------------------------
def get_window(spx: pd.DataFrame, 
              as_of_date: pd.Timestamp, 
              lookback_years: int = default_lookback_years) -> pd.Series:
    """
    Extract all daily returns in the trailing lookback_years window.

    Parameters:
        - spx: full SPX DataFrame from load_spx()
        - as_of_date: the calculation date
        - lookback_years: hoe many years back the window extends 

    Returns:
        pd.Series of daily returns within the window, crhnologically sorted (oldest first)
    """
    window_start = as_of_date - pd.DateOffset(years = lookback_years)
    mask = (spx["date"] > window_start) & (spx["date"] <= as_of_date)
    
    return spx.loc[mask, "daily_return"].reset_index(drop = True)

#------------------------------------------------------
# Function 2 - Apply the Outlier Filter
#------------------------------------------------------
def apply_outlier_filter(window: pd.Series, 
                        outlier_cutoff: float = default_outlier_cutoff) -> pd.Series:
    """
    Remove returns whose absolute value exceeds outlier_cutoff.

    Parameters:
        - window: full return window from get_window()
        - outlier_cutoff: threshold in decimal form 

    Returns: 
        pd.Series — filtered pool ready for Monte Carlo sampling.
    """
    return window[window.abs() <= outlier_cutoff].reset_index(drop = True)

#------------------------------------------------------
# Function 3 - Full Diagnostic Picture for one date
#------------------------------------------------------
def describe_window(spx: pd.DataFrame, 
                    as_of_date: pd.Timestamp, 
                    lookback_years: int = default_lookback_years, 
                    outlier_cutoff: float = default_outlier_cutoff, 
                    mc_draws: int = default_MC_draws) -> dict:

    """
    Compute the full diagnostic picture for a single calculation date.

    Parameters:
        - spx: full SPX DataFrame from load_spx()
        - as_of_date: the calculation date
        - lookback_years: hoe many years back the window extends 
        - outlier_cutoff: outlier threshold in decimal form
        mc_draws: number of Monte Carlo draws 

    Returns:
        dict with all window statistics and the two Series objects 
        (full_window and filtered_pool).
    """
    full_window = get_window(spx, as_of_date, lookback_years)
    filtered_pool = apply_outlier_filter(full_window, outlier_cutoff)

    outliers_down = full_window[full_window < -outlier_cutoff]
    outliers_up = full_window[full_window > outlier_cutoff]
    
    # return
    result = {
        "as_of_date": as_of_date,
        "window_start": as_of_date - pd.DateOffset(years = lookback_years),
        "lookback_years": lookback_years,
        "outlier_cutoff": outlier_cutoff,
        "mc_draws": mc_draws,
        "n_full_window": len(full_window),
        "n_outliers_down": len(outliers_down),
        "n_outliers_up": len(outliers_up),
        "n_filtered_pool": len(filtered_pool),
        "outlier_pct": len(full_window[full_window.abs() > outlier_cutoff]) / len(full_window),
        "full_min": full_window.min(),
        "full_max": full_window.max(),
        "full_mean": full_window.mean(),
        "full_std": full_window.std(),
        "filtered_min": filtered_pool.min(),
        "filtered_max": filtered_pool.max(),
        "filtered_mean": filtered_pool.mean(),
        "filtered_std": filtered_pool.std(),
        "left_tail": sorted(full_window.nsmallest(10).tolist()),
        "outlier_values_down": sorted(outliers_down.tolist()),
        "outlier_values_up": sorted(outliers_up.tolist(), reverse=True),
        "full_window": full_window,
        "filtered_pool": filtered_pool,
    }

    return result

#------------------------------------------------------
# Print test report
#------------------------------------------------------

def print_window_report(d: dict) -> None:
    """Print a test report"""
    
    print("=" * 65)
    print("  STEP 2 — ROLLING WINDOW REPORT")
    print("=" * 65)
    print(f"\n  Calculation date  : {d['as_of_date'].date()}")
    print(f"Window start: {d['window_start'].date()}")
    print(f"Lookback: {d['lookback_years']} year(s)")
    print(f"Outlier cutoff: ± {d['outlier_cutoff']:.1%}")
    print(f"MC draws: {d['mc_draws']:,}")
 
    print(f"\n WINDOW COUNTS")
    print(f"{'Full window (all returns)':<35} {d['n_full_window']:>6,} days")
    print(f"{'Outliers down (< -{:.1%})':<35}".format(d['outlier_cutoff']) +
          f"{d['n_outliers_down']:>6,}")
    print(f"{'Outliers up (> +{:.1%})':<35}".format(d['outlier_cutoff']) +
          f"{d['n_outliers_up']:>6,}")
    print(f"{'Outlier % of window':<35} {d['outlier_pct']:>6.3%}")
    print(f"{'Filtered pool (for MC)':<35} {d['n_filtered_pool']:>6,} days")
 
    print(f"\nFULL WINDOW STATISTICS  (includes outliers)")
    print(f"{'Range':<35} [{d['full_min']:.5f},  {d['full_max']:.5f}]")
    print(f"{'Mean':<35} {d['full_mean']:.7f}")
    print(f"{'Std deviation':<35} {d['full_std']:.7f}")
 
    print(f"\nFILTERED POOL STATISTICS  (outliers removed)")
    print(f"{'Range':<35} [{d['filtered_min']:.5f},  {d['filtered_max']:.5f}]")
    print(f"{'Mean':<35} {d['filtered_mean']:.7f}")
    print(f"{'Std deviation':<35} {d['filtered_std']:.7f}")
 
    print(f"\nLEFT TAIL — 10 worst returns in full window")
    print(f"{'  '.join(f'{v:.5f}' for v in d['left_tail'])}")
 
    if d['outlier_values_down']:
        print(f"\n  OUTLIERS EXCLUDED FROM MC POOL")
        print(f"Down: {'  '.join(f'{v:.5f}' for v in d['outlier_values_down'])}")
    if d['outlier_values_up']:
        print(f"Up: {'  '.join(f'{v:.5f}' for v in d['outlier_values_up'])}")
 
    print("\n" + "=" * 65)
    print("Step 2 complete. Window and filtered pool ready for Step 3.")
    print("=" * 65)

#------------------------------------------------------
# User input
#------------------------------------------------------

def prompt_parameters() -> tuple:
    """
    Ask the user to enter the three key parameters interactively. 
    Pressing enter without typing anything keeps the default inputs.

    Returns:
        (lookback_years, outlier_cutoff, mc_draws)
    """
    print("\n" + "=" * 65)
    print("PARAMETER CONFIGURATION")
    print("Press Enter to keep the default (DRM system) value.")
    print("=" * 65)

    # --- Lookback years ---
    raw = int(input(f"\nLookback window in years [default = {default_lookback_years}]: ").strip())
    
    if raw == "":
        lookback_years = default_lookback_years
    else:
        try:
            lookback_years = raw
            if lookback_years < 1 or lookback_years > 20:
                print("Out of range (1-20). Using default")
                lookback_years = default_lookback_years
        except ValueError:
            print("Not a valid integer. Using default")
            lookback_years = default_lookback_years

    # --- Outlier cutoff ---
    raw = float(input(f"\nOutlier cutoff (decimal) [default = {default_outlier_cutoff}]: ").strip())
    if raw == "":
        outlier_cutoff = default_outlier_cutoff
    else:
        try:
            outlier_cutoff = raw
            if outlier_cutoff < 0.01 or outlier_cutoff > 0.20:
                print("Out of range (0.01 - 0.20). Using default.")
                outlier_cutoff = default_outlier_cutoff
        except ValueError:
            print("Not a valid number. Using default.")
            outlier_cutoff = default_outlier_cutoff

    # --- MC draws ---
    raw = int(input(f"\nMonte Carlo draws [default = {default_MC_draws}]: ").strip())
    if raw == "":
        mc_draws = default_MC_draws
    else:
        try:
            mc_draws = raw
            if mc_draws < 100 or mc_draws > 100_000:
                print("Out of range (100 - 100_000). Using default.")
                mc_draws = default_MC_draws
        except ValueError:
            print("Not a valid number. Using default.")
            mc_draws = default_MC_draws

    print(f"\nRunning with: lookback = {lookback_years}yr "
         f"cutoff = {outlier_cutoff:.1%}  draws = {mc_draws:,}")

    return lookback_years, outlier_cutoff, mc_draws


#------------------------------------------------------
# Entry Point
#------------------------------------------------------
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, csv_filename)

    spx = load_spx(csv_path)
    latest_date = spx["date"].max()

    # Ask the user for input parameters
    lookback_years, outlier_cutoff, mc_draws = prompt_parameters()

    # Run the analysis with those parameters
    d = describe_window(spx, latest_date, lookback_years, outlier_cutoff, mc_draws)
    print_window_report(d)