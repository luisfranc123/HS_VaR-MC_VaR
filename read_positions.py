"""
======================================================
Read Positions from DRM Workbook
======================================================
What this file does:
Opens a DRM workbook and extracts two things from the 
VaR detail sheet:

    1. Base posisitions (VAR000) — the fund's option holdings
    at today's prices, ready for compute_greeks()

    2. Account metadata — from the VaR MC Returns sheet:
    account name, fund NAV, run date, and the DRM
    system's own VaR numbers (used for comparison)
"""
import os
import sys
import re
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

#======================================================
# Constants 
#======================================================
# Underlying whose positions we reprice under each scenario.
# SPY moves approximately as SPX/10.029 — the DRM system
# stores  the repriced SPY hedgeprice directly in each scenario
# row, so we use hedgeprice as S for SPY positions. 

reprice_underlyings = {"spx", "xsp", "spy"}

# skipped for row — not restricted under SPX scenarios
skip_underlyings = {"mags"}

# Worksheets subfolder name — DRM files live here
worksheets_folder = "Worksheets"

# ======================================================
# Helpers
# ======================================================

def _safe_float(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None

def _is_numeric(v) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False

#======================================================
# 1. Account discovery
#======================================================
def find_drm_files(folder: str) -> list:
    """
    Scan a folder for DRM workbook files and return a sorted 
    list of account entries.

    Filename pattern expected:
        DRM_X{account}_M_{Date}...xlsx
    
    Parameters:
        folder: path to scan
    
    Returns: 
    list of dicts, each with keys:
        - index — display number 
        - account — parsed account name 
        - filename — filename
        - filepath — full path to the file
    """
    pattern = re.compile(r"DRM_X(.+?)_M_", re.IGNORECASE)
    results = []

    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith(".xlsx"):
            continue
        if not fname.upper().startswith("DRM_"):
            continue
    
        match = pattern.search(fname)
        account = match.group(1).lower() if match else fname

        results.append({
            "account": account, 
            "filename": fname, 
            "filepath": os.path.join(folder, fname), 
        })

    for i, r in enumerate(results, 1):
        r["index"] = i
    
    return results

def prompt_account_selection(folder: str) -> dict | None:
    """
    Display the list of available DRM workbooks and ask the 
    user to pick one

    Parameters: 
        folder: the worksheets subfolder to scan
    
    Returns: 
        dict with keys (account, filename, filepath, index)
        or None if the user chooses SPX-only mode.
    """
    files = find_drm_files(folder)

    if not files:
        print("[Note] No DRM workbooks found in worksheets folder.")
        print("Running in SPX-only mode.")
        return None
    
    print("\n Available Accounts: ")
    print("  " + "─" * 60)
    for f in files:
        print(f"[{f['index']}]   {f['account']:16}  {f['filename']}")
    print(f"[0] SPX index only (no accunt file)")
    print("  " + "─" * 60)

    while True: 
        raw = input(f"\n Select account [0-{len(files)}]: ").strip()
        try:
            choice = int(raw)
            if choice == 0:
                return None
            if 1 <= choice <= len(files):
                selected = files[choice - 1]
                print(f"\n -> Selected: {selected['account']}")
                print(f"\n -> File:       {selected['filename']}")
                return selected
            print(f"Please enter a number between 0 and {len(files)}.")
        except ValueError:
            print(" Not a valid number. Try again.")

#======================================================
# 2. Read base positions (VAR000)
#======================================================
def read_base_positions(filepath: str) -> list:
    """
    Read the VAR000 base case option positions from the 
    VaR Detail sheet.

    VAR000 rows represent the fund's actual holdings at
    today's market prices. These are the inputs to 
    compute_greeks() and to the fund-level VaR repricing.

    Parameters:
        filepath: full path to the DRM workbook
    
        Returns: 
        List of position dicts. Each dict contains: 

        Black-Scholes inputs: 
            symbol — contract identifier
            underlying — "spx", "spy", "xsp"
            option_type — "call" or "put"
            quantity — contracts (negative = short position)
            spc — shares per contract (typically 100)
            S — current underlying price (hedgeprice)
            K — strike price
            T_days — trading days to expiry (obbDaysToExpir)
            r — risk-free rate (annual decimal)
            sigma — implied volatility (annual decimal)
            q — dividend yield (0.0 — not in workbook)
        
            DRM comparison values.
    """
    df = pd.read_excel(filepath, sheet_name = "VaR Detail", header = 0)
    df.replace("\\N", np.nan, inplace = True)

    base = df[df["rmid"] == "VAR0000"].copy()  
    positions = []

    for _, row in base.iterrows():
        symbol = str(row["securitySymbol"])
        sectype = str(row["sectype2"]).upper()
        undsym = str(row["undsym"]).lower()

        # skip cash rows
        if symbol == "$cash" or sectype == "STOCK":
            continue

        # skip underlyings we are not repricing
        if undsym in skip_underlyings:
            continue

        # only process underlying in our recipe set
        if undsym not in reprice_underlyings:
            continue

        # determine option type
        if sectype =="CALL":
            option_type = "call"
        elif sectype == "PUT":
            option_type = "put"
        else:
            continue

        # Use the DRM system's own T_days value directly. 
        # obbDaysToExpir is already in trading days
        t_days = _safe_float(row["obbDaysToExpir"])
        if t_days is None or t_days <= 0:
            continue # skip expire or undated positions

        vfactor = _safe_float(row["VFactor"]) or 1.0
        thv = _safe_float(row["thv"])
        hedgeprice = float(row["hedgeprice"])
        base_price = thv if (vfactor == 2 and thv is not None) else hedgeprice

        positions.append({
            "symbol": symbol,
            "underlying": undsym,
            "option_type": option_type,
            "quantity": float(row["quantity"]),
            "spc": float(row["SPC"]),
            "S": hedgeprice,  
            "base_price": base_price,
            "K": float(row["strikePrice"]),
            "T_days": t_days,
            "r": float(row["rate"]),
            "sigma": float(row["sigma"]),
            "q": 0.0,
            "vfactor": vfactor,
            "drm_thv": thv,
            "drm_delta_dollars": _safe_float(row["DeltaDollars"]),
            "drm_gamma": _safe_float(row["Gamma"]),
            "drm_vega": _safe_float(row["Vega"]),
            "drm_theta": _safe_float(row["Theta"]),
            "drm_rho": _safe_float(row["Rho"]),
            "drm_dlt_obb": _safe_float(row["dlt_obb"]),
            "drm_posn_dollars": _safe_float(row["PosnDollars"]),
        })

    print(f"[Positions] {len(positions)} option positions loaded "
          f"(VAR0000, SPX/SPXW/SPY only, MAGS excluded)")
    return positions

# ======================================================
# 3. Generate scenario SPX levels 
# ======================================================
def generate_scenario_spx_levels(spx: pd.DataFrame, 
                                 base_spx: float, 
                                 n_scenarios: int = 944,
                                 outlier_cutoff: float = 0.035, 
                                 seed: int = 123) -> pd.Series:
    """
    Generate scenario SPX levels by sampling from the FULL 
    SPX return history. 

    Parameters:
        - spx: full SPX DataFrame from load_spx()
        - base_spx: today's SPX closing level
        (scenario_level = base_spx * (1 + return))
        - n_scenarios: number of scenarios to generate (default 944)
        - outlier_cutoff: returns beyond ±this are excluded (default 3.5%)
        - seed: random seed for reproducibility 
    
        Returns: 
            pd.Series of floats, length n_scenarios, sorted ascending 
    """
    # apply outlier filter
    all_returns = spx["daily_return"].dropna()
    filtered_pool = all_returns[all_returns.abs() <= outlier_cutoff].values

    n_pool = len(filtered_pool)
    print(f"[Scenarios] Full history pool: {n_pool:,} returns  "
          f"(range: {filtered_pool.min():.4f} -> {filtered_pool.max():.4f})")
    
    # sample n_scenarios returns with replacement
    rng = np.random.default_rng(seed)
    sampled = rng.choice(filtered_pool, size = n_scenarios, replace = True)

    # convert returns to SPX levels and sort ascending
    levels = np.sort(base_spx*(1 + sampled))

    print(f"[Scenarios] Generated {n_scenarios} scenario SPX levels  "
          f"(range: {levels.min():,.2f} → {levels.max():,.2f})")
    
    return pd.Series(levels)

# ======================================================
# 4. Read account metadata
# ======================================================
def read_account_metadata(filepath: str) -> dict:
    """
    Read summary metadata from VaR MC returns sheet. 

    This gives us the DRM system's own VaR numbers and fund
    information, which we use for comparison and report headers.

    Parameters: 
        filepath: full path to the DRM workbook
    
    Returns: 
    dict with keys:
        account — fund account name (e.g. "xattr2m")
        fund_ticker — short name (e.g. "ATTR")
        calc_date — calculation date
        nav_t0 — fund NAV as of T0
        close_t0 — fund close price as of T0
        drm_fund_var_1d — DRM's T1 fund 1-day VaR
        drm_fund_var_20d — DRM's T1 fund 20-day VaR
        drm_idx_var_1d — DRM's T1 index 1-day VaR
        drm_idx_var_20d — DRM's T1 index 20-day VaR
        drm_var_ratio — DRM's fund/index VaR ratio
        reference_index — the reference index name (e.g. "SPX")
        exception_1d — DRM T0 1-day exception flag
        exception_20d — DRM T1 20-day exception flag
    """
    raw = pd.read_excel(filepath, sheet_name = "VaR MC Returns", header = None)
    raw.replace("\\N", np.nan, inplace = True)

    # Row 1 = labels, Row 2 = values
    # Row 3 = fund info (ticker, NAV)
    labels = raw.iloc[0].tolist()
    values = raw.iloc[2].tolist()
    fund_row = raw.iloc[3].tolist()

    def _get(label_substr):
        """Find column by partial label match and return its value"""
        for i, lbl in enumerate(labels):
            if isinstance(lbl, str) and label_substr.lower() in lbl.lower():
                v = values[i]
                return float(v) if _is_numeric(v) else v
        return None
    
    # Parse referebce index name from row 2 col 0
    idx_label = str(raw.iloc[2, 0]) if pd.notna(raw.iloc[2, 0]) else ""
    idx_match = re.search(r"\[IDX\]=(\w+)", idx_label)
    ref_index = idx_match.group(1) if idx_match else "SPX"

    # Fin ticker from row 3
    fund_ticker = str(fund_row[1]) if pd.notna(fund_row[1]) else ""
    nav_t0 = _safe_float(fund_row[3])
    close_t0 = _safe_float(fund_row[5])

    # calc date from row 1 col 1
    calc_date = str(raw.iloc[1, 1]) if pd.notna(raw.iloc[1, 1]) else ""

    return {
        "account": str(raw.iloc[1, 0]),
        "fund_ticker": fund_ticker,
        "calc_date": calc_date,
        "nav_t0": nav_t0,
        "close_t0": close_t0,
        "drm_fund_var_1d": _get("T1_FundVaR_1Day"),
        "drm_fund_var_20d": _get("T1_FundVaR_20Day"),
        "drm_idx_var_1d": _get("T1_[IDX]VaR_1Day"),
        "drm_idx_var_20d": _get("T1_[IDX]VaR_20Day"),
        "drm_var_ratio": _get("T1_Ratio"),
        "reference_index": ref_index,
        "exception_1d": _get("T0_1Day_Exception"),
        "exception_20d": _get("T1_20Day_Exception"),
    }

# ======================================================
# 5. Master Load Function — called by export.py
# ======================================================
def load_account(filepath: str, 
                 spx: pd.DataFrame, 
                 n_scenarios: int = 944, 
                 outlier_cutoff: float = 0.035, 
                 seed: int = 123) -> dict:
    """
    Load everything needed from a DRM workbook.

    Parameters: 
        filepath: full path to the DRM workbook
    
    Returns: 
    dict with keys:
        metadata — from read_acciount_metadata()
        positions — from read_base_positions()
        scenario_levels — from generate_scenario_spx_levels()
    """
    print(f"\n Loading: {os.path.basename(filepath)}")
    metadata = read_account_metadata(filepath)
    positions = read_base_positions(filepath)
    
    # Read base SPX level from VAR0000 SPX rows
    df_var = pd.read_excel(filepath, sheet_name="VaR Detail", header=0)
    df_var.replace("\\N", np.nan, inplace=True)
    spx_rows = df_var[(df_var["rmid"] == "VAR0000") &
                        (df_var["undsym"].str.lower().isin(["spx", "xsp"]))]
    base_spx = float(spx_rows["hedgeprice"].iloc[0]) \
                if not spx_rows.empty else float(spx["close"].iloc[-1])
    
    scenario_levels = generate_scenario_spx_levels(
        spx, base_spx, n_scenarios, outlier_cutoff, seed 
    )
    
    print(f"\nAccount: {metadata['account']}  "
          f"({metadata['fund_ticker']})")
    print(f"Calc date: {metadata['calc_date']}")
    print(f"NAV T0: {metadata['nav_t0']}")
    print(f"Base SPX: {base_spx:,.2f}")
    if metadata["drm_fund_var_1d"]:
        print(f"DRM VaR (1D fund): "
              f"{metadata['drm_fund_var_1d']:.4%}")

    return {
        "metadata": metadata,
        "positions": positions,
        "scenario_levels": scenario_levels,
        "base_spx": base_spx,
    }

# ======================================================
# Prompt Scenario Parameters
# ======================================================
def prompt_scenario_parameters() -> tuple:
    """
    Ask the user to enter the scenario generation parameters.
    Pressing enterd keeps the default value.

    Returns: 
        (n_scenarios, outlier_cutoff, seed)
    """
    print("\n" + "=" * 65)
    print("  SCENARIO GENERATION PARAMETERS")
    print("  Press Enter to keep the default value.")
    print("=" * 65)

    # --- Number of scenarios ---
    raw = input(f"\n Number of MC scenarios [feault = 944]:  ").strip()
    if raw == "":
        n_scenarios = 944
    else:
        try:
            v = int(raw)
            n_scenarios = v if 100 <= v <= 10_000 else 944
            if not (100 <= v <= 10_000):
                print(" Out of range (100 - 10,000). Using default.")
        except ValueError:
            print(" Not a valid integer. Using default.")
            n_scenarios = 944
    # --- Outlier cutoff ---
    raw = input(f"\n Outlier cutoff for MC darws [feault = 0.035]:  ").strip()
    if raw == "":
        outlier_cutoff = 0.035
    else:
        try:
            v = float(raw)
            outlier_cutoff = v if 0.01 <= v <= 0.20 else 0.035
            if not (0.01 <= v <= 0.20):
                print(" Out of range (0.01 - 0.20). Using default.")
        except ValueError:
            print(" Not a valid number. Using default.")
            outlier_cutoff = 0.035
    
    print(f"\n  Parameters confirmed:")
    print(f"Scenarios: {n_scenarios:,}")
    print(f"Outlier cutoff : ±{outlier_cutoff:.1%}")
    
    return n_scenarios, outlier_cutoff

# ======================================================
# Entry Point
# ======================================================
if __name__ == "__main__":

    from load_data import load_spx

    script_dir = os.path.dirname(os.path.abspath(__file__))
    worksheets_dir = os.path.join(script_dir, worksheets_folder)
    csv_path = os.path.join(script_dir, "spxtr_level_data.csv")

    print("\n" + "=" * 65)
    print("READ POSITIONS — Verification")
    print("=" * 65)

    # load SPX data
    spx = load_spx(csv_path)
    files = find_drm_files(worksheets_dir)

    if not files:
        print(f"\n No DRM found in: {worksheets_dir}")
        sys.exit(1)
    
    # Use the first file found
    data = load_account(files[0]["filepath"], spx)

    print(f"\n METADATA")
    for k, v in data["metadata"].items():
        if v is not None:
            fmt = f"{v:.6f}" if isinstance(v, float) else str(v)
            print(f"  {k:<25} {fmt}")

    print(f"\n  POSITIONS ({len(data['positions'])} total)")
    print(f"  {'Symbol':<25} {'Type':<5} {'Qty':>9} "
          f"{'S':>9} {'K':>9} {'T_days':>7} "
          f"{'sigma':>6}  {'DRM_thv':>12}")
    print(f"  {'-'*92}")
    for p in data["positions"]:
        thv = f"{p['drm_thv']:>12.4f}" if p["drm_thv"] else "         N/A"
        print(f"  {p['symbol']:<25} {p['option_type']:<5} "
              f"{p['quantity']:>9.2f} {p['S']:>9.2f} "
              f"{p['K']:>9.2f} {p['T_days']:>7.2f} "
              f"{p['sigma']:>6.3f}  {thv}")
    
    s = data["scenario_levels"]
    print(f"\n  SCENARIO LEVELS (generated from full history)")
    print(f"  Count: {len(s)}")
    print(f"  Range: {s.min():,.2f}  →  {s.max():,.2f}")
    print(f"  Worst 5: {', '.join(f'{v:,.2f}' for v in s.values[:5])}")
    print(f"  Best 5: {', '.join(f'{v:,.2f}' for v in s.values[-5:])}")

    print("\n" + "=" * 65)
    print("  Verification complete.")
    print("=" * 65)



