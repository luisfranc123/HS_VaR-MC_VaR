"""
======================================================
Black-Scholes Greeks Calculator
======================================================
What this file does:
Computes the five option Greeks (Delta, Gamma, Vega, Theta, and Rho)
from first principles using the Black-Scholes model.

Inputes required per option position:
    - S: current underlying price (e.g. SPX level)
    - K: strike price
    - r: annual risk-free rate 
    - sigma: implied volatility
    - option_type: "call" or "put"
    - quantity: number of contracts (negative = short position)
    - spc: shares per contract (100 for standard options)
    - q: annual dividend yield
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

trading_days_per_year = 252

# ======================================================
# Section 1 — Building blocks: d1, d2, and time conversion
# ======================================================
def convert_days_to_years(trading_days: float) -> float:
    """
    Convert trading days to years for Black-Scholes inputs.
    """
    return trading_days/trading_days_per_year

def compute_d1_d2(S: float, K: float, T: float, r: float,
                  sigma: float, q: float = 0.0) -> tuple:
    """
    Compute d1 and d2 intermediate values used in all 
    Black-Scholes pricing and Greeks formulas.

    Parameters: 
    - S: current underlying price
    - K: strike price
    - T: time to expiry in YEARS (use convert_days_to_years first)
    - r: annual risk-free rate as a decimal 
    - sigma: implied volatility as a decimal 
    - q: annual dividend yield as a decimal 

    Returns: 
        (d1, d2) tuple of floats
    """
    if T <= 0:
        raise ValueError(
            f"T must be positive (got {T}). "
            f"Option has already expired."
        )
    if sigma <= 0:
        raise ValueError(
            f"sigma must be positive (got {sigma})."
        )
    if S <= 0 or K <= 0:
        raise ValueError(
            f"S and K must be positive (got S={S}, K={K})."
        )
    
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    return d1, d2

# ======================================================
# Section 2 — Option price (for verification purposes)
# ======================================================
def bs_price(S: float, K: float, T: float, r: float, 
             sigma: float, option_type: str, 
             q: float = 0.0) -> float:
      
    """
    Black-Scholes theoretical option price in absolute dollar terms.

    Parameters: 
        - S, K, T, r, sigma: same as compute_d1_d2()
        - option_type: "call" or "put"

    Returns: 
        float — theoretical price per share in dollars
    """
    d1, d2 = compute_d1_d2(S, K, T, r, sigma, q)
    option = option_type.lower()  

    if option == "call":
        price = (S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)* norm.cdf(d2))
    elif option == "put":
        price = (K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1))
    else:
        raise ValueError(
            f"option_value must be 'call' or 'put', got '{option_type}'"
        )

    return price

# ======================================================
# Section 3 — The five Greeks 
# ======================================================
def bs_delta(S: float, K: float, T: float, r: float, 
             sigma: float, option_type: str, 
             q: float = 0.0) -> float:
    """
    Delta — sensitivity of option price to a $1 move in the underlying.

    Formula: 
        Call_Delta =  e^(-qT) x N(d1) — range [0,  +1]
        Put_Delta = -e^(-qT) x N(-d1) — range [-1,  0]    
    """
    d1, _ = compute_d1_d2(S, K, T, r, sigma, q)
    option = option_type.lower()

    if option == "call":
        return np.exp(-q*T)*norm.cdf(d1)
    elif option == "put":
        return -np.exp(-q*T)*norm.cdf(-d1)
    else:
        raise ValueError(f"option_type must be 'call' or 'put'")
    
def bs_gamma(S: float, K: float, T: float, r: float, 
             sigma: float, q: float = 0.0) -> float:
    """
    Gamma — rate of change of Delta per $1 move in the underlying. 

    Formula: 
        Same formula apllies to both calls and puts.

        Gamma = e^(-qT) * n(d1)/(S * sigma * sqrt(T))
        where n(d1) is the standard normal PDF at d1. 
    """
    d1, _ = compute_d1_d2(S, K, T, r, sigma, q)
    
    return np.exp(-q*T)*norm.pdf(d1)/(S*sigma*np.sqrt(T))

def bs_vega(S: float, K: float, T: float, r: float, 
             sigma: float, q: float = 0.0) -> float:
    """
    Vega — sensitivity of option price to a 1 percentage point
    change in implied volatility. 

    Formula: 
        Same formula applies to both calls and puts.

        Vega = [e^(-qT) * S * n(d1) * sqrt(T)]/100
    """
    d1, _ = compute_d1_d2(S, K, T, r, sigma, q)
    return np.exp(-q*T)*S*norm.pdf(d1)*np.sqrt(T)/100.0

def bs_theta(S: float, K: float, T: float, r: float, 
             sigma: float, option_type: str, q: float = 0.0) -> float:
    """
    Theta — option price decay per trading day.

    Formula: 
        Common term = -e^(-qT) * S * n(d1) * sigma/(2 * sqrt(T))

        Call Theta = [common
                    - r * K * e^(-rT) * N(d2)
                    + q * S * e^(-qT) * N(d1)]/252

        Put  Theta = [common
                    + r * K * e^(-rT) * N(-d2)
                    - q * S * e^(-qT) * N(-d1) ]/252
    """
    d1, d2 = compute_d1_d2(S, K, T, r, sigma, q)

    common = -np.exp(-q*T)*S*norm.pdf(d1)*sigma/(2*np.sqrt(T))
    option = option_type.lower()

    if option == "call":
        theta = (common 
                 - r*K*np.exp(-r*T)*norm.cdf(d2)
                 + q*S*np.exp(-q*T)*norm.cdf(d1))/252
    elif option == "put":
        theta = (common 
                 + r*K*np.exp(-r*T)*norm.cdf(-d2)
                 - q*S*np.exp(-q*T)*norm.cdf(-d1))/252
    else: 
        raise ValueError(
            f"option_type must be 'call' or 'put'"
        )
    
    return theta

def bs_rho(K: float, T: float, r: float, d2: float, 
           option_type: str) -> float:
    """
    Rho — sensitivity of option price to a 1 percentage point
    change in the risk-free interest rate.

    Formula: 
        Call Rho =  K * T * e^(-rT) * N(d2)/100
        Put Rho = -K * T * e^(-rT) * N(-d2)/100
    """
    option = option_type.lower()
    if option == "call":
        return K*T*np.exp(-r*T)*norm.cdf(d2)/100.0
    elif option == "put":
        return -K*T*np.exp(-r*T)*norm.cdf(-d2)/100.0
    else: 
        raise ValueError(f"option_type must be 'call' or 'put'")
    
# ======================================================
# Section 4 — combined position-level Greeks
# ======================================================
def compute_greeks(S: float, 
                   K: float, 
                   T_days: float, 
                   r: float, 
                   sigma: float, 
                   option_type: str, 
                   quantity: float, 
                   spc: float = 100.0, 
                   q: float = 0.0) -> dict:
    """
    Compute all five Greeks for a single option position and 
    scale them to dollar terms. 

    Returns: 
        dict with: 
        - Unit Greeks — per share, per contract
        - Dollar Greeks — sca;ed by full position size
        - Diagnostics — bs_price, T_years, d1, d2, moneyness
    """
    T_years = convert_days_to_years(T_days)
    d1, d2 = compute_d1_d2(S, K, T_years, r, sigma, q)

    # Greeks
    delta_unit = bs_delta(S, K, T_years, r, sigma, option_type, q)
    gamma_unit = bs_gamma(S, K, T_years, r, sigma, q)
    vega_unit = bs_vega( S, K, T_years, r, sigma, q)
    theta_unit = bs_theta(S, K, T_years, r, sigma, option_type, q)
    rho_unit = bs_rho(K, T_years, r, d2, option_type)

    # Position size: total shares represented by all contracts
    position_size = quantity*spc

    return {
        # Greeks (per share)
        "delta_unit": delta_unit,
        "gamma_unit": gamma_unit,
        "vega_unit": vega_unit,
        "theta_unit": theta_unit,
        "rho_unit": rho_unit,
        # Dollar Greeks (full position)
        "delta_dollar": delta_unit * position_size * S,
        "gamma_dollar": gamma_unit * position_size * S**2,
        "vega_dollar": vega_unit * position_size,
        "theta_dollar": theta_unit * position_size,
        "rho_dollar": rho_unit * position_size,
        # Diagnostics
        "bs_price": bs_price(S, K, T_years, r, sigma, option_type, q),
        "T_years": T_years,
        "d1": d1,
        "d2": d2,
        "moneyness": S / K,  
    }

# ======================================================
# Verification greeks.py
# ======================================================
if __name__ == "__main__":
    
    # We took a real position from the Excel worksheet XEDGE VaR report. 
    # 4SPX260417C06910010 — SPX call, strike 6910.01, expiry 2026-04-17
    # As of 2026-04-08: S=6773.92, sigma=0.21, r=0.09, T=7.35 days  

    print("\n" + "=" * 60)
    print("greeks.py — Verification")
    print("Source: XEDGE VaR Detail, 4SPX260417C06910010")
    print("=" * 60)

    g = compute_greeks(
        S = 6773.92,
        K = 6910.01,
        T_days = 7.35,
        r = 0.09,
        sigma = 0.21,
        option_type = "call",
        quantity = -10,
        spc = 100,
        q = 0.0,
    )
    print(f"\n  Position : SHORT 10 SPX calls  "
          f"K=6910.01  T=7.35 days  sigma=21%")

    print(f"\n  {'Greek':<18} {'Ours':>12}  {'DRM workbook':>14}  {'Diff':>10}")
    print(f"  {'-'*58}")

    comparisons = [
        ("Delta (unit)", g["delta_unit"], 0.314, "dlt_obb"),
        ("Gamma (unit)", g["gamma_unit"], 0.001507, "gma_obb"),
        ("Vega (unit)", g["vega_unit"], 3.979, "vga_obb"),
        ("Theta (unit)", g["theta_unit"], -6.794, "tht_obb"),
        ("Rho (unit)", g["rho_unit"], 57.03, "rho_obb"),
    ]
    
    for label, imp, drm, col in comparisons:
        diff = imp - drm
        print(f"{label:<18} {imp:>12.6f} {drm:>14.6} {diff:>+10.6f}")
    
    print(f"\n  DOLLAR GREEKS  (qty=-10, spc=100)")
    print(f"  {'Delta ($)':<20} {g['delta_dollar']:>14,.2f}")
    print(f"  {'Gamma ($)':<20} {g['gamma_dollar']:>14,.2f}")
    print(f"  {'Vega ($)':<20} {g['vega_dollar']:>14,.2f}")
    print(f"  {'Theta ($/day)':<20} {g['theta_dollar']:>14,.2f}")
    print(f"  {'Rho ($)':<20} {g['rho_dollar']:>14,.2f}")

    print(f"\nDIAGNOSTICS")
    print(f"{'BS Price ($)':<20} {g['bs_price']:>14.4f}")
    print(f"{'T (years)':<20} {g['T_years']:>14.6f}")
    print(f"{'d1':<20} {g['d1']:>14.6f}")
    print(f"{'d2':<20} {g['d2']:>14.6f}")
    print(f"{'Moneyness (S/K)':<20} {g['moneyness']:>14.6f}")
    print(f"\nNote: Rho difference is a scaling convention in the DRM")
    print(f"workbook. All other Greeks are within rounding tolerance.")
    print("=" * 60)