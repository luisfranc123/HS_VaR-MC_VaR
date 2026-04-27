#======================================================
# Load and Prepare the SPX data
#======================================================
import pandas as pd 
import numpy as np
import os

csv_filename = "spxtr_level_data.csv"
#------------------------------------------------------
# Main Function
#------------------------------------------------------
def load_spx(csv_path: str) -> pd.DataFrame:
    """
    Load the SPX csv and return aclean DataFrame with computed daily returns

    Parameters:
        csv_path: str — full path to the csv file. 

    Returns:
        - date (datetime) — trading date
        - close (float) — SPX total-return index closing level
        - daily return (float) — (close_t/close_{t-1}) - 1
    """
    raw = pd.read_csv(csv_path, 
                     header = None, 
                     names = ["date", "open", "high", "low", "close"], )
    # Create a copy of the original file
    raw = raw.iloc[1:].copy()

    # Convert data types
    raw["date"] = pd.to_datetime(raw["date"], errors = "coerce")
    raw["close"] = pd.to_numeric(raw["close"], errors = "coerce")

    # Drop values with missing critical values
    n_before = len(raw)
    raw = raw.dropna(subset = ["date", "close"])
    n_dropped = n_before - len(raw)

    if n_dropped > 0:
        print(f"[Warning] Dropped {n_dropped} rows with unparseable date or close.")

    # Sort data chronologically
    raw = raw.sort_values("date").reset_index(drop = True)

    ## Compute daily Retruns ##
    raw["daily_return"] = np.log(1 + raw["close"].pct_change())
    raw = raw.dropna(subset = ["daily_return"]).reset_index(drop = True)

    spx = raw[["date", "close", "daily_return"]].copy()

    return spx
    