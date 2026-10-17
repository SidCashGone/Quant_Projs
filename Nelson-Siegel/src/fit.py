#import dependencies
import re
import pandas as pd
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from ns_model import nelson_siegel, calibrate

"""Import data"""

df = pd.read_csv('Nelson-Siegel/data/daily-treasury-rates-2025.csv')
print(df.head())
print(df.columns.tolist())

"""
Data loading: Daily Treasury Par Yield Curve Rates
"""
def header_to_years(col):
    """'1 Mo' -> 1/12, '1.5 Month' -> 1.5/12, '10 Yr' -> 10.0; None if not a maturity."""
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(mo|month|yr|year)s?\s*", str(col), flags=re.I)
    if not m:
        return None
    n, unit = float(m.group(1)), m.group(2).lower()
    return n / 12 if unit.startswith("mo") else n


def load_par_yield_curve(path):
    """Load the CSV & format:
    
    Returns a df with dates=index, maturities (in years) as col,
    values = dec
    """
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.set_index("Date").sort_index()

    mapping = {c: header_to_years(c) for c in df.columns}
    mapping = {c: y for c, y in mapping.items() if y is not None}
    if not mapping:
        raise ValueError(f"No maturity columns recognized in {df.columns.tolist()}")
    sub = df[list(mapping)].apply(pd.to_numeric, errors="coerce") / 100.0
    sub = sub.rename(columns=mapping)

    return sub[sorted(sub.columns)]

def get_curve(df, date):
    """Return (tau, y_obs) for one date, drop NaNs"""
    row = df.loc[date]
    mask = row.notna().values
    tau, y = row.index.values[mask].astype(float), row.values[mask]
    if len(tau) < 4:
        return ValueError(f"Only {len(tau)} points for {date}: need at least 4 to fit 4 parameters.")
    return tau, y 

#FITTING

def solve_betas(tau, y_obs, lambd):
    """Use linear least squares to solve for b0, b1, b2"""
    f0, f1, f2 = calibrate(tau, lambd)
    X = np.column_stack([f0, f1, f2])
    betas, *_ = np.linalg.lstsq(X, y_obs, rcond=None)
    return betas


"""
sanity check to ensure parameters fit
"""
def fit_ns(tau, y_obs):
    """Fit N_S parameters; if none, then estimate it"""
    tau = np.asarray(tau, dtype=float)
    y_obs = np.asarray(y_obs, dtype=float)

    def sse(lambd):
        if lambd <= 1e-6:
            return 1e12
        b0, b1, b2 = solve_betas(tau, y_obs, lambd)
        resid = nelson_siegel(tau, b0, b1, b2, lambd) - y_obs
        return np.sum(resid ** 2)

    res = minimize_scalar(sse, bounds=(0.01, 5.0), method='bounded')
    lambd = res.x
    b0, b1, b2 = solve_betas(tau, y_obs, lambd)
    return b0, b1, b2, lambd


