"""
Main Volatility Surface Mapper
We will use this to import methods in Heston_func
and use data fetched from Greeks_fetcher to create a Volatility Surface.
"""

#Import dependencies
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import greeks_package as gp
from scipy.optimize import brentq, minimize
from scipy.stats import norm
from tqdm import tqdm

#Import Methods
from Heston_func import HestonParams, heston_call_prices

# ============================================================================
# MARKET DATA FETCHING (Using Greeks_fetcher)
# ============================================================================

def fetch_market_data(ticker="SPY", opt_type="c", max_days=20):
    """
    Fetch live options data with Greeks using Greeks_fetcher.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    opt_type : str
        'c' for calls, 'p' for puts
    max_days : int
        Maximum days to expiry
    
    Returns:
    --------
    df : pandas DataFrame
        Options data with Greeks
    """
    print(f"\nFetching {ticker} options data (max {max_days} days to expiry)...")
    
    # Download options chain
    opts = gp.download_options(ticker, opt_type=opt_type, max_days=max_days)
    
    if opts.empty:
        raise ValueError(f"No options data found for {ticker} with max_days={max_days}")
    
    # Get Greeks for each option
    all_greeks = opts.apply(gp.greeks, axis=1, ticker=ticker)
    
    # Combine with original data
    full_data = opts.join(all_greeks)
    
    # Clean and prepare data
    full_data = clean_market_data(full_data)
    
    print(f"Fetched {len(full_data)} options")
    print(f"Available columns: {full_data.columns.tolist()}")
    
    return full_data

def filter_iv_outliers(df, iv_column='implied_vol', lower_bound=0.01, upper_bound=1.5):
    """
    Filter out implausible implied volatility values.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Market data
    iv_column : str
        Column name for implied volatility
    lower_bound : float
        Minimum plausible IV (1% by default)
    upper_bound : float
        Maximum plausible IV (150% by default)
    
    Returns:
    --------
    df : pandas DataFrame
        Filtered data
    """
    original_count = len(df)
    mask = (df[iv_column] >= lower_bound) & (df[iv_column] <= upper_bound)
    df = df[mask].copy()

    # Remove illiquid quotes (0 bid, huge spread)
    if 'bid' in df.columns and 'ask' in df.columns:
        spread_mask = (df['bid'] > 0) & ((df['ask'] - df['bid']) / df['ask'] < 0.5)
        df = df[spread_mask]

    # Vectorized per-maturity 3-sigma filter (replaces the loop)
    grp = df.groupby('maturity')[iv_column]
    mean_iv = grp.transform('mean')
    std_iv = grp.transform('std').fillna(0)
    count = grp.transform('count')
    within_3sigma = (df[iv_column] >= mean_iv - 3*std_iv) & (df[iv_column] <= mean_iv + 3*std_iv)
    keep = within_3sigma | (count <= 5)
    df = df[keep]

    removed = original_count - len(df)
    if removed > 0:
        print(f"  Removed {removed} IV anomalies ({removed/original_count*100:.1f}%)")
    return df
    
def clean_market_data(df):
    """
    Clean and prepare market data for calibration.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Raw data from Greeks_fetcher
    
    Returns:
    --------
    df : pandas DataFrame
        Cleaned data with standardized columns
    """
    # Create a copy
    df = df.copy()
    
    # Map your actual column names to standard names
    # Your data has: strike, lastPrice, impliedVolatility, Days to Expiry, Delta, Vega, Theta, Gamma
    column_mapping = {
        'strike': 'strike',
        'lastPrice': 'market_price',
        'bid': 'bid',
        'ask': 'ask',
        'impliedVolatility': 'implied_vol',
        'Days to Expiry': 'days_to_expiry',
        'Delta': 'delta',
        'Gamma': 'gamma',
        'Vega': 'vega',
        'Theta': 'theta',
        'Rho': 'rho_greek',  # Renamed to avoid confusion with Heston's rho
        'expiry': 'expiry_date'
    }
    
    # Apply renaming (only for columns that exist)
    for old, new in column_mapping.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
    
    # Calculate maturity in years
    if 'days_to_expiry' in df.columns:
        df['maturity'] = df['days_to_expiry'] / 365.0
    
    # If days_to_expiry not available, calculate from expiry date
    elif 'expiry_date' in df.columns:
        df['expiry_date'] = pd.to_datetime(df['expiry_date'])
        df['days_to_expiry'] = (df['expiry_date'] - datetime.now()).dt.days
        df['maturity'] = df['days_to_expiry'] / 365.0
    
    # Ensure numeric types
    numeric_cols = ['strike', 'market_price', 'bid', 'ask', 'delta', 'gamma', 
                    'vega', 'theta', 'implied_vol', 'maturity', 'days_to_expiry']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove invalid rows
    df = df.dropna(subset=['strike', 'maturity', 'market_price'])
    
    # Remove options with zero or negative prices
    df = df[df['market_price'] > 0.05]
    
    # Remove options with very short or long maturities
    df = df[(df['maturity'] >= 0.05) & (df['maturity'] <= 2.0)]

    #Filter IV Anomalies
    if 'implied_vol' in df.columns:
        print(f"  Before IV filtering: {len(df)} options")
        df = filter_iv_outliers(df, iv_column='implied_vol', 
                                 lower_bound=0.01, upper_bound=1.5)
        print(f"  After IV filtering: {len(df)} options")
    
    # Filter for reasonable strikes (within ~50% of spot - you'll set actual spot)
    # This will be adjusted when we know S0
    if 'strike' in df.columns:
        # We'll filter by S0 later in the main function
        pass
    
    # Sort for cleaner output
    df = df.sort_values(['maturity', 'strike']).reset_index(drop=True)
    
    return df


def get_current_spot(ticker="SPY"):
    """
    Get current spot price for the ticker.
    Uses the options data to infer or fetch from API.
    """
    try:
        # Try to get from yfinance
        import yfinance as yf
        ticker_obj = yf.Ticker(ticker)
        spot = ticker_obj.history(period="1d")['Close'].iloc[-1]
        print(f"Current {ticker} spot price: ${spot:.2f}")
        return float(spot)
    except:
        # Fallback to typical values
        fallback_prices = {'SPY': 590.0, 'QQQ': 520.0, 'AAPL': 220.0, 'MSFT': 420.0}
        spot = fallback_prices.get(ticker, 500.0)
        print(f"Could not fetch spot price, using fallback: ${spot:.2f}")
        return spot

# ============================================================================
# IMPLIED VOLATILITY FUNCTIONS
# ============================================================================

def black_scholes_call(S0, K, r, T, sigma):
    """Black-Scholes call price for implied vol calculation."""
    if sigma <= 0 or T <= 0:
        return max(0, S0 - K * np.exp(-r * T))
    
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def implied_volatility(price, S0, K, r, T, initial_guess=0.2):
    """Calculate implied volatility from option price."""
    if price <= 0 or price >= S0:
        return np.nan
    
    intrinsic = max(0, S0 - K * np.exp(-r * T))
    if price < intrinsic:
        return np.nan
    
    def f(sigma):
        return black_scholes_call(S0, K, r, T, sigma) - price
    
    try:
        return brentq(f, 0.001, 5.0, xtol=1e-6)
    except (ValueError, RuntimeError):
        return np.nan
    

# ============================================================================
# CALIBRATION (Implemented Here)
# ============================================================================

def calibrate_heston_to_smile(market_df, S0, r, verbose=True, max_iterations=50):
    """
    Calibrate Heston parameters specifically to match the volatility smile.
    This focuses on fitting the curvature (smile) rather than just level.
    
    IMPLEMENTED IN vol_surface.py
    """
    print("\n" + "="*60)
    print("STEP: Calibrating Heston to Volatility Smile")
    print("="*60)
    
    # Get ATM strike (closest to spot)
    atm_strike = market_df.iloc[(market_df['strike'] - S0).abs().argsort()[:1]]['strike'].values[0]
    print(f"  ATM strike: {atm_strike:.2f}")
    
    # Get market IVs grouped by maturity
    maturities = sorted(market_df['maturity'].unique())
    
    # Prepare market data as a dictionary: {maturity: {strike: iv}}
    market_smiles = {}
    for T in maturities:
        df_T = market_df[market_df['maturity'] == T]
        if len(df_T) > 5:  # Need enough points for smile
            market_smiles[T] = {
                'strikes': df_T['strike'].values,
                'ivs': df_T['implied_vol'].values
            }
    
    print(f"  Found {len(market_smiles)} maturities with sufficient data")

    print("\n  Starting optimization...")
    print("  This may take 2-5 minutes depending on your system.")
    print("  Progress will update every 5 iterations.\n")

    # Use a mutable object to track iterations across function calls
    state = {'iteration_count': 0}
    
    def objective(params):
        v0, theta, kappa, sigma, rho = params
        
        # Clamp parameters to reasonable ranges
        v0 = max(0.01, min(0.5, v0))
        theta = max(0.01, min(0.5, theta))
        kappa = max(0.1, min(10, kappa))
        sigma = max(0.05, min(2.0, sigma))
        rho = max(-0.95, min(0.95, rho))
        
        total_error = 0
        n_points = 0

        # Update iteration counter
        state['iteration_count'] += 1
        current_iter = state['iteration_count']

        # Show progress every 5 iterations
        if current_iter % 5 == 0:
            print(f"    Iteration {current_iter}: Evaluating error...", end='\r')
        
        for T, smile_data in market_smiles.items():
            strikes_T = smile_data['strikes']
            market_ivs = smile_data['ivs']
            params_heston = HestonParams(S0=S0, K=np.array(strikes_T), r=r, T=float(T),
                                        v0=v0, theta=theta, kappa=kappa, sigma=sigma, rho=rho)
            prices, _, _, _ = heston_call_prices(params_heston)

            for K, price, market_iv in zip(strikes_T, prices, market_ivs):
                model_iv = implied_volatility(price, S0, K, r, float(T))
                price, _, _, _ = heston_call_prices(params_heston)
                
                # Convert to implied volatility
                model_iv = implied_volatility(price, S0, K, r, float(T))
                
                if not np.isnan(model_iv) and not np.isnan(market_iv):
                    # Weight by moneyness (more weight to ATM)
                    moneyness = abs(K / S0 - 1)
                    weight = 1.0 / (0.1 + moneyness)  # Higher weight near ATM
                    
                    error = (model_iv - market_iv) ** 2 * weight
                    total_error += error
                    n_points += 1

        mse = total_error / max(1, n_points)

        # Show current error on iteration milestone
        if current_iter % 5 == 0:
            print(f"    Iteration {current_iter}: MSE = {mse:.6f}     ")

        return mse
    
    # Start with parameters that produce a smile (higher vol of vol, negative rho)
    initial_params = [0.04, 0.04, 2.0, 0.5, -0.7]  # sigma=0.5 to create curvature
    
    bounds = [
        (0.01, 0.5),   # v0
        (0.01, 0.5),   # theta
        (0.1, 10.0),   # kappa
        (0.05, 2.0),   # sigma (allow higher vol of vol for smile)
        (-0.95, 0.95)  # rho
    ]
    
    # Callback to show progress
    def callback(xk):
        print(f"    New candidate parameters: v0={xk[0]:.4f}, sigma={xk[3]:.4f}, rho={xk[4]:.4f}")
    
    result = minimize(
        objective,
        initial_params,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iterations},
        callback=callback if verbose else None
    )
    
    print("\n  Optimization completed!")
    print(f"  Total iterations: {result.nit}")
    print(f"  Function evaluations: {result.nfev}")
    
    v0, theta, kappa, sigma, rho = result.x
    
    print("\n" + "="*60)
    print("CALIBRATION RESULTS")
    print("="*60)
    print(f"  v0 (initial variance): {v0:.4f}")
    print(f"  theta (long-term variance): {theta:.4f}")
    print(f"  kappa (mean reversion speed): {kappa:.2f}")
    print(f"  sigma (vol of vol): {sigma:.4f}  <-- KEY for smile curvature")
    print(f"  rho (correlation): {rho:.4f}   <-- KEY for skew")
    print(f"  Final MSE: {result.fun:.6f}")
    print(f"  Success: {result.success}")
    
    if not result.success:
        print(f"  Message: {result.message}")
    
    # Provide guidance
    print("\n" + "-"*40)
    print("CALIBRATION GUIDANCE:")
    if sigma < 0.2:
        print("Warning: sigma is low (<0.2). The volatility smile may be too flat.")
    elif sigma > 0.8:
        print("sigma is high (>0.8). Good for pronounced smile.")
    else:
        print("sigma is in reasonable range (0.2-0.8).")
    
    if abs(rho) < 0.3:
        print("Warning: |rho| is small (<0.3). The volatility skew may be weak.")
    elif abs(rho) > 0.7:
        print("|rho| is high (>0.7). Good for pronounced skew.")
    else:
        print(" rho is in reasonable range (0.3-0.7).")
    
    return {'v0': v0, 'theta': theta, 'kappa': kappa, 'sigma': sigma, 'rho': rho}


# ============================================================================
# VOLATILITY SURFACE GENERATION
# ============================================================================

def create_surface_from_market_data(market_df, S0, r, heston_params):
    """
    Create volatility surface from market data.
    
    Parameters:
    -----------
    market_df : pandas DataFrame
        Market data from Greeks_fetcher
    S0 : float
        Current spot price
    r : float
        Risk-free rate
    heston_params : dict
        Heston model parameters
    
    Returns:
    --------
    surfaces : dict
        Dictionary containing market_iv_surface, heston_iv_surface, diff_surface
    """
    print("\n" + "="*60)
    print("CREATING VOLATILITY SURFACE FROM MARKET DATA")
    print("="*60)
    
    # Filter for reasonable strikes (e.g., 70% to 130% of spot)
    min_strike = S0 * 0.7
    max_strike = S0 * 1.3
    market_df = market_df[(market_df['strike'] >= min_strike) & (market_df['strike'] <= max_strike)]
    
    # Get unique strikes and maturities from market data
    strikes = sorted(market_df['strike'].unique())
    maturities = sorted(market_df['maturity'].unique())
    
    print(f"\nMarket data grid: {len(maturities)} maturities × {len(strikes)} strikes")
    print(f"Strikes: {strikes[0]:.0f} to {strikes[-1]:.0f}")
    print(f"Maturities: {maturities[0]:.2f} to {maturities[-1]:.2f} years")
    
    # Pivot market implied vols to surface
    if 'implied_vol' in market_df.columns:
        market_iv_surface = market_df.pivot_table(
            index='maturity', 
            columns='strike', 
            values='implied_vol',
            aggfunc='first'
        )
    else:
        print("Warning: No implied_vol in market data, calculating from prices...")
        market_df['calc_iv'] = market_df.apply(
            lambda row: implied_volatility(row['market_price'], S0, row['strike'], r, row['maturity']),
            axis=1
        )
        market_iv_surface = market_df.pivot_table(
            index='maturity', 
            columns='strike', 
            values='calc_iv',
            aggfunc='first'
        )

    #Interpolate across strikes/maturities to ensure grid isn't sparse
    market_iv_surface = market_iv_surface.interpolate(axis=1, limit_direction='both') \
                                       .interpolate(axis=0, limit_direction='both')
    
    # Calculate Heston surface on same grid
    heston_iv_surface = calculate_heston_surface(
        heston_params, S0, r, strikes, maturities
    )
    
    # Align surfaces
    common_maturities = market_iv_surface.index.intersection(heston_iv_surface.index)
    common_strikes = market_iv_surface.columns.intersection(heston_iv_surface.columns)
    
    market_aligned = market_iv_surface.loc[common_maturities, common_strikes]
    heston_aligned = heston_iv_surface.loc[common_maturities, common_strikes]
    
    # Calculate difference
    diff_surface = heston_aligned - market_aligned
    
    # Calculate error metrics
    mae = diff_surface.abs().mean().mean()
    rmse = np.sqrt((diff_surface**2).mean().mean())
    
    print(f"\nComparison Metrics:")
    print(f"  Mean Absolute Error: {mae:.4f} ({mae*100:.2f}%)")
    print(f"  RMSE: {rmse:.4f} ({rmse*100:.2f}%)")
    
    return {
        'market_surface': market_aligned,
        'heston_surface': heston_aligned,
        'diff_surface': diff_surface,
        'mae': mae,
        'rmse': rmse
    }


def calculate_heston_surface(heston_params, S0, r, strikes, maturities, verbose=True):
    """
    Calculate Heston implied volatility surface.
    
    Parameters:
    -----------
    heston_params : dict
        Heston parameters (v0, theta, kappa, sigma, rho)
    S0 : float
        Current spot price
    r : float
        Risk-free rate
    strikes : array
        Strike prices
    maturities : array
        Time to maturities in years
    verbose : bool
        Print progress
    
    Returns:
    --------
    iv_surface : pd.DataFrame
        Implied volatilities
    """
    v0 = heston_params.get('v0', 0.04)
    theta = heston_params.get('theta', 0.04)
    kappa = heston_params.get('kappa', 2.0)
    sigma = heston_params.get('sigma', 0.3)
    rho = heston_params.get('rho', -0.7)
    
    # Initialize DataFrame
    iv_surface = pd.DataFrame(index=maturities, columns=strikes, dtype=float)
    price_surface = pd.DataFrame(index=maturities, columns=strikes, dtype=float)
    
    total = len(maturities) * len(strikes)
    count = 0
    
    print("\nCalculating Heston surface...")

    # Use tqdm for a nice progress bar
    with tqdm(total=total, desc="    Progress", unit="opt", 
              bar_format="{l_bar}{bar:40}{r_bar}") as pbar:
    
        for T in maturities:
            params = HestonParams(S0=S0, K=np.array(strikes), r=r, T=float(T),
                           v0=v0, theta=theta, kappa=kappa, sigma=sigma, rho=rho)
            prices, _, _, _ = heston_call_prices(params)  # returns array
            for K, price in zip(strikes, prices):
                iv_surface.loc[T, K] = implied_volatility(price, S0, K, r, float(T))
            
            # Sort for clean output
            iv_surface = iv_surface.sort_index().sort_index(axis=1)
            price_surface = price_surface.sort_index().sort_index(axis=1)
    
    # Check if Heston IV is still linear
    iv_values = iv_surface.values.flatten()
    iv_values = iv_values[~np.isnan(iv_values)]
    if len(iv_values) > 0:
        iv_range = iv_values.max() - iv_values.min()
        print(f"\n  Heston IV range: {iv_values.min():.4f} to {iv_values.max():.4f} (range: {iv_range:.4f})")
        if iv_range < 0.05:
            print("WARNING: Heston IV range is very small (linear/flat). Try increasing sigma (vol of vol).")
    
    return iv_surface, price_surface


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_surface_comparison(surfaces_dict, ticker="SPY"):
    """
    Plot market vs Heston volatility surfaces.
    """
    market_surface = surfaces_dict['market_surface']
    heston_surface = surfaces_dict['heston_surface']
    diff_surface = surfaces_dict['diff_surface']
    
    fig = plt.figure(figsize=(18, 5))
    
    # Plot 1: Market Surface
    ax1 = fig.add_subplot(131, projection='3d')
    plot_surface_3d(market_surface, ax1, title="Market Implied Volatility")
    
    # Plot 2: Heston Surface
    ax2 = fig.add_subplot(132, projection='3d')
    plot_surface_3d(heston_surface, ax2, title="Heston Model Volatility")
    
    # Plot 3: Difference
    ax3 = fig.add_subplot(133, projection='3d')
    plot_surface_3d(diff_surface, ax3, title="Heston - Market Difference")
    
    plt.suptitle(f"{ticker} Volatility Surface Comparison", fontsize=14)
    plt.tight_layout()
    return fig


def plot_surface_3d(surface_df, ax=None, title="Volatility Surface"):
    """Helper to plot 3D surface."""
    if ax is None:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
    
    # Clean data
    surface_df = surface_df.apply(pd.to_numeric, errors='coerce')
    surface_df = surface_df.dropna(how='all', axis=0).dropna(how='all', axis=1)
    
    if surface_df.empty:
        ax.set_title(f"{title} (No Data)")
        return ax
    
    T_vals = surface_df.index.values.astype(float)
    K_vals = surface_df.columns.values.astype(float)
    Z = surface_df.values.astype(float)
    Z = np.nan_to_num(Z, nan=0.0)
    
    T_grid, K_grid = np.meshgrid(T_vals, K_vals, indexing='ij')
    
    surf = ax.plot_surface(T_grid, K_grid, Z, cmap='viridis', 
                           edgecolor='none', alpha=0.9)
    
    ax.set_xlabel('Maturity (years)')
    ax.set_ylabel('Strike')
    ax.set_zlabel('Implied Vol')
    ax.set_title(title)
    
    plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    return ax


def plot_volatility_smile_comparison(surfaces_dict, maturities=None):
    """
    Plot volatility smiles for specific maturities.
    """
    market_surface = surfaces_dict['market_surface']
    heston_surface = surfaces_dict['heston_surface']
    
    if maturities is None:
        # Select up to 4 representative maturities
        all_maturities = market_surface.index.values
        n = min(4, len(all_maturities))
        maturities = [all_maturities[i] for i in range(0, len(all_maturities), max(1, len(all_maturities)//n))][:n]
    
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(maturities)))
    
    for idx, T in enumerate(maturities):
        # Find closest maturity in surfaces
        market_T = market_surface.index[market_surface.index.get_indexer([T], method='nearest')[0]]
        heston_T = heston_surface.index[heston_surface.index.get_indexer([T], method='nearest')[0]]
        
        market_vols = market_surface.loc[market_T]
        heston_vols = heston_surface.loc[heston_T]
        
        strikes = market_vols.index.values
        market_vals = market_vols.values
        heston_vals = heston_vols.values
        
        # Plot market
        mask = ~np.isnan(market_vals)
        if mask.any():
            plt.plot(strikes[mask], market_vals[mask], 'o-', 
                    color=colors[idx], linewidth=2, markersize=6,
                    label=f'Market (T={market_T:.2f}Y)')
        
        # Plot Heston
        mask_h = ~np.isnan(heston_vals)
        if mask_h.any():
            plt.plot(strikes[mask_h], heston_vals[mask_h], 's--', 
                    color=colors[idx], linewidth=2, markersize=6,
                    label=f'Heston (T={heston_T:.2f}Y)')
    
    plt.xlabel('Strike Price', fontsize=12)
    plt.ylabel('Implied Volatility', fontsize=12)
    plt.title('Volatility Smile Comparison: Market vs Heston', fontsize=14)
    plt.legend(loc='best', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_volatility_surface_analysis(ticker="SPY", max_days=20, r=0.05):
    """
    Main function: Fetch market data, create surfaces, visualize.
    """
    print("="*60)
    print(f"VOLATILITY SURFACE ANALYSIS FOR {ticker}")
    print("="*60)
    
    # Step 1: Fetch market data
    market_df = fetch_market_data(ticker=ticker, opt_type="c", max_days=max_days)
    
    # Step 2: Get current spot price
    S0 = get_current_spot(ticker)
    
    print(f"\n{'='*60}")
    print("MARKET DATA SUMMARY")
    print(f"{'='*60}")
    print(f"  Total options: {len(market_df)}")
    print(f"  Maturity range: {market_df['maturity'].min():.2f} to {market_df['maturity'].max():.2f} years")
    print(f"  Strike range: {market_df['strike'].min():.0f} to {market_df['strike'].max():.0f}")
    print(f"  Spot price (S0): ${S0:.2f}")
    
    # Step 3: Calibrate Heston parameters (CALLS THE NEW FUNCTION)
    heston_params = calibrate_heston_to_smile(market_df, S0, r, verbose=True)
    
    # Step 4: Create market surface (with filtered data)
    print(f"\n{'='*60}")
    print("CREATING MARKET VOLATILITY SURFACE")
    print(f"{'='*60}")
    
    min_strike = S0 * 0.7
    max_strike = S0 * 1.3
    market_df_filtered = market_df[(market_df['strike'] >= min_strike) & 
                                    (market_df['strike'] <= max_strike)]
    
    # First, clean the data more aggressively
    market_df_clean = market_df_filtered.copy()
    market_df_clean = market_df_clean[market_df_clean['implied_vol'] > 0.05]  # Remove IV < 5%
    market_df_clean = market_df_clean[market_df_clean['implied_vol'] < 0.60]  # Remove IV > 60%

    # Group by maturity and strike, take mean (or median) to handle duplicates
    market_iv_surface = market_df_clean.groupby(['maturity', 'strike'])['implied_vol'].mean().unstack()

    # Sort columns and index
    market_iv_surface = market_iv_surface.sort_index().sort_index(axis=1)

    print(f"\nMarket IV Surface shape: {market_iv_surface.shape}")
    print(f"Market IV range: {market_iv_surface.min().min():.4f} to {market_iv_surface.max().max():.4f}")
    print(f"Sample market IV values:\n{market_iv_surface.iloc[:3, :5]}")
    
    # Step 5: Create Heston surface on same grid
    strikes = sorted(market_iv_surface.columns)
    maturities = sorted(market_iv_surface.index)
    
    print(f"\n  Surface grid: {len(maturities)} maturities × {len(strikes)} strikes")
    
    heston_iv_surface, heston_price_surface = calculate_heston_surface(
        heston_params, S0, r, strikes, maturities, verbose=True
    )
    
    # Step 6: Align and compare
    common_maturities = market_iv_surface.index.intersection(heston_iv_surface.index)
    common_strikes = market_iv_surface.columns.intersection(heston_iv_surface.columns)
    
    market_aligned = market_iv_surface.loc[common_maturities, common_strikes]
    heston_aligned = heston_iv_surface.loc[common_maturities, common_strikes]
    diff_surface = heston_aligned - market_aligned
    
    mae = diff_surface.abs().mean().mean()
    rmse = np.sqrt((diff_surface**2).mean().mean())
    
    print(f"\n{'='*60}")
    print("FINAL METRICS")
    print(f"{'='*60}")
    print(f"  MAE: {mae:.4f} ({mae*100:.2f}%)")
    print(f"  RMSE: {rmse:.4f} ({rmse*100:.2f}%)")
    
    surfaces = {
        'market_surface': market_aligned,
        'heston_surface': heston_aligned,
        'diff_surface': diff_surface,
        'mae': mae,
        'rmse': rmse
    }
    
    # Step 7: Visualize
    print(f"\n{'='*60}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*60}")
    
    plot_surface_comparison(surfaces, ticker)
    plt.savefig(f'{ticker}_surface_comparison.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {ticker}_surface_comparison.png")
    
    plot_volatility_smile_comparison(surfaces)
    plt.savefig(f'{ticker}_smile_comparison.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {ticker}_smile_comparison.png")
    
    market_aligned.to_csv(f'{ticker}_market_surface.csv')
    heston_aligned.to_csv(f'{ticker}_heston_surface.csv')
    print(f"\n  Saved: {ticker}_market_surface.csv, {ticker}_heston_surface.csv")
    
    plt.show()
    
    return surfaces, heston_params

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Run analysis for SPY
    surfaces, best_params = run_volatility_surface_analysis(
        ticker="SPY",
        max_days=20,      # Options expiring within 10 days (TESTING)
        r=0.05,           # 5% risk-free rate (adjust as needed)
    )