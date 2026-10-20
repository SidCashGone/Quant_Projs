import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fit import load_par_yield_curve, get_curve, fit_ns, fit_nss
from ns_model import nelson_siegel
from nss_model import calculate_nss

if __name__ == "__main__":
    df = load_par_yield_curve("Nelson-Siegel/data/daily-treasury-rates-2025.csv")
    print("Shape:", df.shape)
    print("Maturities (years):", list(df.columns))

    date = "2025-01-02"
    tau, y = get_curve(df, date)


    #Original NS Model
    b0, b1, b2, lam = fit_ns(tau, y)
    y_hat_ns = nelson_siegel(tau, b0, b1, b2, lam)
    rmse_ns = np.sqrt(np.mean((y_hat_ns - y) ** 2)) * 1e4

    #NSS Model
    b0, b1, b2, b3, l1, l2 = fit_nss(tau, y)
    y_hat_nss = calculate_nss(tau, b0, b1, b2, b3, l1, l2)
    rmse_nss = np.sqrt(np.mean((y_hat_nss - y) ** 2)) * 1e4


    print(f"\nFit for {date}:")
    print(f"  β0 = {b0:.5f}  ({b0*100:.3f}%)")
    print(f"  β1 = {b1:.5f}  ({b1*100:.3f}%)")
    print(f"  β2 = {b2:.5f}  ({b2*100:.3f}%)")
    print(f"  β3 = {b2:.5f}  ({b2*100:.3f}%)")
    print(f"  λ1  = {lam:.4f}")
    print(f"  λ2  = {lam:.4f}")

    print(f"  RMSE_NS = {rmse_ns:.2f} bps")
    print(f"  RMSE_NSS = {rmse_nss:.2f} bps")

#Model plot
df = load_par_yield_curve("Nelson-Siegel/data/daily-treasury-rates-2025.csv")
 
# Pick dates spread across the year (nearest available trading day)
targets = ["2025-01-02", "2025-04-08", "2025-09-02", "2025-12-31"]
dates = [df.index[df.index.get_indexer([pd_t], method="nearest")[0]]
         for pd_t in __import__("pandas").to_datetime(targets)]
 
tau_grid = np.linspace(1/12, 30, 400)
 
fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
for ax, d in zip(axes.ravel(), dates):
    tau, y = get_curve(df, d)
    b0, b1, b2, lam = fit_ns(tau, y)
    y_hat = nelson_siegel(tau, b0, b1, b2, lam)
    rmse = np.sqrt(np.mean((y_hat - y) ** 2)) * 1e4
 
    ax.plot(tau_grid, nelson_siegel(tau_grid, b0, b1, b2, lam) * 100,
            color="tab:blue", label="Nelson-Siegel fit")
    ax.scatter(tau, y * 100, color="tab:red", s=25, zorder=3, label="Observed")
    ax.set_title(f"{d.date()}  (RMSE {rmse:.1f} bps, λ={lam:.2f})", fontsize=10)
    ax.grid(alpha=0.3)
 
for ax in axes[1]:
    ax.set_xlabel("Maturity (years)")
for ax in axes[:, 0]:
    ax.set_ylabel("Yield (%)")
axes[0, 0].legend()
fig.suptitle("Nelson-Siegel Fit vs Observed Treasury Par Yields")
fig.tight_layout()
fig.savefig("ns_fit_2025.png", dpi=150)


"""
COMPARISON GRAPHS
These graphs will be used to compare the method's yield via single-day
and RMSE returns.
"""

"""
Single-day Return
Choose an arbitrary point in the dataset to calculate NS & NSS
"""
date = "2025-02-03"
tau, y = get_curve(df, date)
#Original NS Model
b0, b1, b2, lam = fit_ns(tau, y)
y_hat_ns = nelson_siegel(tau, b0, b1, b2, lam)
rmse_ns = np.sqrt(np.mean((y_hat_ns - y) ** 2)) * 1e4

#NSS Model
b0, b1, b2, b3, l1, l2 = fit_nss(tau, y)
y_hat_nss = calculate_nss(tau, b0, b1, b2, b3, l1, l2)
rmse_nss = np.sqrt(np.mean((y_hat_nss - y) ** 2)) * 1e4

#Plotting
tau_grid = np.linspace(1/12, 30, 400)
plt.figure(figsize=(10,6))

#Plot grid lines for maturities
plt.scatter(tau, y*100, color='black', s=40, zorder=3, label='Observed Par Yields')

#Plot NS & NSS
plt.plot(tau_grid, nelson_siegel(tau_grid, b0, b1, b2, lam) * 100, 
         color="tab:blue", linestyle="--", linewidth=2, 
         label=f"Original NS (RMSE: {rmse_ns:.1f} bps)")


plt.plot(tau_grid, calculate_nss(tau_grid, b0, b1, b2, b3, l1, l2) * 100, 
         color="tab:orange", linewidth=2, 
         label=f"NSS (RMSE: {rmse_nss:.1f} bps)")

#Formatting
plt.title(f"Yield Curve Fit Comparison: {date}", fontsize=14, fontweight="bold")
plt.xlabel("Maturity (Years)", fontsize=12)
plt.ylabel("Yield (%)", fontsize=12)
plt.grid(alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()

# Save and show
plt.savefig("ns_vs_nss_single_day.png", dpi=150)
plt.show() # If running locally, this will open the window

