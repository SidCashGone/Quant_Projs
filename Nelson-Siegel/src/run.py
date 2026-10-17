import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fit import load_par_yield_curve, get_curve, fit_ns
from ns_model import nelson_siegel

if __name__ == "__main__":
    df = load_par_yield_curve("Nelson-Siegel/data/daily-treasury-rates-2025.csv")
    print("Shape:", df.shape)
    print("Maturities (years):", list(df.columns))

    date = "2025-01-02"
    tau, y = get_curve(df, date)
    b0, b1, b2, lam = fit_ns(tau, y)

    print(f"\nFit for {date}:")
    print(f"  β0 = {b0:.5f}  ({b0*100:.3f}%)")
    print(f"  β1 = {b1:.5f}  ({b1*100:.3f}%)")
    print(f"  β2 = {b2:.5f}  ({b2*100:.3f}%)")
    print(f"  λ  = {lam:.4f}")

    y_hat = nelson_siegel(tau, b0, b1, b2, lam)
    rmse = np.sqrt(np.mean((y_hat - y) ** 2))
    print(f"  RMSE = {rmse*10000:.2f} bps")

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

