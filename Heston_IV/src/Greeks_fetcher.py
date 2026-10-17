"""
This is to be used to retrieve any Options chain data from a given time
Data extracted will be directly from Yahoo Finance.
"""

import greeks_package as gp
import pandas as pd

#Test for downloading options within 180 days (approx 6 months)
opts = gp.download_options("SPY", opt_type="c", max_days=180)

#Get the Greeks
all_greeks = opts.apply(gp.greeks, axis=1, ticker="SPY")

#Combine with data
full_data = opts.join(all_greeks)

#Print out to see outputs
print("Columns in fetched data:")
print(full_data.columns.tolist())
print("\nFirst 5 rows:")
print(full_data.head())
print("\nData types:")
print(full_data.dtypes)