This project encapsulates the Nelson-Siegel model to fit the term structure of interest rates. 
The ability to estimate the spot rate curve or bond yield curve plays a vital part to investors and policy makers.
Hence, as an initial start, we begin to investigate the Nelson-Siegel model, which transforms the nonlinear estimation problem into a simple linear problem, fixing the shape parameter that causes the nonlinearity.

We will start off by looking at the Nelson-Siegel function:

y(tau) = beta_0 + beta_1(1-e^-lambda * tau / lambda * tau) + beta_2((1-e^-lambda * tau / lambda * tau) - e^-lamda * tau)

where:

beta_0: long-run level: Asymptotic yield as maturity grows to infinity.
beta_1: slope: the spread between short & long spreads; negative beta = upward-sloping curve.
beta_2: curvature: the magnitude of the hump/trough of the curve.
lambda: decay parameter: the rate of decay of slope & curvature w.r.t. maturity.

1. Importing data
- We will use the U.S treasury data, via daily yield returns. This is open-source (available through their website).
- Current dataset: Using 2025 rates for tests.
- Rates available: 4, 6, 8, 13, 17, 26, 52 weeks (for 2025)

2. Fitting data
We write a method such that we properly fit the columns based on the timeframe that we are looking at (in years). Once done, then create an array that is formatted within the (tau, y_obs) order for each date, dropping all empty values.

3. Models
The original model is used to find the yield curves. As the original equation yields a single, smooth curve, it does not fit for more irregular data. Hence, the introduction of other models is being added.

Nelson-Siegel-Svensson:
We add a second decay factor, as well as beta_3, an extra dependent variable, in order to account for an extra hump in the yield rates, which was not previously accounted for in the original model. In this case, the cases occurred are the variables:
Beta_3: Creates the curvature of the model, influenced by lambda_1
Beta_3: Creates the ‘hump’ / second curvature, influenced by lambda_2
