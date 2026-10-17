#Import dependencies
import numpy as np

"""Return 3-factor loading for given maturities & lambda"""
def calibrate(tau, lambd):
    tau = np.asarray(tau, dtype=float)
    x = lambd * tau

    #Avoid div by 0
    exp_term = np.exp(-x)
    f0 = np.ones_like(tau)
    f1 = np.where(x<1e-8, 1.0, (1-exp_term) / x)
    f2 = f1 - exp_term

    return f0, f1, f2


"""
Main Nelson-Siegel function to calculate y(tau):
"""
def nelson_siegel(tau, beta_0, beta_1, beta_2, lambd):
    f0, f1, f2 = calibrate(tau, lambd)
    return beta_0*f0 + beta_1*f1 + beta_2*f2

