import numpy as np

"""
NSS Model has 6 params:
beta_1 to 4, lambdas 1 and 2.
"""

def calibrate_nss(tau, lambd1, lambd2):
    tau = np.asarray(tau, dtype=float)
    x1 = tau*lambd1
    x2 = tau*lambd2

    #Create exponents for the coefficients to betas
    exp1 = np.exp(-x1)
    exp2 = np.exp(-x2)

    #Create the main exponents now
    f0 = np.ones_like(tau)
    f1 = np.where(x1 < 1e-8, 1.0, (1-exp1)/x1)
    f2 = f1 - exp1

    f3 = np.where(x2 < 1e-8, 1.0, (1-exp2)/x2) - exp2

    return f0, f1, f2, f3

def calculate_nss(tau, beta_0, beta_1, beta_2, beta_3, lambd1, lambd2):
    #Main equation to calculate NSS
    f0, f1, f2, f3 = calibrate_nss(tau, lambd1, lambd2)
    return f0*beta_0 + f1*beta_1 + f2*beta_2 + f3*beta_3



