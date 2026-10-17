"""
This is the main calculating model, which will be used to calculate the Volatility
via the Heston Model.

It simplifies the main stochastic process using a characteristic process
"""

import numpy as np
import pandas as pandas
from scipy.integrate import quad

#Need to also find a method to import data to test the model.

#Define params for Heston Model
class HestonParams:
    def __init__(self, S0, K, r, T, v0, theta, kappa, sigma, rho):
        self.S0 = S0
        self.K = K
        self.r = r
        self.T = T
        self.v0 = v0
        self.theta = theta
        self.kappa = kappa
        self.sigma = sigma
        self.rho = rho

    def lns0(self):
        return np.log(self.S0)
    
    def log_strike(self):
        return np.log(self.K)

"""
Function for Heston Characteristic Function
"""
def heston_char_func(u, params):
    #To return phi(u), which will be used to create the expectation
    #phi(u) = E[e^{iuln(S_T)}] from 0 to infinite

    #Extract Params
    lnS0 = params.lns0()
    K = params.log_strike()
    r = params.r
    T = params.T
    v0 = params.v0
    theta = params.theta
    kappa = params.kappa
    sigma = params.sigma
    rho = params.rho

    # SPECIAL CASE: Deterministic variance (σ = 0 or very close to zero)
    if sigma < 1e-8:  # Use tolerance instead of exact equality
        # Integrated variance: ∫₀^T v(t) dt
        if kappa < 1e-8:  # If no mean reversion either
            integrated_var = v0 * T
        else:
            integrated_var = theta * T + (v0 - theta) * (1 - np.exp(-kappa * T)) / kappa
        
        # Characteristic function of normal distribution
        mu = lnS0 + (r - 0.5 * integrated_var) * T
        sigma_sq = integrated_var
        
        # Return normal characteristic function: exp(i*u*mu - 0.5*sigma_sq*u^2)
        return np.exp(1j * u * mu - 0.5 * sigma_sq * u**2)

    # Variable for imaginary number
    i = 1j

    # Define d and g
    a = kappa - rho * sigma * i * u
    b = sigma**2
    c = (i*u + u**2)

    # d(u) = sqrt( (rho*sigma*i*u - kappa)^2 + sigma*(i*u + u^2))
    d = np.sqrt( (rho * sigma * i * u - kappa)**2 + sigma**2 * (i*u + u**2))

    # g(u) = (kappa - rho*sigma*i*u - d) / (kappa - rho*sigma*i*u + d) )
    num = kappa - rho * sigma * i * u - d
    den = kappa - rho * sigma * i * u + d
    g = num / den

    #Function to calculate B(u, tau)
    exp_neg_dT = np.exp(-d * T)
    B = (kappa - rho * sigma * i * u - d) / b * (1 - exp_neg_dT) / (1 - g * exp_neg_dT)

    #Function to calculate A(u, tau)
    A = (kappa * theta) / b * ((kappa - rho * sigma * i * u - d) * T - 2 * np.log((1 - g * exp_neg_dT) / (1-g)))

    #Function to calculate phi(u) = exp(A + B*v0 + i*u(ln S0 + r*tau))
    exponent = A + B * v0 + i * u * (lnS0 + r * T)

    return np.exp(exponent)

"""
Gil-Palaez Integrands for quad
"""
def integrand_pi2(u, params):
    # Π₂: Re[ e^{-i u k} φ(u) / (i u) ]
    K = params.log_strike()
    phi_u = heston_char_func(u, params)

    i = 1j
    factor = np.exp(-i * u * K) * phi_u / (i * u)

    #We want the factor to be within a real plane, so handle any unknowns
    bad = np.isnan(factor) | np.isinf(factor)
    factor = np.where(bad, 0.0, factor)
    
    return np.real(factor)

def integrand_pi1(u, params):
    # Π₁: Re[ e^{-i u k} φ(u - i) / (i u φ(-i)) ]

    #Handle u near zero
    if abs(u) < 1e-10:
        return 0.0
    
    K = params.log_strike()
    
    #Find phi(u - i) first
    u_minus_i = u - 1j

    phi_u_minus_i = heston_char_func(u_minus_i, params)
    phi_neg_i = heston_char_func(-1j, params)

    factor = np.exp(-1j * u * K) * phi_u_minus_i / (1j * u * phi_neg_i)
    
    return np.real(factor)

"""
Method to handle any singularities for the quad
"""
def safe_integrand_pi2(u, params, eps=1e-10):
    if u < eps:
        return 0.0
    
    return integrand_pi2(u, params)

"""
MAIN OPTIONS PRICING MODEL
"""
def heston_call_prices(params):
    #Compute European calls

    #Compute Π₂ = 1/2 + (1/π) ∫₀^∞ Re[ ... ] du
    result_pi2, error_pi2 = quad(integrand_pi2, 0, np.inf, args=(params,), limit=200, epsabs=1e-8)
    pi2 = 0.5 + (1/np.pi) * result_pi2

    #Compute Π₁ = 1/2 + (1/π) ∫₀^∞ Re[ ... ] du
    result_pi1, error_pi1 = quad(integrand_pi1, 0, np.inf, args=(params,), limit=200, epsabs=1e-8)
    pi1 = 0.5 + (1/np.pi) * result_pi1

    #Finally, we can calculate the call price
    call_price = params.S0 * pi1 - params.K * np.exp(-params.r * params.T) * pi2

    return call_price, pi1, pi2, (error_pi1, error_pi2)

"""
Test case: run with only 1 price
"""

def test_heston():
    """
    Initial test params
    """
    params = HestonParams(
        S0=100, K=100, r=0.05, T=1.0,
        v0=0.04, theta=0.04, kappa=2.0,
        sigma=0.0, rho=0.0
    )

    #Define Heston Price:
    call_price, pi1, pi2, errors = heston_call_prices(params)

    # At u = 0, φ(0) should = 1
    phi_0 = heston_char_func(0.0, params)
    print(f"φ(0) = {phi_0} (should be 1+0j)")
    
    # At u = -i, φ(-i) should = E[S_T] = S0 * e^{rT}
    phi_neg_i = heston_char_func(-1j, params)
    expected = params.S0 * np.exp(params.r * params.T)
    print(f"φ(-i) = {phi_neg_i}")
    print(f"Expected E[S_T] = {expected}")
    print(f"Ratio = {phi_neg_i / expected} (should be 1.0)")

    print(f"Heston price: {call_price:.6f}")
    print(f"Π₁ (probability 1): {pi1:.6f}")
    print(f"Π₂ (probability 2): {pi2:.6f}")
    print(f"Integration errors: {errors}")


if __name__ == "__main__":
    test_heston()