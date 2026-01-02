"""
Option pricing utilities: Black-Scholes, COS method, FFT.
"""

import numpy as np
from scipy.stats import norm
from scipy.integrate import quad


def black_scholes_call(S, K, T, r, sigma):
    """Black-Scholes call option price."""
    if T <= 0 or sigma <= 0:
        return max(S - K * np.exp(-r * T), 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def black_scholes_put(S, K, T, r, sigma):
    """Black-Scholes put option price."""
    if T <= 0 or sigma <= 0:
        return max(K * np.exp(-r * T) - S, 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def cos_pricer(cf, S, K, T, r, option_type='call', N=128, L=10):
    """
    COS method for option pricing given a characteristic function.
    
    Parameters
    ----------
    cf : callable
        Characteristic function: cf(u) -> complex
        Should be the CF of log(S_T) under the risk-neutral measure
    S : float
        Spot price
    K : float
        Strike price
    T : float
        Time to maturity
    r : float
        Risk-free rate
    option_type : str
        'call' or 'put'
    N : int
        Number of cosine terms
    L : float
        Truncation range multiplier
    
    Returns
    -------
    float
        Option price
    """
    x = np.log(S)
    
    # Estimate truncation range [a, b]
    # Simple heuristic based on standard deviation
    c1 = r * T
    c2 = 0.0  # Would need to compute from CF for general case
    c4 = 3.0  # Kurtosis approximation
    
    a = c1 - L * np.sqrt(abs(c2) + np.sqrt(abs(c4)))
    b = c1 + L * np.sqrt(abs(c2) + np.sqrt(abs(c4)))
    
    k = np.arange(N)
    
    # Payoff coefficients
    if option_type.lower() == 'call':
        U_k = _call_payoff_coef(k, a, b, np.log(K))
    else:
        U_k = _put_payoff_coef(k, a, b, np.log(K))
    
    # Evaluate CF at k*pi/(b-a)
    cf_vals = np.array([cf(k_val * np.pi / (b - a)) for k_val in k])
    
    # Cosine series
    chi_k = np.cos(k * np.pi * (x - a) / (b - a))
    chi_k[0] *= 0.5
    
    price = np.exp(-r * T) * np.real(np.sum(cf_vals * U_k * chi_k))
    
    return max(price, 0)


def _call_payoff_coef(k, a, b, log_strike):
    """Fourier-cosine coefficients for call payoff."""
    c = log_strike
    k = np.asarray(k)
    U_k = np.zeros_like(k, dtype=float)
    
    mask_0 = (k == 0)
    if np.any(mask_0):
        U_k[mask_0] = np.exp(c) * (b - c) + (c - a)
    
    mask_pos = (k > 0)
    k_pos = k[mask_pos]
    
    if len(k_pos) > 0:
        kpi = k_pos * np.pi / (b - a)
        term1 = (np.cos(kpi * (c - a)) - 1) / kpi**2
        term2 = ((c - a) * np.sin(kpi * (c - a))) / kpi
        U_k[mask_pos] = 2 * np.exp(c) * (term1 + term2) / (b - a)
    
    return U_k


def _put_payoff_coef(k, a, b, log_strike):
    """Fourier-cosine coefficients for put payoff."""
    c = log_strike
    k = np.asarray(k)
    U_k = np.zeros_like(k, dtype=float)
    
    mask_0 = (k == 0)
    if np.any(mask_0):
        U_k[mask_0] = (b - c) - np.exp(c) * (b - c)
    
    mask_pos = (k > 0)
    k_pos = k[mask_pos]
    
    if len(k_pos) > 0:
        kpi = k_pos * np.pi / (b - a)
        term1 = (1 - np.cos(kpi * (c - a))) / kpi**2
        term2 = ((c - a) * np.sin(kpi * (c - a))) / kpi
        U_k[mask_pos] = 2 * (term1 - np.exp(c) * term2) / (b - a)
    
    return U_k


def fft_pricer(cf, S, K_array, T, r, option_type='call', N=4096, alpha=1.5):
    """
    FFT-based option pricing for multiple strikes (Carr-Madan method).
    
    Parameters
    ----------
    cf : callable
        Characteristic function
    S : float
        Spot price
    K_array : array-like
        Array of strike prices
    T : float
        Time to maturity
    r : float
        Risk-free rate
    option_type : str
        'call' or 'put'
    N : int
        Number of FFT points (power of 2)
    alpha : float
        Damping parameter (>0 for calls, <0 for puts)
    
    Returns
    -------
    array
        Option prices for each strike
    """
    # FFT grid
    eta = 0.25  # Spacing in log-strike
    lambda_val = 2 * np.pi / (N * eta)
    
    b = N * lambda_val / 2
    
    u = np.arange(N) * eta
    
    # Modified CF with damping
    psi_u = np.exp(-r * T) * cf(u - (alpha + 1) * 1j) / (alpha**2 + alpha - u**2 + 1j * (2 * alpha + 1) * u)
    
    # Simpson's rule weights
    w = np.ones(N)
    w[0] = 0.5
    w = w * eta
    
    # FFT
    x = np.exp(1j * b * u) * psi_u * w
    fft_vals = np.fft.fft(x)
    
    # Extract prices at desired strikes
    k_array = np.log(K_array / S)
    
    prices = []
    for k in k_array:
        j = int((k + b) / lambda_val)
        if 0 <= j < N:
            price = np.real(np.exp(-alpha * k) * fft_vals[j] / np.pi)
            prices.append(max(price, 0))
        else:
            prices.append(np.nan)
    
    return np.array(prices)