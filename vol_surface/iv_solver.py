"""
Implied volatility computation using robust root-finding methods.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq, newton


def black_scholes_call(S, K, T, r, sigma):
    """Compute Black-Scholes call option price."""
    if T <= 0 or sigma <= 0:
        return max(S - K * np.exp(-r * T), 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def black_scholes_put(S, K, T, r, sigma):
    """Compute Black-Scholes put option price."""
    if T <= 0 or sigma <= 0:
        return max(K * np.exp(-r * T) - S, 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def vega(S, K, T, r, sigma):
    """Compute Black-Scholes vega (derivative w.r.t. sigma)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)


def implied_vol_call(price, S, K, T, r, tol=1e-6, max_iter=100):
    """
    Compute implied volatility for a call option using Newton-Raphson.
    
    Parameters
    ----------
    price : float
        Market price of the call option
    S : float
        Spot price
    K : float
        Strike price
    T : float
        Time to maturity (years)
    r : float
        Risk-free rate
    tol : float
        Convergence tolerance
    max_iter : int
        Maximum iterations
    
    Returns
    -------
    float
        Implied volatility (annualized)
    """
    # Edge cases
    if T <= 0:
        return 0.0
    
    intrinsic = max(S - K * np.exp(-r * T), 0)
    if price <= intrinsic + 1e-10:
        return 0.0  # At or below intrinsic value
    
    # Initial guess: Brenner-Subrahmanyam approximation
    sigma = np.sqrt(2 * np.pi / T) * price / S
    sigma = max(sigma, 0.01)  # Lower bound
    
    for i in range(max_iter):
        price_est = black_scholes_call(S, K, T, r, sigma)
        diff = price_est - price
        
        if abs(diff) < tol:
            return sigma
        
        v = vega(S, K, T, r, sigma)
        if v < 1e-10:  # Avoid division by zero
            break
        
        sigma -= diff / v  # Newton step
        sigma = max(sigma, 1e-4)  # Ensure positive
    
    # Fallback: Brent's method
    try:
        sigma = brentq(
            lambda sig: black_scholes_call(S, K, T, r, sig) - price,
            1e-4, 5.0, xtol=tol
        )
        return sigma
    except:
        return np.nan


def implied_vol_put(price, S, K, T, r, tol=1e-6, max_iter=100):
    """
    Compute implied volatility for a put option using Newton-Raphson.
    
    Parameters
    ----------
    price : float
        Market price of the put option
    S : float
        Spot price
    K : float
        Strike price
    T : float
        Time to maturity (years)
    r : float
        Risk-free rate
    tol : float
        Convergence tolerance
    max_iter : int
        Maximum iterations
    
    Returns
    -------
    float
        Implied volatility (annualized)
    """
    # Edge cases
    if T <= 0:
        return 0.0
    
    intrinsic = max(K * np.exp(-r * T) - S, 0)
    if price <= intrinsic + 1e-10:
        return 0.0
    
    # Initial guess
    sigma = np.sqrt(2 * np.pi / T) * price / S
    sigma = max(sigma, 0.01)
    
    for i in range(max_iter):
        price_est = black_scholes_put(S, K, T, r, sigma)
        diff = price_est - price
        
        if abs(diff) < tol:
            return sigma
        
        v = vega(S, K, T, r, sigma)
        if v < 1e-10:
            break
        
        sigma -= diff / v
        sigma = max(sigma, 1e-4)
    
    # Fallback: Brent's method
    try:
        sigma = brentq(
            lambda sig: black_scholes_put(S, K, T, r, sig) - price,
            1e-4, 5.0, xtol=tol
        )
        return sigma
    except:
        return np.nan


def implied_vol(price, S, K, T, r, option_type='call', **kwargs):
    """
    Convenience wrapper for implied volatility computation.
    
    Parameters
    ----------
    option_type : str
        'call' or 'put'
    """
    if option_type.lower() == 'call':
        return implied_vol_call(price, S, K, T, r, **kwargs)
    elif option_type.lower() == 'put':
        return implied_vol_put(price, S, K, T, r, **kwargs)
    else:
        raise ValueError("option_type must be 'call' or 'put'")