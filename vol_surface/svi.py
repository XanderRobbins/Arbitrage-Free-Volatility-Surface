"""
SVI (Stochastic Volatility Inspired) parameterization for volatility smiles.

References:
- Gatheral & Jacquier (2014): "Arbitrage-free SVI volatility surfaces"
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution


def svi_raw(k, a, b, rho, m, sigma):
    """
    Raw SVI parameterization for total implied variance.
    
    w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))
    
    Parameters
    ----------
    k : array-like
        Log-moneyness: k = log(K/F) where F = S*e^(rT)
    a : float
        Vertical shift
    b : float
        Slope (must be >= 0)
    rho : float
        Correlation (-1 <= rho <= 1)
    m : float
        Horizontal shift
    sigma : float
        Variance parameter (must be > 0)
    
    Returns
    -------
    array-like
        Total implied variance w = sigma_impl^2 * T
    """
    k = np.asarray(k)
    term = rho * (k - m) + np.sqrt((k - m)**2 + sigma**2)
    return a + b * term


def svi_raw_to_iv(k, T, a, b, rho, m, sigma):
    """Convert SVI total variance to implied volatility."""
    w = svi_raw(k, a, b, rho, m, sigma)
    return np.sqrt(np.maximum(w, 0) / T)


def fit_svi_slice(strikes, ivs, T, S, r, method='least_squares'):
    """
    Fit SVI parameters to a single expiry slice.
    
    Parameters
    ----------
    strikes : array-like
        Strike prices
    ivs : array-like
        Implied volatilities
    T : float
        Time to maturity
    S : float
        Spot price
    r : float
        Risk-free rate
    method : str
        'least_squares' or 'differential_evolution'
    
    Returns
    -------
    dict
        Fitted parameters and diagnostics
    """
    strikes = np.asarray(strikes)
    ivs = np.asarray(ivs)
    
    # Convert to log-moneyness and total variance
    F = S * np.exp(r * T)
    k = np.log(strikes / F)
    w_market = ivs**2 * T
    
    # Objective: minimize sum of squared errors
    def objective(params):
        a, b, rho, m, sigma = params
        w_model = svi_raw(k, a, b, rho, m, sigma)
        return np.sum((w_model - w_market)**2)
    
    # Constraints for no-arbitrage (simplified)
    def constraint_b(params):
        return params[1]  # b >= 0
    
    def constraint_sigma(params):
        return params[4]  # sigma > 0
    
    def constraint_rho(params):
        return 1 - abs(params[2])  # |rho| <= 1
    
    # Initial guess
    atm_iv = np.median(ivs)
    a0 = atm_iv**2 * T * 0.5
    b0 = 0.1
    rho0 = 0.0
    m0 = 0.0
    sigma0 = 0.1
    
    x0 = [a0, b0, rho0, m0, sigma0]
    
    if method == 'differential_evolution':
        # Global optimization
        bounds = [
            (0, atm_iv**2 * T * 2),  # a
            (0, 1),                   # b
            (-0.999, 0.999),          # rho
            (-0.5, 0.5),              # m
            (0.01, 1.0)               # sigma
        ]
        result = differential_evolution(objective, bounds, seed=42, maxiter=1000)
    else:
        # Local optimization
        bounds = [
            (0, None),        # a
            (0, None),        # b
            (-0.999, 0.999),  # rho
            (None, None),     # m
            (1e-6, None)      # sigma
        ]
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
    
    params = result.x
    a, b, rho, m, sigma = params
    
    # Compute fitted IVs
    w_fit = svi_raw(k, a, b, rho, m, sigma)
    iv_fit = np.sqrt(np.maximum(w_fit, 0) / T)
    
    # Diagnostics
    rmse = np.sqrt(np.mean((iv_fit - ivs)**2))
    
    return {
        'a': a,
        'b': b,
        'rho': rho,
        'm': m,
        'sigma': sigma,
        'rmse': rmse,
        'success': result.success,
        'iv_fit': iv_fit
    }


def fit_svi_surface(market_data, S, r, method='least_squares'):
    """
    Fit SVI to all expiries in market data.
    
    Parameters
    ----------
    market_data : pd.DataFrame
        Must have columns: ['strike', 'expiry', 'iv']
    S : float
        Spot price
    r : float
        Risk-free rate
    
    Returns
    -------
    dict
        {expiry: svi_params_dict}
    """
    surface_params = {}
    
    for T in sorted(market_data['expiry'].unique()):
        slice_data = market_data[market_data['expiry'] == T]
        strikes = slice_data['strike'].values
        ivs = slice_data['iv'].values
        
        result = fit_svi_slice(strikes, ivs, T, S, r, method)
        surface_params[T] = result
    
    return surface_params