"""
Static arbitrage checks for option prices and implied volatilities.
"""

import numpy as np
from .iv_solver import black_scholes_call, black_scholes_put


def check_put_call_parity(call_price, put_price, S, K, T, r, tol=1e-3):
    """
    Check put-call parity: C - P = S - K*e^(-rT)
    
    Returns
    -------
    dict
        {'is_violated': bool, 'lhs': float, 'rhs': float, 'diff': float}
    """
    lhs = call_price - put_price
    rhs = S - K * np.exp(-r * T)
    diff = abs(lhs - rhs)
    
    return {
        'is_violated': diff > tol,
        'lhs': lhs,
        'rhs': rhs,
        'diff': diff
    }


def check_butterfly_arbitrage(strikes, ivs, S, T, r, tol=1e-6):
    """
    Check butterfly arbitrage: ensure convexity of option prices w.r.t. strike.
    
    For three consecutive strikes K1 < K2 < K3:
    C(K2) <= [(K3-K2)*C(K1) + (K2-K1)*C(K3)] / (K3-K1)
    
    Parameters
    ----------
    strikes : array-like
        Sorted strike prices
    ivs : array-like
        Implied volatilities corresponding to strikes
    S : float
        Spot price
    T : float
        Time to maturity
    r : float
        Risk-free rate
    
    Returns
    -------
    list of dict
        Violations with details
    """
    strikes = np.asarray(strikes)
    ivs = np.asarray(ivs)
    
    if len(strikes) < 3:
        return []
    
    # Sort by strike
    idx = np.argsort(strikes)
    strikes = strikes[idx]
    ivs = ivs[idx]
    
    violations = []
    
    for i in range(1, len(strikes) - 1):
        K1, K2, K3 = strikes[i-1], strikes[i], strikes[i+1]
        iv1, iv2, iv3 = ivs[i-1], ivs[i], ivs[i+1]
        
        # Compute call prices
        C1 = black_scholes_call(S, K1, T, r, iv1)
        C2 = black_scholes_call(S, K2, T, r, iv2)
        C3 = black_scholes_call(S, K3, T, r, iv3)
        
        # Butterfly spread price
        butterfly_price = C1 - 2*C2 + C3
        
        if butterfly_price < -tol:
            violations.append({
                'type': 'butterfly',
                'strikes': (K1, K2, K3),
                'ivs': (iv1, iv2, iv3),
                'butterfly_price': butterfly_price
            })
    
    return violations


def check_calendar_arbitrage(iv_surface_dict, strikes, r, tol=1e-6):
    """
    Check calendar arbitrage: longer-dated options should cost more.
    
    Parameters
    ----------
    iv_surface_dict : dict
        {expiry: {strike: iv}} nested dict
    strikes : array-like
        Strikes to check
    r : float
        Risk-free rate
    
    Returns
    -------
    list of dict
        Violations
    """
    expiries = sorted(iv_surface_dict.keys())
    if len(expiries) < 2:
        return []
    
    violations = []
    
    for K in strikes:
        for i in range(len(expiries) - 1):
            T1, T2 = expiries[i], expiries[i+1]
            
            if K not in iv_surface_dict[T1] or K not in iv_surface_dict[T2]:
                continue
            
            iv1 = iv_surface_dict[T1][K]
            iv2 = iv_surface_dict[T2][K]
            
            # Compare total variance (sigma^2 * T should be increasing)
            var1 = iv1**2 * T1
            var2 = iv2**2 * T2
            
            if var2 < var1 - tol:
                violations.append({
                    'type': 'calendar',
                    'strike': K,
                    'expiries': (T1, T2),
                    'ivs': (iv1, iv2),
                    'variances': (var1, var2)
                })
    
    return violations


def check_all_arbitrage(market_data, S, r, tol=1e-3):
    """
    Run all arbitrage checks on a DataFrame of market data.
    
    Parameters
    ----------
    market_data : pd.DataFrame
        Must have columns: ['strike', 'expiry', 'call_price', 'put_price', 'iv']
    S : float
        Spot price
    r : float
        Risk-free rate
    
    Returns
    -------
    dict
        Summary of all violations
    """
    violations = {
        'put_call_parity': [],
        'butterfly': [],
        'calendar': []
    }
    
    # Check put-call parity
    for _, row in market_data.iterrows():
        if 'call_price' in row and 'put_price' in row:
            result = check_put_call_parity(
                row['call_price'], row['put_price'],
                S, row['strike'], row['expiry'], r, tol
            )
            if result['is_violated']:
                violations['put_call_parity'].append({
                    'strike': row['strike'],
                    'expiry': row['expiry'],
                    **result
                })
    
    # Check butterfly per expiry
    for T in market_data['expiry'].unique():
        slice_data = market_data[market_data['expiry'] == T].sort_values('strike')
        strikes = slice_data['strike'].values
        ivs = slice_data['iv'].values
        
        butterfly_viols = check_butterfly_arbitrage(strikes, ivs, S, T, r, tol)
        violations['butterfly'].extend(butterfly_viols)
    
    # Check calendar
    iv_surface_dict = {}
    for T in market_data['expiry'].unique():
        iv_surface_dict[T] = {}
        slice_data = market_data[market_data['expiry'] == T]
        for _, row in slice_data.iterrows():
            iv_surface_dict[T][row['strike']] = row['iv']
    
    strikes = market_data['strike'].unique()
    calendar_viols = check_calendar_arbitrage(iv_surface_dict, strikes, r, tol)
    violations['calendar'].extend(calendar_viols)
    
    return violations