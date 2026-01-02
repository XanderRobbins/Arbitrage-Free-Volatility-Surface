"""
Heston stochastic volatility model calibration and pricing.

Model:
    dS_t = r*S_t*dt + sqrt(v_t)*S_t*dW1_t
    dv_t = kappa*(theta - v_t)*dt + xi*sqrt(v_t)*dW2_t
    dW1_t * dW2_t = rho*dt
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import quad


class HestonModel:
    """Heston stochastic volatility model."""
    
    def __init__(self, kappa, theta, xi, rho, v0):
        """
        Parameters
        ----------
        kappa : float
            Mean reversion speed
        theta : float
            Long-term variance
        xi : float
            Volatility of volatility
        rho : float
            Correlation between asset and variance
        v0 : float
            Initial variance
        """
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho
        self.v0 = v0
    
    def characteristic_function(self, u, S, T, r):
        """
        Heston characteristic function for log(S_T).
        
        Parameters
        ----------
        u : complex
            Frequency parameter
        S : float
            Spot price
        T : float
            Time to maturity
        r : float
            Risk-free rate
        
        Returns
        -------
        complex
            Characteristic function value
        """
        kappa, theta, xi, rho, v0 = self.kappa, self.theta, self.xi, self.rho, self.v0
                
        # Heston characteristic function (Albrecher et al. formulation)
        i = complex(0, 1)
        
        d = np.sqrt((rho * xi * u * i - kappa)**2 + xi**2 * (u * i + u**2))
        g = (kappa - rho * xi * u * i - d) / (kappa - rho * xi * u * i + d)
        
        C = r * u * i * T + (kappa * theta / xi**2) * (
            (kappa - rho * xi * u * i - d) * T - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g))
        )
        
        D = ((kappa - rho * xi * u * i - d) / xi**2) * (
            (1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T))
        )
        
        return np.exp(C + D * v0 + i * u * np.log(S))
    
    def price_call_cos(self, S, K, T, r, N=128):
        """
        Price European call using COS method (Fang & Oosterlee 2008).
        
        Parameters
        ----------
        S : float
            Spot price
        K : float
            Strike price
        T : float
            Time to maturity
        r : float
            Risk-free rate
        N : int
            Number of terms in cosine expansion
        
        Returns
        -------
        float
            Call option price
        """
        # Truncation range [a, b] for log-asset price
        L = 10  # Controls range
        a = -L * np.sqrt(self.theta * T)
        b = L * np.sqrt(self.theta * T)
        
        x = np.log(S)
        k = np.arange(N)
        
        # Cosine series coefficients
        U_k = 2 / (b - a) * self._call_payoff_coefficient(k, a, b, np.log(K))
        
        # Characteristic function evaluated at k*pi/(b-a)
        cf_vals = np.array([
            self.characteristic_function(k_val * np.pi / (b - a), S, T, r)
            for k_val in k
        ])
        
        # Recover option price
        chi_k = np.cos(k * np.pi * (x - a) / (b - a))
        chi_k[0] *= 0.5  # Adjust for k=0 term
        
        price = np.exp(-r * T) * np.real(np.sum(cf_vals * U_k * chi_k))
        
        return max(price, 0)  # Ensure non-negative
    
    def _call_payoff_coefficient(self, k, a, b, log_strike):
        """Helper: Fourier-cosine coefficients for call payoff."""
        c = log_strike
        k = np.asarray(k)
        
        U_k = np.zeros_like(k, dtype=float)
        
        # k = 0 case
        mask_0 = (k == 0)
        if np.any(mask_0):
            U_k[mask_0] = np.exp(c) * (b - c) + (c - a)
        
        # k > 0 case
        mask_pos = (k > 0)
        k_pos = k[mask_pos]
        
        if len(k_pos) > 0:
            kpi = k_pos * np.pi / (b - a)
            
            term1 = (np.cos(kpi * (c - a)) - 1) / kpi**2
            term2 = ((c - a) * np.sin(kpi * (c - a))) / kpi
            
            U_k[mask_pos] = 2 * np.exp(c) * (term1 + term2) / (b - a)
        
        return U_k
    
    def price_put_cos(self, S, K, T, r, N=128):
        """Price European put using put-call parity."""
        call_price = self.price_call_cos(S, K, T, r, N)
        put_price = call_price - S + K * np.exp(-r * T)
        return max(put_price, 0)


def calibrate_heston(market_data, S, r, initial_guess=None, method='local'):
    """
    Calibrate Heston model to market implied volatilities.
    
    Parameters
    ----------
    market_data : pd.DataFrame
        Must have columns: ['strike', 'expiry', 'iv', 'option_type']
    S : float
        Spot price
    r : float
        Risk-free rate
    initial_guess : dict, optional
        Initial parameter values
    method : str
        'local' (L-BFGS-B) or 'global' (differential_evolution)
    
    Returns
    -------
    dict
        Calibrated parameters and diagnostics
    """
    from .iv_solver import black_scholes_call, black_scholes_put
    
    # Default initial guess
    if initial_guess is None:
        initial_guess = {
            'kappa': 2.0,
            'theta': 0.04,
            'xi': 0.3,
            'rho': -0.5,
            'v0': 0.04
        }
    
    x0 = [initial_guess['kappa'], initial_guess['theta'], 
          initial_guess['xi'], initial_guess['rho'], initial_guess['v0']]
    
    # Objective: minimize sum of squared relative price errors
    def objective(params):
        kappa, theta, xi, rho, v0 = params
        
        # Check Feller condition: 2*kappa*theta > xi^2
        if 2 * kappa * theta < xi**2:
            return 1e10
        
        model = HestonModel(kappa, theta, xi, rho, v0)
        
        total_error = 0
        count = 0
        
        for _, row in market_data.iterrows():
            K = row['strike']
            T = row['expiry']
            iv_market = row['iv']
            
            # Get market price
            if row.get('option_type', 'call').lower() == 'call':
                market_price = black_scholes_call(S, K, T, r, iv_market)
                model_price = model.price_call_cos(S, K, T, r, N=64)
            else:
                market_price = black_scholes_put(S, K, T, r, iv_market)
                model_price = model.price_put_cos(S, K, T, r, N=64)
            
            # Relative error (avoid division by zero)
            if market_price > 1e-6:
                error = ((model_price - market_price) / market_price)**2
                total_error += error
                count += 1
        
        return total_error / max(count, 1) if count > 0 else 1e10
    
    # Parameter bounds
    bounds = [
        (0.1, 10.0),      # kappa
        (0.001, 1.0),     # theta
        (0.01, 2.0),      # xi
        (-0.99, 0.99),    # rho
        (0.001, 1.0)      # v0
    ]
    
    if method == 'global':
        result = differential_evolution(objective, bounds, seed=42, maxiter=100, 
                                       workers=1, updating='deferred')
    else:
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 200})
    
    kappa, theta, xi, rho, v0 = result.x
    
    return {
        'kappa': kappa,
        'theta': theta,
        'xi': xi,
        'rho': rho,
        'v0': v0,
        'objective': result.fun,
        'success': result.success,
        'model': HestonModel(kappa, theta, xi, rho, v0)
    }