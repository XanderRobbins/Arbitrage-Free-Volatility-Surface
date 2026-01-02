"""
Main VolatilitySurface class: orchestrates IV computation, SVI fitting, 
arbitrage checks, and Heston calibration.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from .iv_solver import implied_vol
from .arbitrage import check_all_arbitrage
from .svi import fit_svi_surface, svi_raw_to_iv
from .heston import calibrate_heston


class VolatilitySurface:
    """
    Complete volatility surface construction and analysis.
    
    Workflow:
    1. Load market data
    2. Compute implied volatilities
    3. Check for arbitrage violations
    4. Fit SVI parameterization
    5. Calibrate Heston model
    6. Compare and visualize
    """
    
    def __init__(self, S, r=0.0):
        """
        Parameters
        ----------
        S : float
            Current spot price
        r : float
            Risk-free rate (annualized)
        """
        self.S = S
        self.r = r
        self.market_data = None
        self.svi_params = None
        self.heston_params = None
        self.arbitrage_violations = None
    
    def load_data(self, data):
        """
        Load market option data.
        
        Parameters
        ----------
        data : pd.DataFrame
            Must contain: ['strike', 'expiry', 'option_type', 'price']
            Optional: ['bid', 'ask', 'volume']
        """
        required_cols = ['strike', 'expiry', 'option_type', 'price']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data must contain columns: {required_cols}")
        
        self.market_data = data.copy()
        return self
    
    def compute_ivs(self, overwrite=False):
        """
        Compute implied volatilities for all options.
        
        Parameters
        ----------
        overwrite : bool
            If True, recompute even if 'iv' column exists
        """
        if self.market_data is None:
            raise ValueError("Must load data first using .load_data()")
        
        if 'iv' in self.market_data.columns and not overwrite:
            print("IVs already computed. Use overwrite=True to recompute.")
            return self
        
        print("Computing implied volatilities...")
        
        ivs = []
        for _, row in self.market_data.iterrows():
            iv = implied_vol(
                price=row['price'],
                S=self.S,
                K=row['strike'],
                T=row['expiry'],
                r=self.r,
                option_type=row['option_type']
            )
            ivs.append(iv)
        
        self.market_data['iv'] = ivs
        
        # Filter out failed computations
        valid_mask = ~np.isnan(self.market_data['iv'])
        n_invalid = (~valid_mask).sum()
        
        if n_invalid > 0:
            print(f"Warning: {n_invalid} IVs failed to converge (removed)")
            self.market_data = self.market_data[valid_mask]
        
        print(f"✓ Computed {len(self.market_data)} implied volatilities")
        return self
    
    def check_arbitrage(self, tol=1e-3):
        """
        Run all arbitrage checks.
        
        Parameters
        ----------
        tol : float
            Tolerance for violations
        """
        if 'iv' not in self.market_data.columns:
            raise ValueError("Must compute IVs first using .compute_ivs()")
        
        print("Checking for arbitrage violations...")
        
        self.arbitrage_violations = check_all_arbitrage(
            self.market_data, self.S, self.r, tol
        )
        
        # Print summary
        for key, violations in self.arbitrage_violations.items():
            n = len(violations)
            status = "✗" if n > 0 else "✓"
            print(f"{status} {key}: {n} violations")
        
        return self
    
    def fit_svi(self, method='least_squares'):
        """
        Fit SVI parameterization to the surface.
        
        Parameters
        ----------
        method : str
            'least_squares' or 'differential_evolution'
        """
        if 'iv' not in self.market_data.columns:
            raise ValueError("Must compute IVs first")
        
        print("Fitting SVI parameterization...")
        
        self.svi_params = fit_svi_surface(
            self.market_data, self.S, self.r, method
        )
        
        # Print diagnostics
        for T, params in self.svi_params.items():
            print(f"  T={T:.3f}: RMSE={params['rmse']:.6f}, success={params['success']}")
        
        print(f"✓ Fitted SVI to {len(self.svi_params)} expiries")
        return self
    
        def calibrate_heston(self, method='local', initial_guess=None):

            if 'iv' not in self.market_data.columns:
                raise ValueError("Must compute IVs first")
        
        print("Calibrating Heston model...")
        
        self.heston_params = calibrate_heston(
            self.market_data, self.S, self.r, initial_guess, method
        )
        
        print(f"✓ Heston calibration complete (error={self.heston_params['objective']:.6f})")
        print(f"  κ={self.heston_params['kappa']:.4f}, "
              f"θ={self.heston_params['theta']:.4f}, "
              f"ξ={self.heston_params['xi']:.4f}")
        print(f"  ρ={self.heston_params['rho']:.4f}, "
              f"v₀={self.heston_params['v0']:.4f}")
        
        return self
    
    def plot_smile(self, expiry, include_svi=True, include_heston=True, figsize=(10, 6)):
        """
        Plot IV smile for a single expiry.
        
        Parameters
        ----------
        expiry : float
            Time to maturity
        include_svi : bool
            Overlay SVI fit
        include_heston : bool
            Overlay Heston model prices
        """
        if 'iv' not in self.market_data.columns:
            raise ValueError("Must compute IVs first")
        
        # Filter data for this expiry
        tol = 1e-6
        slice_data = self.market_data[np.abs(self.market_data['expiry'] - expiry) < tol]
        
        if len(slice_data) == 0:
            print(f"No data found for expiry T={expiry}")
            return
        
        slice_data = slice_data.sort_values('strike')
        strikes = slice_data['strike'].values
        ivs = slice_data['iv'].values
        
        plt.figure(figsize=figsize)
        
        # Market data
        plt.plot(strikes, ivs, 'o', label='Market', markersize=8, alpha=0.7)
        
        # SVI fit
        if include_svi and self.svi_params is not None and expiry in self.svi_params:
            params = self.svi_params[expiry]
            F = self.S * np.exp(self.r * expiry)
            k_fine = np.log(np.linspace(strikes.min(), strikes.max(), 100) / F)
            iv_svi = svi_raw_to_iv(k_fine, expiry, params['a'], params['b'], 
                                    params['rho'], params['m'], params['sigma'])
            K_fine = F * np.exp(k_fine)
            plt.plot(K_fine, iv_svi, '-', label=f'SVI (RMSE={params["rmse"]:.4f})', linewidth=2)
        
        # Heston fit
        if include_heston and self.heston_params is not None:
            from .iv_solver import implied_vol_call
            model = self.heston_params['model']
            
            iv_heston = []
            for K in strikes:
                price = model.price_call_cos(self.S, K, expiry, self.r, N=64)
                iv = implied_vol_call(price, self.S, K, expiry, self.r)
                iv_heston.append(iv)
            
            plt.plot(strikes, iv_heston, '--', label='Heston', linewidth=2, alpha=0.8)
        
        plt.xlabel('Strike Price', fontsize=12)
        plt.ylabel('Implied Volatility', fontsize=12)
        plt.title(f'Volatility Smile (T={expiry:.3f} years)', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_surface_3d(self, model='market', figsize=(12, 8), elev=20, azim=45):
        """
        Plot 3D volatility surface.
        
        Parameters
        ----------
        model : str
            'market', 'svi', or 'heston'
        """
        if 'iv' not in self.market_data.columns:
            raise ValueError("Must compute IVs first")
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        if model == 'market':
            # Scatter plot of market IVs
            strikes = self.market_data['strike'].values
            expiries = self.market_data['expiry'].values
            ivs = self.market_data['iv'].values
            
            ax.scatter(strikes, expiries, ivs, c=ivs, cmap='viridis', 
                      s=50, alpha=0.7, edgecolors='k', linewidth=0.5)
            title = 'Market Implied Volatility Surface'
        
        elif model == 'svi':
            if self.svi_params is None:
                raise ValueError("Must fit SVI first using .fit_svi()")
            
            # Create mesh
            expiries = sorted(self.svi_params.keys())
            strikes_per_expiry = []
            
            for T in expiries:
                slice_data = self.market_data[self.market_data['expiry'] == T]
                strikes_per_expiry.append(slice_data['strike'].values)
            
            K_min = min([s.min() for s in strikes_per_expiry])
            K_max = max([s.max() for s in strikes_per_expiry])
            
            K_grid = np.linspace(K_min, K_max, 50)
            T_grid = np.array(expiries)
            
            K_mesh, T_mesh = np.meshgrid(K_grid, T_grid)
            IV_mesh = np.zeros_like(K_mesh)
            
            for i, T in enumerate(expiries):
                params = self.svi_params[T]
                F = self.S * np.exp(self.r * T)
                k = np.log(K_grid / F)
                IV_mesh[i, :] = svi_raw_to_iv(k, T, params['a'], params['b'],
                                              params['rho'], params['m'], params['sigma'])
            
            ax.plot_surface(K_mesh, T_mesh, IV_mesh, cmap='viridis', 
                          alpha=0.8, edgecolor='none')
            title = 'SVI Fitted Volatility Surface'
        
        elif model == 'heston':
            if self.heston_params is None:
                raise ValueError("Must calibrate Heston first")
            
            # Create mesh
            from .iv_solver import implied_vol_call
            
            expiries = self.market_data['expiry'].unique()
            strikes_range = self.market_data['strike'].values
            K_min, K_max = strikes_range.min(), strikes_range.max()
            
            K_grid = np.linspace(K_min, K_max, 30)
            T_grid = np.linspace(expiries.min(), expiries.max(), 20)
            
            K_mesh, T_mesh = np.meshgrid(K_grid, T_grid)
            IV_mesh = np.zeros_like(K_mesh)
            
            model_obj = self.heston_params['model']
            
            for i in range(len(T_grid)):
                for j in range(len(K_grid)):
                    T = T_mesh[i, j]
                    K = K_mesh[i, j]
                    price = model_obj.price_call_cos(self.S, K, T, self.r, N=64)
                    IV_mesh[i, j] = implied_vol_call(price, self.S, K, T, self.r)
            
            ax.plot_surface(K_mesh, T_mesh, IV_mesh, cmap='plasma', 
                          alpha=0.8, edgecolor='none')
            title = 'Heston Model Volatility Surface'
        
        else:
            raise ValueError("model must be 'market', 'svi', or 'heston'")
        
        ax.set_xlabel('Strike Price', fontsize=11)
        ax.set_ylabel('Time to Maturity', fontsize=11)
        ax.set_zlabel('Implied Volatility', fontsize=11)
        ax.set_title(title, fontsize=14, pad=20)
        ax.view_init(elev=elev, azim=azim)
        
        plt.tight_layout()
        plt.show()
    
    def compare_models(self, expiry=None, metric='rmse'):
        """
        Compare SVI and Heston fits to market data.
        
        Parameters
        ----------
        expiry : float, optional
            If None, compare across all expiries
        metric : str
            'rmse' or 'mape' (mean absolute percentage error)
        
        Returns
        -------
        pd.DataFrame
            Comparison metrics
        """
        if self.svi_params is None or self.heston_params is None:
            raise ValueError("Must fit both SVI and Heston first")
        
        from .iv_solver import implied_vol_call
        
        results = []
        
        expiries_to_check = [expiry] if expiry is not None else self.market_data['expiry'].unique()
        
        for T in expiries_to_check:
            slice_data = self.market_data[self.market_data['expiry'] == T]
            
            if len(slice_data) == 0:
                continue
            
            strikes = slice_data['strike'].values
            iv_market = slice_data['iv'].values
            
            # SVI predictions
            if T in self.svi_params:
                params = self.svi_params[T]
                F = self.S * np.exp(self.r * T)
                k = np.log(strikes / F)
                iv_svi = svi_raw_to_iv(k, T, params['a'], params['b'],
                                       params['rho'], params['m'], params['sigma'])
            else:
                iv_svi = np.full_like(iv_market, np.nan)
            
            # Heston predictions
            model = self.heston_params['model']
            iv_heston = []
            for K in strikes:
                price = model.price_call_cos(self.S, K, T, self.r, N=64)
                iv = implied_vol_call(price, self.S, K, T, self.r)
                iv_heston.append(iv)
            iv_heston = np.array(iv_heston)
            
            # Compute metrics
            if metric == 'rmse':
                svi_error = np.sqrt(np.mean((iv_svi - iv_market)**2))
                heston_error = np.sqrt(np.mean((iv_heston - iv_market)**2))
            elif metric == 'mape':
                svi_error = np.mean(np.abs((iv_svi - iv_market) / iv_market)) * 100
                heston_error = np.mean(np.abs((iv_heston - iv_market) / iv_market)) * 100
            else:
                raise ValueError("metric must be 'rmse' or 'mape'")
            
            results.append({
                'expiry': T,
                'n_options': len(strikes),
                f'SVI_{metric}': svi_error,
                f'Heston_{metric}': heston_error
            })
        
        df = pd.DataFrame(results)
        print(f"\nModel Comparison ({metric.upper()}):")
        print(df.to_string(index=False))
        print(f"\nAverage {metric.upper()}:")
        print(f"  SVI:    {df[f'SVI_{metric}'].mean():.6f}")
        print(f"  Heston: {df[f'Heston_{metric}'].mean():.6f}")
        
        return df
    
    def summary(self):
        """Print summary statistics of the surface."""
        if self.market_data is None:
            print("No data loaded")
            return
        
        print("=" * 60)
        print("VOLATILITY SURFACE SUMMARY")
        print("=" * 60)
        print(f"Spot price:       ${self.S:.2f}")
        print(f"Risk-free rate:   {self.r*100:.2f}%")
        print(f"Total options:    {len(self.market_data)}")
        
        if 'iv' in self.market_data.columns:
            print(f"\nImplied Volatility Stats:")
            print(f"  Mean:   {self.market_data['iv'].mean():.4f}")
            print(f"  Median: {self.market_data['iv'].median():.4f}")
            print(f"  Min:    {self.market_data['iv'].min():.4f}")
            print(f"  Max:    {self.market_data['iv'].max():.4f}")
        
        print(f"\nExpiries: {sorted(self.market_data['expiry'].unique())}")
        
        if self.arbitrage_violations is not None:
            print(f"\nArbitrage Violations:")
            for key, violations in self.arbitrage_violations.items():
                print(f"  {key}: {len(violations)}")
        
        if self.svi_params is not None:
            print(f"\n✓ SVI fitted to {len(self.svi_params)} expiries")
        
        if self.heston_params is not None:
            print(f"✓ Heston calibrated (error={self.heston_params['objective']:.6f})")
        
        print("=" * 60)