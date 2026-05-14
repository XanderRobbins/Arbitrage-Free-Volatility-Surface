"""
Main VolatilitySurface class: orchestrates IV computation, SVI fitting,
arbitrage checks, and Heston calibration.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

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

    def __repr__(self) -> str:
        n = len(self.market_data) if self.market_data is not None else 0
        return f"VolatilitySurface(S={self.S}, r={self.r}, n_options={n})"

    def load_data(self, data):
        """
        Load market option data.

        Parameters
        ----------
        data : pd.DataFrame
            Must contain: ['strike', 'expiry', 'option_type', 'price']
            Optional: ['bid', 'ask', 'volume']
        """
        required_cols = ["strike", "expiry", "option_type", "price"]
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

        if "iv" in self.market_data.columns and not overwrite:
            print("IVs already computed. Use overwrite=True to recompute.")
            return self

        print("Computing implied volatilities...")

        ivs = []
        for _, row in self.market_data.iterrows():
            iv = implied_vol(
                price=row["price"],
                S=self.S,
                K=row["strike"],
                T=row["expiry"],
                r=self.r,
                option_type=row["option_type"],
            )
            ivs.append(iv)

        self.market_data["iv"] = ivs

        # Filter out failed computations
        valid_mask = ~np.isnan(self.market_data["iv"])
        n_invalid = (~valid_mask).sum()

        if n_invalid > 0:
            print(f"Warning: {n_invalid} IVs failed to converge (removed)")
            self.market_data = self.market_data[valid_mask]

        print(f"OK - Computed {len(self.market_data)} implied volatilities")
        return self

    def check_arbitrage(self, tol=1e-3):
        """
        Run all arbitrage checks.

        Parameters
        ----------
        tol : float
            Tolerance for violations
        """
        if "iv" not in self.market_data.columns:
            raise ValueError("Must compute IVs first using .compute_ivs()")

        print("Checking for arbitrage violations...")

        self.arbitrage_violations = check_all_arbitrage(
            self.market_data, self.S, self.r, tol
        )

        # Print summary
        for key, violations in self.arbitrage_violations.items():
            n = len(violations)
            status = "X" if n > 0 else "OK"
            print(f"{status} {key}: {n} violations")

        return self

    def fit_svi(self, method="least_squares"):
        """
        Fit SVI parameterization to the surface.

        Parameters
        ----------
        method : str
            'least_squares' or 'differential_evolution'
        """
        if "iv" not in self.market_data.columns:
            raise ValueError("Must compute IVs first")

        print("Fitting SVI parameterization...")

        self.svi_params = fit_svi_surface(self.market_data, self.S, self.r, method)

        # Print diagnostics
        for T, params in self.svi_params.items():
            print(
                f"  T={T:.3f}: RMSE={params['rmse']:.6f}, success={params['success']}"
            )

        print(f"OK - Fitted SVI to {len(self.svi_params)} expiries")
        return self

    def calibrate_heston(self, method="local", initial_guess=None):
        """
        Calibrate Heston stochastic volatility model.

        Parameters
        ----------
        method : str
            'local' or 'global' optimization
        initial_guess : dict, optional
            Initial guess for parameters
        """
        if "iv" not in self.market_data.columns:
            raise ValueError("Must compute IVs first")

        print("Calibrating Heston model...")

        self.heston_params = calibrate_heston(
            self.market_data, self.S, self.r, initial_guess, method
        )

        print(
            f"OK - Heston calibration complete (error={self.heston_params['objective']:.6f})"
        )
        print(
            f"  κ={self.heston_params['kappa']:.4f}, "
            f"θ={self.heston_params['theta']:.4f}, "
            f"ξ={self.heston_params['xi']:.4f}"
        )
        print(
            f"  ρ={self.heston_params['rho']:.4f}, "
            f"v₀={self.heston_params['v0']:.4f}"
        )

        return self

    def plot_smile(
        self, expiry, include_svi=True, include_heston=True, figsize=(10, 6)
    ):
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
        if "iv" not in self.market_data.columns:
            raise ValueError("Must compute IVs first")

        # Filter data for this expiry
        tol = 1e-6
        slice_data = self.market_data[np.abs(self.market_data["expiry"] - expiry) < tol]

        if len(slice_data) == 0:
            print(f"No data found for expiry T={expiry}")
            return

        slice_data = slice_data.sort_values("strike")
        strikes = slice_data["strike"].values
        ivs = slice_data["iv"].values

        plt.figure(figsize=figsize)

        # Market data
        plt.plot(strikes, ivs, "o", label="Market", markersize=8, alpha=0.7)

        # Create fine grid for smooth curves
        F = self.S * np.exp(self.r * expiry)
        k_fine = np.log(np.linspace(strikes.min(), strikes.max(), 100) / F)
        K_fine = F * np.exp(k_fine)

        # SVI fit
        if include_svi and self.svi_params is not None and expiry in self.svi_params:
            params = self.svi_params[expiry]
            iv_svi = svi_raw_to_iv(
                k_fine,
                expiry,
                params["a"],
                params["b"],
                params["rho"],
                params["m"],
                params["sigma"],
            )
            plt.plot(
                K_fine,
                iv_svi,
                "-",
                label=f'SVI (RMSE={params["rmse"]:.4f})',
                linewidth=2,
            )

        # Heston fit
        if include_heston and self.heston_params is not None:
            from .iv_solver import implied_vol_call

            model = self.heston_params["model"]

            iv_heston = []
            for K in K_fine:
                price = model.price_call_cos(self.S, K, expiry, self.r, N=64)
                iv = implied_vol_call(price, self.S, K, expiry, self.r)
                iv_heston.append(iv)

            plt.plot(K_fine, iv_heston, "--", label="Heston", linewidth=2, alpha=0.8)

        plt.xlabel("Strike Price", fontsize=12)
        plt.ylabel("Implied Volatility", fontsize=12)
        plt.title(f"Volatility Smile (T={expiry:.3f} years)", fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_surface_3d(self, model="market", figsize=(12, 8), elev=20, azim=45):
        """
        Plot 3D volatility surface.

        Parameters
        ----------
        model : str
            'market', 'svi', or 'heston'
        """
        if "iv" not in self.market_data.columns:
            raise ValueError("Must compute IVs first")

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        if model == "market":
            # Scatter plot of market IVs
            strikes = self.market_data["strike"].values
            expiries = self.market_data["expiry"].values
            ivs = self.market_data["iv"].values

            ax.scatter(
                strikes,
                expiries,
                ivs,
                c=ivs,
                cmap="viridis",
                s=50,
                alpha=0.7,
                edgecolors="k",
                linewidth=0.5,
            )
            title = "Market Implied Volatility Surface"

        elif model == "svi":
            if self.svi_params is None:
                raise ValueError("Must fit SVI first using .fit_svi()")

            # Create mesh
            expiries = sorted(self.svi_params.keys())
            strikes_per_expiry = []

            for T in expiries:
                slice_data = self.market_data[
                    np.abs(self.market_data["expiry"] - T) < 1e-6
                ]
                strikes_per_expiry.append(slice_data["strike"].values)

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
                IV_mesh[i, :] = svi_raw_to_iv(
                    k,
                    T,
                    params["a"],
                    params["b"],
                    params["rho"],
                    params["m"],
                    params["sigma"],
                )

            ax.plot_surface(
                K_mesh, T_mesh, IV_mesh, cmap="viridis", alpha=0.8, edgecolor="none"
            )
            title = "SVI Fitted Volatility Surface"

        elif model == "heston":
            if self.heston_params is None:
                raise ValueError("Must calibrate Heston first")

            # Create mesh
            from .iv_solver import implied_vol_call

            expiries = self.market_data["expiry"].unique()
            strikes_range = self.market_data["strike"].values
            K_min, K_max = strikes_range.min(), strikes_range.max()

            K_grid = np.linspace(K_min, K_max, 30)
            T_grid = np.linspace(expiries.min(), expiries.max(), 20)

            K_mesh, T_mesh = np.meshgrid(K_grid, T_grid)
            IV_mesh = np.zeros_like(K_mesh)

            model_obj = self.heston_params["model"]

            for i in range(len(T_grid)):
                for j in range(len(K_grid)):
                    T = T_mesh[i, j]
                    K = K_mesh[i, j]
                    price = model_obj.price_call_cos(self.S, K, T, self.r, N=64)
                    IV_mesh[i, j] = implied_vol_call(price, self.S, K, T, self.r)

            ax.plot_surface(
                K_mesh, T_mesh, IV_mesh, cmap="plasma", alpha=0.8, edgecolor="none"
            )
            title = "Heston Model Volatility Surface"

        else:
            raise ValueError("model must be 'market', 'svi', or 'heston'")

        ax.set_xlabel("Strike Price", fontsize=11)
        ax.set_ylabel("Time to Maturity", fontsize=11)
        ax.set_zlabel("Implied Volatility", fontsize=11)
        ax.set_title(title, fontsize=14, pad=20)
        ax.view_init(elev=elev, azim=azim)

        plt.tight_layout()
        plt.show()

    def compare_models(self, expiry=None, metric="rmse"):
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

        expiries_to_check = (
            [expiry] if expiry is not None else self.market_data["expiry"].unique()
        )

        for T in expiries_to_check:
            slice_data = self.market_data[np.abs(self.market_data["expiry"] - T) < 1e-6]

            if len(slice_data) == 0:
                continue

            strikes = slice_data["strike"].values
            iv_market = slice_data["iv"].values

            # SVI predictions
            if T in self.svi_params:
                params = self.svi_params[T]
                F = self.S * np.exp(self.r * T)
                k = np.log(strikes / F)
                iv_svi = svi_raw_to_iv(
                    k,
                    T,
                    params["a"],
                    params["b"],
                    params["rho"],
                    params["m"],
                    params["sigma"],
                )
            else:
                iv_svi = np.full_like(iv_market, np.nan)

            # Heston predictions
            model = self.heston_params["model"]
            iv_heston = []
            for K in strikes:
                price = model.price_call_cos(self.S, K, T, self.r, N=64)
                iv = implied_vol_call(price, self.S, K, T, self.r)
                iv_heston.append(iv)
            iv_heston = np.array(iv_heston)

            # Filter out NaN values from both SVI and Heston, keeping aligned pairs
            valid_mask = ~(np.isnan(iv_svi) | np.isnan(iv_heston) | np.isnan(iv_market))
            iv_svi = iv_svi[valid_mask]
            iv_heston = iv_heston[valid_mask]
            iv_market = iv_market[valid_mask]

            # Skip if no valid data remains
            if len(iv_market) == 0:
                continue

            # Compute metrics
            if metric == "rmse":
                svi_error = np.sqrt(np.mean((iv_svi - iv_market) ** 2))
                heston_error = np.sqrt(np.mean((iv_heston - iv_market) ** 2))
            elif metric == "mape":
                svi_error = np.mean(np.abs((iv_svi - iv_market) / iv_market)) * 100
                heston_error = (
                    np.mean(np.abs((iv_heston - iv_market) / iv_market)) * 100
                )
            else:
                raise ValueError("metric must be 'rmse' or 'mape'")

            results.append(
                {
                    "expiry": T,
                    "n_options": len(strikes),
                    f"SVI_{metric}": svi_error,
                    f"Heston_{metric}": heston_error,
                }
            )

        df = pd.DataFrame(results)
        print(f"\nModel Comparison ({metric.upper()}):")
        print(df.to_string(index=False))
        print(f"\nAverage {metric.upper()}:")
        print(f"  SVI:    {df[f'SVI_{metric}'].mean():.6f}")
        print(f"  Heston: {df[f'Heston_{metric}'].mean():.6f}")

        return df

    def greek_surface(self, greek="delta", figsize=(12, 8)):
        """
        Plot a Greek across the volatility surface.

        Parameters
        ----------
        greek : str
            'delta' or 'vega'
        figsize : tuple
            Figure size

        Returns
        -------
        np.ndarray
            Greek values on the surface grid
        """
        if self.heston_params is None:
            raise ValueError("Must calibrate Heston first")

        if greek not in ("delta", "vega"):
            raise ValueError("greek must be 'delta' or 'vega'")

        model = self.heston_params["model"]

        # Create mesh
        expiries = self.market_data["expiry"].unique()
        strikes_range = self.market_data["strike"].values
        K_min, K_max = strikes_range.min(), strikes_range.max()

        K_grid = np.linspace(K_min, K_max, 30)
        T_grid = np.linspace(expiries.min(), expiries.max(), 20)

        K_mesh, T_mesh = np.meshgrid(K_grid, T_grid)
        greek_mesh = np.zeros_like(K_mesh)

        bump_size_spot = 0.0001  # 0.01% bump for delta
        bump_size_vol = 0.01  # 1% bump for vega

        for i in range(len(T_grid)):
            for j in range(len(K_grid)):
                T = T_mesh[i, j]
                K = K_mesh[i, j]

                if greek == "delta":
                    # Finite difference: (C(S+h) - C(S)) / h
                    S_bump = self.S * (1 + bump_size_spot)
                    price_up = model.price_call_cos(S_bump, K, T, self.r, N=64)
                    price = model.price_call_cos(self.S, K, T, self.r, N=64)
                    h = S_bump - self.S
                    greek_mesh[i, j] = (price_up - price) / h

                else:  # vega
                    # Finite difference: (C(v+h) - C(v)) / h for a small vol bump
                    # Use perturbation of v0 in Heston
                    kappa, theta, xi, rho, v0 = (
                        model.kappa,
                        model.theta,
                        model.xi,
                        model.rho,
                        model.v0,
                    )

                    v0_perturbed = v0 * (1 + bump_size_vol)
                    model_bumped = type(model)(kappa, theta, xi, rho, v0_perturbed)

                    price_up = model_bumped.price_call_cos(self.S, K, T, self.r, N=64)
                    price = model.price_call_cos(self.S, K, T, self.r, N=64)
                    h = v0 * bump_size_vol
                    greek_mesh[i, j] = (price_up - price) / h

        # Plot
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        ax.plot_surface(
            K_mesh, T_mesh, greek_mesh, cmap="RdYlGn", alpha=0.8, edgecolor="none"
        )

        ax.set_xlabel("Strike Price", fontsize=11)
        ax.set_ylabel("Time to Maturity", fontsize=11)
        ax.set_zlabel(greek.capitalize(), fontsize=11)
        ax.set_title(
            f"{greek.capitalize()} Surface (Heston Model)", fontsize=14, pad=20
        )

        plt.tight_layout()
        plt.show()

        return greek_mesh

    def term_structure(self, moneyness=1.0):
        """
        Plot ATM (or near-ATM) implied volatility term structure.

        Parameters
        ----------
        moneyness : float
            K/S ratio to query (default 1.0 = ATM)

        Returns
        -------
        pd.DataFrame
            With columns ['expiry', 'atm_iv']
        """
        if "iv" not in self.market_data.columns:
            raise ValueError("Must compute IVs first")

        results = []

        for T in sorted(self.market_data["expiry"].unique()):
            slice_data = self.market_data[np.abs(self.market_data["expiry"] - T) < 1e-6]

            # Find strike closest to target moneyness
            target_strike = moneyness * self.S
            slice_data = slice_data.copy()
            slice_data["strike_diff"] = np.abs(slice_data["strike"] - target_strike)
            closest_row = slice_data.loc[slice_data["strike_diff"].idxmin()]

            results.append(
                {
                    "expiry": T,
                    "strike": closest_row["strike"],
                    "atm_iv": closest_row["iv"],
                }
            )

        df = pd.DataFrame(results)

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(
            df["expiry"], df["atm_iv"], "o-", linewidth=2, markersize=8, label="Market"
        )

        # Overlay SVI ATM vol if available
        if self.svi_params is not None:
            svi_ivs = []
            for T in df["expiry"].values:
                if T in self.svi_params:
                    params = self.svi_params[T]
                    # At-the-money: k = log(1) = 0
                    w_atm = params["a"] + params["b"] * params["sigma"] * np.sqrt(
                        1 - params["rho"] ** 2
                    )
                    iv_atm = np.sqrt(w_atm / T)
                    svi_ivs.append(iv_atm)
                else:
                    svi_ivs.append(np.nan)

            plt.plot(
                df["expiry"],
                svi_ivs,
                "s--",
                linewidth=2,
                markersize=7,
                label="SVI ATM",
                alpha=0.8,
            )

        plt.xlabel("Time to Maturity (years)", fontsize=12)
        plt.ylabel("Implied Volatility", fontsize=12)
        plt.title(f"Volatility Term Structure (Moneyness={moneyness:.2f})", fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

        return df[["expiry", "atm_iv"]]

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

        if "iv" in self.market_data.columns:
            print("\nImplied Volatility Stats:")
            print(f"  Mean:   {self.market_data['iv'].mean():.4f}")
            print(f"  Median: {self.market_data['iv'].median():.4f}")
            print(f"  Min:    {self.market_data['iv'].min():.4f}")
            print(f"  Max:    {self.market_data['iv'].max():.4f}")

        print(f"\nExpiries: {sorted(self.market_data['expiry'].unique())}")

        if self.arbitrage_violations is not None:
            print("\nArbitrage Violations:")
            for key, violations in self.arbitrage_violations.items():
                print(f"  {key}: {len(violations)}")

        if self.svi_params is not None:
            print(f"\nOK - SVI fitted to {len(self.svi_params)} expiries")

        if self.heston_params is not None:
            print(f"OK - Heston calibrated (error={self.heston_params['objective']:.6f})")

        print("=" * 60)
