"""
Publication-quality visualizations of volatility surfaces.
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from vol_surface import VolatilitySurface
from vol_surface.svi import svi_raw_to_iv
from generate_synthetic_data import generate_vol_surface

# Modern styling
plt.style.use('seaborn-v0_8-darkgrid')
mpl.rcParams['figure.figsize'] = (14, 8)
mpl.rcParams['font.size'] = 11
mpl.rcParams['lines.linewidth'] = 2.5
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.alpha'] = 0.3
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['figure.dpi'] = 100

# Color palette
colors = {
    'market': '#1f77b4',      # Blue
    'svi': '#ff7f0e',         # Orange
    'heston': '#2ca02c',      # Green
    'accent': '#d62728'       # Red
}


def plot_smile_professional(surface, expiry, save_path=None, show=True):
    """Plot volatility smile with SVI overlay."""
    # Filter data for this expiry
    mask = np.isclose(surface.market_data['expiry'], expiry, atol=0.001)
    data_expiry = surface.market_data[mask].copy()

    if len(data_expiry) == 0:
        print(f"No data for expiry {expiry}")
        return

    # Sort by strike
    data_expiry = data_expiry.sort_values('strike')

    fig, ax = plt.subplots(figsize=(12, 7))

    # Market smile
    ax.scatter(data_expiry['strike'], data_expiry['iv'] * 100,
               s=100, alpha=0.7, color=colors['market'],
               edgecolors='darkblue', linewidth=1.5, label='Market', zorder=3)

    # SVI fit if available
    if surface.svi_params and expiry in surface.svi_params:
        strike_grid = np.linspace(data_expiry['strike'].min(),
                                   data_expiry['strike'].max(), 200)
        params = surface.svi_params[expiry]
        F = surface.S * np.exp(surface.r * expiry)
        k = np.log(strike_grid / F)
        svi_ivs = svi_raw_to_iv(k, expiry, params['a'], params['b'],
                                params['rho'], params['m'], params['sigma'])
        ax.plot(strike_grid, svi_ivs * 100, color=colors['svi'],
                linewidth=3, label='SVI Fit', zorder=2, alpha=0.9)

    ax.set_xlabel('Strike Price', fontsize=12, fontweight='bold')
    ax.set_ylabel('Implied Volatility (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Volatility Smile (T = {expiry:.3f} years = {expiry*365:.0f} days)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=11, framealpha=0.95)
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('white')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


def plot_surface_3d_professional(surface, model='market', save_path=None, show=True):
    """Publication-quality 3D volatility surface."""
    from mpl_toolkits.mplot3d import Axes3D
    from scipy.interpolate import griddata

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Get unique strikes and expiries
    strikes = np.sort(surface.market_data['strike'].unique())
    expiries = np.sort(surface.market_data['expiry'].unique())

    if model == 'svi' and surface.svi_params:
        from scipy.interpolate import interp1d

        # Create smooth SVI surface mesh
        K_fine = np.linspace(strikes.min(), strikes.max(), 60)
        T_fine = np.linspace(expiries.min(), expiries.max(), 50)
        K_mesh, T_mesh = np.meshgrid(K_fine, T_fine)

        IV_mesh = np.zeros_like(K_mesh)

        # Extract SVI parameters and interpolate them across expiries
        T_params = sorted(surface.svi_params.keys())
        svi_dict = {T: surface.svi_params[T] for T in T_params}

        # Interpolate each SVI parameter across expiries
        param_names = ['a', 'b', 'rho', 'm', 'sigma']
        param_interp = {}

        for param in param_names:
            param_values = [svi_dict[T][param] for T in T_params]
            # Linear interpolation of parameters
            param_interp[param] = interp1d(T_params, param_values,
                                          kind='cubic', fill_value='extrapolate')

        # Compute smooth surface
        for i, T in enumerate(T_fine):
            # Clamp to data range to avoid extrapolation artifacts
            T_clamped = np.clip(T, T_params[0], T_params[-1])

            # Get interpolated params
            a = float(param_interp['a'](T_clamped))
            b = float(param_interp['b'](T_clamped))
            rho = float(np.clip(param_interp['rho'](T_clamped), -0.99, 0.99))
            m = float(param_interp['m'](T_clamped))
            sigma = float(np.clip(param_interp['sigma'](T_clamped), 0.01, 1.0))

            for j, K in enumerate(K_fine):
                F = surface.S * np.exp(surface.r * T)
                k = np.log(K / F)
                iv = svi_raw_to_iv(k, T, a, b, rho, m, sigma)
                IV_mesh[i, j] = float(iv) * 100

        # Plot smooth surface with better visibility
        surf = ax.plot_surface(K_mesh, T_mesh * 365, IV_mesh,
                              cmap='viridis', alpha=0.8,
                              edgecolor='black', antialiased=True,
                              rstride=2, cstride=2, linewidth=0.2,
                              shade=True)

        cbar = plt.colorbar(surf, ax=ax, pad=0.1, shrink=0.8)
        cbar.set_label('IV (%)', fontweight='bold')

    else:
        # Market data: scatter points
        K_grid = []
        T_grid = []
        IV_grid = []

        for T in expiries:
            for K in strikes:
                mask = (np.isclose(surface.market_data['strike'], K, atol=0.1) &
                        np.isclose(surface.market_data['expiry'], T, atol=0.001))
                data_point = surface.market_data[mask]

                if len(data_point) > 0 and model in data_point.columns:
                    iv = data_point[model].iloc[0]
                    if not np.isnan(iv):
                        K_grid.append(K)
                        T_grid.append(T)
                        IV_grid.append(iv * 100)

        if len(IV_grid) == 0:
            print(f"No data for {model} model")
            return

        K_grid = np.array(K_grid)
        T_grid = np.array(T_grid)
        IV_grid = np.array(IV_grid)

        # Scatter plot for market data
        scatter = ax.scatter(K_grid, T_grid * 365, IV_grid,
                            c=IV_grid, cmap='plasma',
                            s=100, alpha=0.6, edgecolors='black', linewidth=1)

        cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
        cbar.set_label('IV (%)', fontweight='bold')

    ax.set_xlabel('Strike Price', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_ylabel('Days to Maturity', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_zlabel('Implied Volatility (%)', fontsize=11, fontweight='bold', labelpad=10)

    title_map = {'market': 'Market IV Surface', 'iv': 'Market IV Surface', 'svi': 'SVI Fitted Surface'}
    ax.set_title(f'{title_map.get(model, model)}',
                 fontsize=14, fontweight='bold', pad=20)

    # Better viewing angle to show smile curvature
    if model == 'svi':
        ax.view_init(elev=20, azim=-60)  # Emphasize strike dimension
    else:
        ax.view_init(elev=25, azim=45)
    fig.patch.set_facecolor('white')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


def plot_term_structure_professional(surface, moneyness=1.0, save_path=None, show=True):
    """Term structure of volatility."""
    # Group by expiry
    term_data = surface.market_data.groupby('expiry').agg({
        'iv': 'mean',
        'strike': 'mean'
    }).reset_index()

    term_data = term_data.sort_values('expiry')
    days = term_data['expiry'] * 365

    fig, ax = plt.subplots(figsize=(12, 7))

    # Market term structure
    ax.plot(days, term_data['iv'] * 100, marker='o', markersize=8,
            linewidth=2.5, color=colors['market'], label='Market',
            alpha=0.8, zorder=3)
    ax.scatter(days, term_data['iv'] * 100, s=150, color=colors['market'],
               edgecolors='darkblue', linewidth=1.5, zorder=4, alpha=0.7)

    # SVI ATM if available
    if surface.svi_params:
        svi_ivs = []
        for T in term_data['expiry']:
            if T in surface.svi_params:
                params = surface.svi_params[T]
                F = surface.S * np.exp(surface.r * T)
                k_atm = np.log(surface.S / F)  # log-moneyness at ATM
                svi_iv = svi_raw_to_iv(k_atm, T, params['a'], params['b'],
                                       params['rho'], params['m'], params['sigma'])
                svi_ivs.append(float(svi_iv) * 100)
            else:
                svi_ivs.append(np.nan)

        if any(~np.isnan(svi_ivs)):
            ax.plot(days, svi_ivs, marker='s', markersize=8,
                   linewidth=2.5, linestyle='--', color=colors['svi'],
                   label='SVI ATM', alpha=0.8, zorder=2)

    ax.set_xlabel('Days to Maturity', fontsize=12, fontweight='bold')
    ax.set_ylabel('Implied Volatility (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Volatility Term Structure (Moneyness = {moneyness:.2f})',
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=11, framealpha=0.95)
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('white')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


def main(show_plots=True):
    print("="*70)
    print("VOLATILITY SURFACE BACKTEST - PUBLICATION QUALITY")
    print("="*70)

    # Generate clean synthetic data
    print("\n[1/4] Generating 400+ synthetic options...")
    data = generate_vol_surface(num_strikes=25, num_expiries=8)

    # Initialize surface
    print("[2/4] Computing implied volatilities...")
    surface = VolatilitySurface(S=580, r=0.04)
    surface.load_data(data[['strike', 'expiry', 'option_type', 'price']])
    surface.compute_ivs()

    print(f"     [OK] Computed {len(surface.market_data)} option IVs")

    # Fit SVI
    print("[3/4] Fitting SVI parameterization...")
    surface.fit_svi()
    print(f"     [OK] Fitted {len(surface.svi_params)} expiries")

    # Visualize
    print("[4/4] Generating publication-quality plots...\n")

    # Smile plot - show multiple expiries
    print("  > Volatility smile...")
    unique_expiries = sorted(surface.market_data['expiry'].unique())
    # Plot the middle expiry to show smile clearly
    expiry_idx = len(unique_expiries) // 2
    expiry = unique_expiries[expiry_idx]
    plot_smile_professional(surface, expiry, show=show_plots)

    # Quick check: show smile exists
    print("  > Checking smile structure...")
    sample_expiry = unique_expiries[len(unique_expiries) // 2]
    sample_data = surface.market_data[np.isclose(surface.market_data['expiry'], sample_expiry, atol=0.001)]
    if len(sample_data) > 0:
        sorted_data = sample_data.sort_values('strike')
        min_iv = sorted_data['iv'].min()
        max_iv = sorted_data['iv'].max()
        atm_data = sorted_data[np.isclose(sorted_data['strike'], surface.S, atol=10)]
        if len(atm_data) > 0:
            atm_iv = atm_data['iv'].mean()
            smile_ratio = max_iv / atm_iv if atm_iv > 0 else 1.0
            print(f"     ATM IV: {atm_iv*100:.2f}%, Wings IV: {max_iv*100:.2f}%, Smile ratio: {smile_ratio:.2f}x")

    # 3D surfaces
    print("  > Market surface...")
    plot_surface_3d_professional(surface, model='iv', show=show_plots)

    print("  > SVI surface...")
    plot_surface_3d_professional(surface, model='svi', show=show_plots)

    # Term structure
    print("  > Term structure...")
    plot_term_structure_professional(surface, moneyness=1.0, show=show_plots)

    # Summary
    surface.summary()
    print("\n" + "="*70)
    print("[DONE] ALL PLOTS COMPLETED - Professional quality backtest")
    print("="*70)


if __name__ == "__main__":
    import sys
    # Pass show_plots=False to disable interactive display (useful for testing)
    # or pass show_plots=True to display plots (default for interactive use)
    show = '--no-show' not in sys.argv
    main(show_plots=show)
