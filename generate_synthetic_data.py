"""
Generate high-quality synthetic options data for backtesting.
Creates realistic volatility surfaces with clean smiles and term structure.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm


def black_scholes_call(S, K, T, r, sigma):
    """Black-Scholes call price."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def black_scholes_put(S, K, T, r, sigma):
    """Black-Scholes put price."""
    return black_scholes_call(S, K, T, r, sigma) - (S - K * np.exp(-r * T))


def generate_vol_surface(spot=580, r=0.04, num_strikes=25, num_expiries=8):
    """
    Generate realistic volatility surface with smile and term structure.

    Returns:
        DataFrame with hundreds of clean option quotes
    """

    # Expiries: 7 days to 1 year
    days_to_expiry = np.array([7, 14, 30, 60, 90, 120, 180, 365])
    expiries = days_to_expiry / 365.0

    # Moneyness grid: 0.85 to 1.15
    moneyness = np.linspace(0.85, 1.15, num_strikes)
    strikes = spot * moneyness

    options = []

    for T in expiries:
        # ATM vol (decreasing term structure)
        atm_vol = 0.22 - 0.08 * np.log(1 + T)  # ~22% short-term, ~14% long-term

        # Smile: strong U-shaped volatility (quadratic in moneyness)
        # At ±15% OTM: 1.0 + 3.0 * 0.15^2 = 1.0 + 0.675 = 1.675 (67.5% smile!)
        vol_surface = atm_vol * (1.0 + 3.0 * (moneyness - 1.0)**2)

        for i, (K, vol) in enumerate(zip(strikes, vol_surface)):
            # Call prices
            call_price = black_scholes_call(spot, K, T, r, vol)
            options.append({
                'strike': K,
                'expiry': T,
                'option_type': 'call',
                'price': call_price,
                'iv_true': vol
            })

            # Put prices
            put_price = black_scholes_put(spot, K, T, r, vol)
            options.append({
                'strike': K,
                'expiry': T,
                'option_type': 'put',
                'price': put_price,
                'iv_true': vol
            })

    df = pd.DataFrame(options)

    # Shuffle to avoid patterns
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\nGenerated {len(df)} synthetic options")
    print(f"Spot: ${spot:.2f}")
    print(f"Strikes: ${strikes.min():.2f} - ${strikes.max():.2f}")
    print(f"Expiries: {days_to_expiry[0]}d - {days_to_expiry[-1]}d")
    print(f"IV range: {df['iv_true'].min():.2%} - {df['iv_true'].max():.2%}\n")

    return df


if __name__ == "__main__":
    data = generate_vol_surface()
    data[['strike', 'expiry', 'option_type', 'price']].to_csv(
        'data/spy_options.csv', index=False
    )
    print("Saved to data/spy_options.csv")
    print("\nSample:")
    print(data.head(10))
