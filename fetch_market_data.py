"""
Fetch real market options data from yfinance for SPY.
Generates hundreds of clean, liquid option quotes.
"""
import pandas as pd
import numpy as np
from datetime import datetime

try:
    import yfinance as yf
except ImportError:
    print("Installing yfinance...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance"])
    import yfinance as yf


def fetch_spy_options(output_file='data/spy_options.csv'):
    """
    Fetch SPY options data for multiple expirations.
    Filters for liquid options (high volume, tight spreads).
    """
    print("Fetching SPY market data...")
    spy = yf.Ticker("SPY")

    # Get current spot price
    spot = spy.info.get('currentPrice', spy.info.get('regularMarketPrice', 590))
    print(f"SPY Spot: ${spot:.2f}")

    # Get available expirations
    expirations = spy.options
    print(f"Available expirations: {len(expirations)}")

    # Select expirations (skip too-short ones, take a good spread)
    selected_expirations = []
    for exp in expirations[:15]:  # First 15 expirations
        exp_date = pd.to_datetime(exp)
        dte = (exp_date - pd.Timestamp.now()).days
        if dte >= 7:  # At least 7 days to expiration
            selected_expirations.append(exp)
            if len(selected_expirations) >= 6:  # Get 6 expirations
                break

    print(f"Selected expirations: {selected_expirations}")

    all_options = []

    for exp in selected_expirations:
        print(f"\nFetching options for expiry: {exp}")

        # Get option chain for this expiration
        chain = spy.option_chain(exp)
        calls = chain.calls.copy()
        puts = chain.puts.copy()

        # Calculate days to expiration
        exp_date = pd.to_datetime(exp)
        today = pd.Timestamp.now()
        dte = (exp_date - today).days + 1  # +1 to avoid zero DTE

        # Convert to years
        T = max(dte / 365.0, 0.001)

        # Process calls
        calls['option_type'] = 'call'
        calls['expiry'] = T
        calls['strike'] = calls['strike'].astype(float)
        calls['price'] = (calls['bid'] + calls['ask']) / 2

        # Process puts
        puts['option_type'] = 'put'
        puts['expiry'] = T
        puts['strike'] = puts['strike'].astype(float)
        puts['price'] = (puts['bid'] + puts['ask']) / 2

        # Combine
        df = pd.concat([calls, puts], ignore_index=True)

        # Filter for liquid options:
        # 1. Bid-ask spread < 5% of mid price
        # 2. Volume > 0 or open interest > 0
        # 3. Strike within 20% of spot
        df['mid_price'] = (df['bid'] + df['ask']) / 2
        df['spread_pct'] = (df['ask'] - df['bid']) / df['mid_price']
        df['moneyness'] = df['strike'] / spot

        liquid = df[
            (df['spread_pct'] < 0.05) &  # Tight spread
            ((df['volume'] > 0) | (df['openInterest'] > 0)) &  # Has activity
            (df['moneyness'] >= 0.8) &  # Not too far OTM put
            (df['moneyness'] <= 1.2)  # Not too far OTM call
        ].copy()

        print(f"  Total options: {len(df)}, Liquid: {len(liquid)}")

        # Select columns
        liquid = liquid[['strike', 'expiry', 'option_type', 'price']].copy()
        liquid.columns = ['strike', 'expiry', 'option_type', 'price']

        all_options.append(liquid)

    # Combine all expirations
    combined = pd.concat(all_options, ignore_index=True)

    # Remove any duplicates and ensure we have variety
    combined = combined.drop_duplicates(subset=['strike', 'expiry', 'option_type'])
    combined = combined.reset_index(drop=True)

    print(f"\n{'='*60}")
    print(f"Total options collected: {len(combined)}")
    print(f"Unique strikes: {combined['strike'].nunique()}")
    print(f"Unique expiries: {combined['expiry'].nunique()}")
    print(f"{'='*60}\n")

    # Save to CSV
    combined.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")

    return combined, spot


if __name__ == "__main__":
    data, spot = fetch_spy_options()
    print("\nSample data:")
    print(data.head(20))
    print(f"\nSpot price: ${spot:.2f}")
