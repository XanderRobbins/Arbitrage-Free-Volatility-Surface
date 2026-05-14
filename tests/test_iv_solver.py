"""
Unit tests for implied volatility computation.
"""

import pytest
import numpy as np
from vol_surface.iv_solver import (
    black_scholes_call,
    black_scholes_put,
    vega,
    implied_vol_call,
    implied_vol_put,
    implied_vol,
)


class TestBlackScholes:
    """Test Black-Scholes pricing functions."""

    def test_call_atm(self):
        """ATM call should be approximately 0.5 * S * N(0.5*sigma*sqrt(T))."""
        S, K, T, r, sigma = 100, 100, 1.0, 0.0, 0.2
        price = black_scholes_call(S, K, T, r, sigma)
        assert 7.0 < price < 9.0  # Rough bounds

    def test_put_atm(self):
        """ATM put with r=0 should equal ATM call (put-call parity)."""
        S, K, T, r, sigma = 100, 100, 1.0, 0.0, 0.2
        call = black_scholes_call(S, K, T, r, sigma)
        put = black_scholes_put(S, K, T, r, sigma)
        assert abs(call - put) < 1e-10

    def test_deep_itm_call(self):
        """Deep ITM call should be approximately S - K*e^(-rT)."""
        S, K, T, r, sigma = 150, 100, 1.0, 0.05, 0.2
        price = black_scholes_call(S, K, T, r, sigma)
        intrinsic = S - K * np.exp(-r * T)
        assert price > intrinsic
        assert price < S  # Can't exceed spot

    def test_deep_otm_call(self):
        """Deep OTM call should be near zero."""
        S, K, T, r, sigma = 100, 200, 1.0, 0.05, 0.2
        price = black_scholes_call(S, K, T, r, sigma)
        assert 0 <= price < 1.0

    def test_zero_time(self):
        """At expiry (T=0), price should be max(S-K, 0)."""
        S, K, r, sigma = 110, 100, 0.05, 0.2
        price = black_scholes_call(S, K, 0, r, sigma)
        assert abs(price - 10.0) < 1e-10

    def test_zero_vol(self):
        """With sigma=0, call is deterministic payoff."""
        S, K, T, r = 105, 100, 1.0, 0.0
        price = black_scholes_call(S, K, T, r, 0.0)
        assert abs(price - 5.0) < 1e-10


class TestVega:
    """Test vega (derivative w.r.t. sigma)."""

    def test_vega_positive(self):
        """Vega should always be positive."""
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
        v = vega(S, K, T, r, sigma)
        assert v > 0

    def test_vega_atm_maximum(self):
        """Vega is maximized near ATM."""
        S, T, r, sigma = 100, 1.0, 0.05, 0.2
        vega_atm = vega(S, 100, T, r, sigma)
        vega_otm = vega(S, 120, T, r, sigma)
        vega_itm = vega(S, 80, T, r, sigma)
        assert vega_atm > vega_otm
        assert vega_atm > vega_itm

    def test_vega_zero_time(self):
        """Vega should be zero at expiry."""
        S, K, r, sigma = 100, 100, 0.05, 0.2
        v = vega(S, K, 0, r, sigma)
        assert v == 0.0


class TestImpliedVolCall:
    """Test implied volatility for calls."""

    def test_roundtrip_atm(self):
        """IV solver should recover original sigma."""
        S, K, T, r, sigma_true = 100, 100, 1.0, 0.05, 0.25
        price = black_scholes_call(S, K, T, r, sigma_true)
        sigma_recovered = implied_vol_call(price, S, K, T, r)
        assert abs(sigma_recovered - sigma_true) < 1e-6

    def test_roundtrip_itm(self):
        """IV solver should work for ITM calls."""
        S, K, T, r, sigma_true = 110, 100, 0.5, 0.03, 0.30
        price = black_scholes_call(S, K, T, r, sigma_true)
        sigma_recovered = implied_vol_call(price, S, K, T, r)
        assert abs(sigma_recovered - sigma_true) < 1e-5

    def test_roundtrip_otm(self):
        """IV solver should work for OTM calls."""
        S, K, T, r, sigma_true = 100, 120, 0.25, 0.02, 0.40
        price = black_scholes_call(S, K, T, r, sigma_true)
        sigma_recovered = implied_vol_call(price, S, K, T, r)
        assert abs(sigma_recovered - sigma_true) < 1e-5

    def test_intrinsic_value(self):
        """Price at intrinsic should return ~0 IV."""
        S, K, T, r = 110, 100, 1.0, 0.0
        price = 10.0  # Intrinsic value
        sigma = implied_vol_call(price, S, K, T, r)
        assert sigma == 0.0 or sigma < 0.01

    def test_deep_otm_fails_gracefully(self):
        """Very cheap OTM options may fail to converge."""
        S, K, T, r = 100, 200, 0.1, 0.05
        price = 0.001  # Tiny price
        sigma = implied_vol_call(price, S, K, T, r)
        # Should either converge or return NaN
        assert sigma >= 0 or np.isnan(sigma)

    def test_zero_time(self):
        """T=0 should return 0 IV."""
        S, K, r = 110, 100, 0.05
        price = 10.0
        sigma = implied_vol_call(price, S, K, 0, r)
        assert sigma == 0.0


class TestImpliedVolPut:
    """Test implied volatility for puts."""

    def test_roundtrip_atm(self):
        """IV solver should recover original sigma for puts."""
        S, K, T, r, sigma_true = 100, 100, 1.0, 0.05, 0.25
        price = black_scholes_put(S, K, T, r, sigma_true)
        sigma_recovered = implied_vol_put(price, S, K, T, r)
        assert abs(sigma_recovered - sigma_true) < 1e-6

    def test_roundtrip_itm(self):
        """IV solver should work for ITM puts."""
        S, K, T, r, sigma_true = 90, 100, 0.5, 0.03, 0.30
        price = black_scholes_put(S, K, T, r, sigma_true)
        sigma_recovered = implied_vol_put(price, S, K, T, r)
        assert abs(sigma_recovered - sigma_true) < 1e-5

    def test_put_call_parity_iv(self):
        """Call and put with same strike should have same IV."""
        S, K, T, r, sigma_true = 100, 100, 1.0, 0.05, 0.25
        call_price = black_scholes_call(S, K, T, r, sigma_true)
        put_price = black_scholes_put(S, K, T, r, sigma_true)

        iv_call = implied_vol_call(call_price, S, K, T, r)
        iv_put = implied_vol_put(put_price, S, K, T, r)

        assert abs(iv_call - iv_put) < 1e-6


class TestImpliedVolWrapper:
    """Test convenience wrapper function."""

    def test_call_dispatch(self):
        """implied_vol should dispatch to call solver."""
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.25
        price = black_scholes_call(S, K, T, r, sigma)
        iv = implied_vol(price, S, K, T, r, option_type="call")
        assert abs(iv - sigma) < 1e-6

    def test_put_dispatch(self):
        """implied_vol should dispatch to put solver."""
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.25
        price = black_scholes_put(S, K, T, r, sigma)
        iv = implied_vol(price, S, K, T, r, option_type="put")
        assert abs(iv - sigma) < 1e-6

    def test_invalid_option_type(self):
        """Invalid option_type should raise error."""
        with pytest.raises(ValueError):
            implied_vol(10.0, 100, 100, 1.0, 0.05, option_type="invalid")


class TestEdgeCases:
    """Test edge cases and numerical stability."""

    def test_very_short_expiry(self):
        """Should handle T very close to 0."""
        S, K, r, sigma = 105, 100, 0.05, 0.2
        price = black_scholes_call(S, K, 0.001, r, sigma)
        assert price > 5.0  # Close to intrinsic

    def test_very_high_vol(self):
        """Should handle high volatility (e.g., crypto)."""
        S, K, T, r = 100, 100, 1.0, 0.05
        price = black_scholes_call(S, K, T, r, 1.5)  # 150% vol
        iv = implied_vol_call(price, S, K, T, r)
        assert abs(iv - 1.5) < 0.01

    def test_very_low_vol(self):
        """Should handle very low volatility."""
        S, K, T, r = 100, 100, 1.0, 0.05
        price = black_scholes_call(S, K, T, r, 0.05)  # 5% vol
        iv = implied_vol_call(price, S, K, T, r)
        assert abs(iv - 0.05) < 0.001
