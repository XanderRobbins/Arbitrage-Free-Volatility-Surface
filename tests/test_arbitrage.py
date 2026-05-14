"""
Unit tests for arbitrage checks.
"""

import pytest
import numpy as np
import pandas as pd
from vol_surface.arbitrage import (
    check_put_call_parity,
    check_butterfly_arbitrage,
    check_calendar_arbitrage,
    check_all_arbitrage,
)


class TestPutCallParity:
    """Test put-call parity checks."""

    def test_parity_satisfied(self):
        """No violation when parity holds."""
        S, K, T, r = 100, 100, 1.0, 0.05
        call_price = 10.0
        put_price = call_price - S + K * np.exp(-r * T)

        result = check_put_call_parity(call_price, put_price, S, K, T, r)
        assert result["is_violated"] == False

    def test_parity_violated(self):
        """Detect violation when parity fails."""
        S, K, T, r = 100, 100, 1.0, 0.05
        call_price = 10.0
        put_price = 5.0  # Too cheap

        result = check_put_call_parity(call_price, put_price, S, K, T, r)
        assert result["is_violated"] == True
        assert result["diff"] > 0.01

    def test_zero_rate(self):
        """Parity with r=0: C - P = S - K."""
        S, K, T, r = 100, 100, 1.0, 0.0
        call_price = 8.0
        put_price = 8.0

        result = check_put_call_parity(call_price, put_price, S, K, T, r)
        assert result["is_violated"] == False


class TestButterflyArbitrage:
    """Test butterfly arbitrage checks (convexity)."""

    def test_no_violation_linear_iv(self):
        """Linear IV curve should have no violations."""
        strikes = np.array([90, 95, 100, 105, 110])
        ivs = np.array([0.20, 0.22, 0.24, 0.26, 0.28])  # Linear
        S, T, r = 100, 1.0, 0.05

        violations = check_butterfly_arbitrage(strikes, ivs, S, T, r)
        assert len(violations) == 0

    def test_no_violation_smile(self):
        """Typical smile shape (convex) should be fine."""
        strikes = np.array([90, 95, 100, 105, 110])
        ivs = np.array([0.28, 0.24, 0.22, 0.24, 0.28])  # Smile
        S, T, r = 100, 1.0, 0.05

        violations = check_butterfly_arbitrage(strikes, ivs, S, T, r)
        assert len(violations) == 0

    def test_violation_non_convex(self):
        """Non-convex IV should trigger violation."""
        strikes = np.array([90, 95, 100, 105, 110])
        ivs = np.array([0.22, 0.30, 0.20, 0.30, 0.22])  # Zigzag
        S, T, r = 100, 1.0, 0.05

        violations = check_butterfly_arbitrage(strikes, ivs, S, T, r)
        assert len(violations) > 0

    def test_too_few_strikes(self):
        """Should return empty list for < 3 strikes."""
        strikes = np.array([95, 100])
        ivs = np.array([0.20, 0.25])
        S, T, r = 100, 1.0, 0.05

        violations = check_butterfly_arbitrage(strikes, ivs, S, T, r)
        assert len(violations) == 0

    def test_unsorted_strikes(self):
        """Should handle unsorted strikes."""
        strikes = np.array([100, 90, 110, 95])
        ivs = np.array([0.22, 0.24, 0.26, 0.23])
        S, T, r = 100, 1.0, 0.05

        # Should not crash
        violations = check_butterfly_arbitrage(strikes, ivs, S, T, r)
        assert isinstance(violations, list)


class TestCalendarArbitrage:
    """Test calendar arbitrage checks."""

    def test_no_violation_increasing_variance(self):
        """Total variance increasing with time is OK."""
        iv_surface = {
            0.25: {100: 0.20},
            0.50: {100: 0.22},
            1.00: {100: 0.24},
        }
        strikes = [100]
        r = 0.05

        violations = check_calendar_arbitrage(iv_surface, strikes, r)
        assert len(violations) == 0

    def test_violation_decreasing_variance(self):
        """Total variance decreasing should violate."""
        iv_surface = {
            0.25: {100: 0.30},
            0.50: {100: 0.20},  # Lower variance than T=0.25
        }
        strikes = [100]
        r = 0.05

        violations = check_calendar_arbitrage(iv_surface, strikes, r)
        assert len(violations) > 0
        assert violations[0]["type"] == "calendar"

    def test_single_expiry(self):
        """Single expiry should have no violations."""
        iv_surface = {0.25: {100: 0.20, 105: 0.22}}
        strikes = [100, 105]
        r = 0.05

        violations = check_calendar_arbitrage(iv_surface, strikes, r)
        assert len(violations) == 0

    def test_missing_strikes(self):
        """Should skip strikes not present in all expiries."""
        iv_surface = {
            0.25: {100: 0.20, 105: 0.22},
            0.50: {100: 0.22},  # Missing K=105
        }
        strikes = [100, 105]
        r = 0.05

        violations = check_calendar_arbitrage(iv_surface, strikes, r)
        # Should only check K=100
        assert all(v["strike"] == 100 for v in violations)


class TestCheckAllArbitrage:
    """Test comprehensive arbitrage checking."""

    def test_clean_data_no_violations(self):
        """Well-behaved data should pass all checks."""
        # Use realistic prices that satisfy put-call parity
        from vol_surface.iv_solver import black_scholes_call, black_scholes_put

        S, r = 100, 0.05
        strikes = [95, 100, 105]
        expiries = [0.25, 0.50]

        data_rows = []
        for T in expiries:
            for K in strikes:
                iv = 0.22  # Constant vol
                call = black_scholes_call(S, K, T, r, iv)
                put = black_scholes_put(S, K, T, r, iv)
                data_rows.append(
                    {
                        "strike": K,
                        "expiry": T,
                        "call_price": call,
                        "put_price": put,
                        "iv": iv,
                    }
                )

        data = pd.DataFrame(data_rows)
        violations = check_all_arbitrage(data, S, r, tol=1e-3)

        assert len(violations["put_call_parity"]) == 0
        assert len(violations["butterfly"]) == 0
        assert len(violations["calendar"]) == 0

    def test_detects_parity_violation(self):
        """Should detect put-call parity violations."""
        data = pd.DataFrame(
            {
                "strike": [100],
                "expiry": [1.0],
                "call_price": [10.0],
                "put_price": [2.0],  # Too cheap
                "iv": [0.20],
            }
        )
        S, r = 100, 0.05

        violations = check_all_arbitrage(data, S, r, tol=1e-3)
        assert len(violations["put_call_parity"]) > 0

    def test_missing_columns_handled(self):
        """Should not crash if call/put prices missing."""
        data = pd.DataFrame(
            {
                "strike": [95, 100, 105],
                "expiry": [0.25, 0.25, 0.25],
                "iv": [0.22, 0.20, 0.22],
            }
        )
        S, r = 100, 0.05

        # Should only check butterfly/calendar
        violations = check_all_arbitrage(data, S, r)
        assert len(violations["put_call_parity"]) == 0


class TestNumericalStability:
    """Test numerical edge cases."""

    def test_very_small_tolerance(self):
        """Strict tolerance should detect tiny violations."""
        S, K, T, r = 100, 100, 1.0, 0.05
        call_price = 10.0
        put_price = call_price - S + K * np.exp(-r * T) + 1e-5

        result = check_put_call_parity(call_price, put_price, S, K, T, r, tol=1e-6)
        assert result["is_violated"] == True

    def test_large_tolerance(self):
        """Large tolerance should allow bigger deviations."""
        S, K, T, r = 100, 100, 1.0, 0.05
        call_price = 10.0
        put_price = call_price - S + K * np.exp(-r * T) + 0.01

        result = check_put_call_parity(call_price, put_price, S, K, T, r, tol=0.1)
        assert result["is_violated"] == False
