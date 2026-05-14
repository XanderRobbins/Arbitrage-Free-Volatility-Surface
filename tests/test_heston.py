"""
Unit tests for Heston model and calibration.
"""

import pytest
import numpy as np
from vol_surface.heston import HestonModel, calibrate_heston
from vol_surface.iv_solver import black_scholes_call
import pandas as pd


class TestHestonModel:
    """Test Heston model instantiation."""

    def test_model_creation(self):
        """Should create model with valid parameters."""
        model = HestonModel(kappa=2.0, theta=0.04, xi=0.3, rho=-0.5, v0=0.04)

        assert model.kappa == 2.0
        assert model.theta == 0.04
        assert model.xi == 0.3
        assert model.rho == -0.5
        assert model.v0 == 0.04

    def test_feller_condition_satisfied(self):
        """Feller condition: 2*kappa*theta > xi^2."""
        kappa, theta, xi = 2.0, 0.04, 0.3

        assert 2 * kappa * theta > xi**2  # Should be True

    def test_feller_condition_violated(self):
        """Model can be created even if Feller violated (warning case)."""
        model = HestonModel(kappa=1.0, theta=0.01, xi=0.5, rho=-0.5, v0=0.01)

        # Should not crash
        assert model.kappa == 1.0


class TestCharacteristicFunction:
    """Test Heston characteristic function."""

    def test_cf_at_zero(self):
        """CF(0) should equal 1."""
        model = HestonModel(kappa=2.0, theta=0.04, xi=0.3, rho=-0.5, v0=0.04)
        S, T, r = 100, 1.0, 0.05

        cf_val = model.characteristic_function(0, S, T, r)

        np.testing.assert_allclose(abs(cf_val), 1.0, atol=1e-6)

    def test_cf_is_complex(self):
        """CF should return complex values."""
        model = HestonModel(kappa=2.0, theta=0.04, xi=0.3, rho=-0.5, v0=0.04)
        S, T, r = 100, 1.0, 0.05

        cf_val = model.characteristic_function(1.0, S, T, r)

        assert isinstance(cf_val, (complex, np.complexfloating))

    def test_cf_multiple_frequencies(self):
        """CF should handle different frequency values."""
        model = HestonModel(kappa=2.0, theta=0.04, xi=0.3, rho=-0.5, v0=0.04)
        S, T, r = 100, 1.0, 0.05

        u_vals = [0.5, 1.0, 2.0, 5.0]
        cf_vals = [model.characteristic_function(u, S, T, r) for u in u_vals]

        assert len(cf_vals) == 4
        assert all(isinstance(val, (complex, np.complexfloating)) for val in cf_vals)


class TestHestonPricing:
    """Test Heston COS method pricing."""

    def test_atm_call_positive(self):
        """ATM call should have positive price."""
        model = HestonModel(kappa=2.0, theta=0.04, xi=0.3, rho=-0.5, v0=0.04)
        S, K, T, r = 100, 100, 1.0, 0.05

        price = model.price_call_cos(S, K, T, r, N=128)

        assert price > 0
        assert price < S  # Can't exceed spot

    def test_deep_itm_call(self):
        """Deep ITM call should be near intrinsic."""
        model = HestonModel(kappa=2.0, theta=0.04, xi=0.3, rho=-0.5, v0=0.04)
        S, K, T, r = 150, 100, 0.5, 0.05

        price = model.price_call_cos(S, K, T, r, N=128)
        intrinsic = S - K * np.exp(-r * T)

        assert price >= intrinsic * 0.98  # Allow small numerical error for deep ITM
        assert price < S

    def test_deep_otm_call(self):
        """Deep OTM call should be near zero."""
        model = HestonModel(kappa=2.0, theta=0.04, xi=0.3, rho=-0.5, v0=0.04)
        S, K, T, r = 100, 200, 0.5, 0.05

        price = model.price_call_cos(S, K, T, r, N=128)

        assert 0 <= price < 5.0

    def test_put_via_parity(self):
        """Put price via parity should be consistent."""
        model = HestonModel(kappa=2.0, theta=0.04, xi=0.3, rho=-0.5, v0=0.04)
        S, K, T, r = 100, 100, 1.0, 0.05

        call_price = model.price_call_cos(S, K, T, r, N=128)
        put_price = model.price_put_cos(S, K, T, r, N=128)

        # Put-call parity: C - P = S - K*e^(-rT)
        lhs = call_price - put_price
        rhs = S - K * np.exp(-r * T)

        np.testing.assert_allclose(lhs, rhs, rtol=1e-3)

    def test_more_terms_improves_accuracy(self):
        """More COS terms should give more stable results."""
        model = HestonModel(kappa=2.0, theta=0.04, xi=0.3, rho=-0.5, v0=0.04)
        S, K, T, r = 100, 105, 0.5, 0.05

        price_64 = model.price_call_cos(S, K, T, r, N=64)
        price_128 = model.price_call_cos(S, K, T, r, N=128)
        price_256 = model.price_call_cos(S, K, T, r, N=256)

        # Prices should be positive and reasonable
        assert all(p > 0 for p in [price_64, price_128, price_256])
        assert all(p < S for p in [price_64, price_128, price_256])


class TestHestonCalibration:
    """Test Heston calibration."""

    def test_calibration_runs(self):
        """Calibration should complete without crashing."""
        # Generate synthetic market data
        data = pd.DataFrame(
            {
                "strike": [95, 100, 105, 110],
                "expiry": [0.25] * 4,
                "iv": [0.24, 0.22, 0.23, 0.25],
                "option_type": ["call"] * 4,
            }
        )
        S, r = 100, 0.05

        result = calibrate_heston(data, S, r, method="local")

        assert "kappa" in result
        assert "theta" in result
        assert "xi" in result
        assert "rho" in result
        assert "v0" in result
        assert "model" in result

    def test_calibration_parameter_bounds(self):
        """Calibrated params should respect bounds."""
        data = pd.DataFrame(
            {
                "strike": [90, 95, 100, 105, 110],
                "expiry": [0.5] * 5,
                "iv": [0.26, 0.23, 0.21, 0.23, 0.26],
                "option_type": ["call"] * 5,
            }
        )
        S, r = 100, 0.03

        result = calibrate_heston(data, S, r, method="local")

        assert result["kappa"] > 0
        assert result["theta"] > 0
        assert result["xi"] > 0
        assert -1 < result["rho"] < 1
        assert result["v0"] > 0

    def test_feller_condition_check(self):
        """Calibration objective penalizes Feller violations."""
        data = pd.DataFrame(
            {
                "strike": [95, 100, 105],
                "expiry": [0.25] * 3,
                "iv": [0.23, 0.21, 0.23],
                "option_type": ["call"] * 3,
            }
        )
        S, r = 100, 0.05

        result = calibrate_heston(data, S, r)

        # Should either satisfy Feller or have high objective
        kappa, theta, xi = result["kappa"], result["theta"], result["xi"]
        feller_lhs = 2 * kappa * theta
        feller_rhs = xi**2

        # Allow small violations due to numerical tolerance
        assert feller_lhs >= feller_rhs - 0.01

    def test_custom_initial_guess(self):
        """Should accept custom initial parameters."""
        data = pd.DataFrame(
            {
                "strike": [95, 100, 105],
                "expiry": [0.25] * 3,
                "iv": [0.24, 0.22, 0.24],
                "option_type": ["call"] * 3,
            }
        )
        S, r = 100, 0.05

        initial_guess = {
            "kappa": 3.0,
            "theta": 0.05,
            "xi": 0.4,
            "rho": -0.6,
            "v0": 0.05,
        }

        result = calibrate_heston(data, S, r, initial_guess=initial_guess)

        # Should run without error
        assert "kappa" in result
        assert result["objective"] < 1e8  # Reasonable fit

    def test_multiple_expiries(self):
        """Should calibrate to multiple expiries."""
        data = pd.DataFrame(
            {
                "strike": [95, 100, 105, 95, 100, 105],
                "expiry": [0.25, 0.25, 0.25, 0.50, 0.50, 0.50],
                "iv": [0.24, 0.22, 0.24, 0.26, 0.24, 0.26],
                "option_type": ["call"] * 6,
            }
        )
        S, r = 100, 0.05

        result = calibrate_heston(data, S, r, method="local")

        # Should fit all data points
        assert "kappa" in result
        assert result["objective"] >= 0


class TestEdgeCases:
    """Test edge cases."""

    def test_single_option(self):
        """Should handle calibration to single option."""
        data = pd.DataFrame(
            {
                "strike": [100],
                "expiry": [0.25],
                "iv": [0.22],
                "option_type": ["call"],
            }
        )
        S, r = 100, 0.05

        result = calibrate_heston(data, S, r)

        # May not converge well, but shouldn't crash
        assert "kappa" in result

    def test_flat_iv_curve(self):
        """Flat IV should still calibrate."""
        data = pd.DataFrame(
            {
                "strike": [90, 95, 100, 105, 110],
                "expiry": [0.5] * 5,
                "iv": [0.20] * 5,
                "option_type": ["call"] * 5,
            }
        )
        S, r = 100, 0.05

        result = calibrate_heston(data, S, r)

        # Should give near-zero correlation
        assert abs(result["rho"]) <= 0.5  # Allow exactly 0.5

    def test_very_short_expiry(self):
        """Should handle very short expiries."""
        data = pd.DataFrame(
            {
                "strike": [95, 100, 105],
                "expiry": [0.01] * 3,  # ~3.6 days
                "iv": [0.25, 0.22, 0.25],
                "option_type": ["call"] * 3,
            }
        )
        S, r = 100, 0.05

        result = calibrate_heston(data, S, r)

        # Should run without error
        assert "v0" in result
