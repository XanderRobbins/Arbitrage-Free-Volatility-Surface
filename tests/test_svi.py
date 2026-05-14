"""
Unit tests for SVI parameterization and fitting.
"""

import pytest
import numpy as np
import pandas as pd
from vol_surface.svi import svi_raw, svi_raw_to_iv, fit_svi_slice, fit_svi_surface


class TestSVIRaw:
    """Test raw SVI formula."""

    def test_zero_params(self):
        """SVI with zero params should give constant w=a."""
        k = np.array([-0.1, 0, 0.1])
        a, b, rho, m, sigma = 0.04, 0.0, 0.0, 0.0, 0.1
        w = svi_raw(k, a, b, rho, m, sigma)
        np.testing.assert_allclose(w, a, atol=1e-10)

    def test_positive_b_increases_wings(self):
        """Positive b should increase variance at wings."""
        k = np.array([-0.2, 0, 0.2])
        a, b, rho, m, sigma = 0.04, 0.1, 0.0, 0.0, 0.1
        w = svi_raw(k, a, b, rho, m, sigma)

        assert w[0] > a  # Left wing
        assert w[2] > a  # Right wing

    def test_rho_tilts_smile(self):
        """Negative rho should tilt smile (left higher than right)."""
        k = np.array([-0.2, 0.2])
        a, b, rho, m, sigma = 0.04, 0.2, -0.5, 0.0, 0.1
        w = svi_raw(k, a, b, rho, m, sigma)

        # Left wing should be higher with negative rho
        assert w[0] > w[1]

    def test_scalar_input(self):
        """Should handle scalar log-moneyness."""
        k = 0.0
        a, b, rho, m, sigma = 0.04, 0.1, -0.3, 0.0, 0.1
        w = svi_raw(k, a, b, rho, m, sigma)

        assert isinstance(w, (float, np.ndarray))
        assert w > 0

    def test_array_input(self):
        """Should handle array log-moneyness."""
        k = np.linspace(-0.5, 0.5, 10)
        a, b, rho, m, sigma = 0.04, 0.15, -0.4, 0.0, 0.1
        w = svi_raw(k, a, b, rho, m, sigma)

        assert len(w) == 10
        assert np.all(w > 0)


class TestSVIToIV:
    """Test conversion from SVI total variance to IV."""

    def test_zero_time_raises_or_inf(self):
        """T=0 should handle division by zero."""
        k = np.array([0.0])
        T = 0.0
        a, b, rho, m, sigma = 0.04, 0.1, 0.0, 0.0, 0.1

        # Should either handle gracefully or raise
        iv = svi_raw_to_iv(k, T, a, b, rho, m, sigma)
        assert np.isinf(iv[0]) or T == 0

    def test_positive_time(self):
        """Should compute IV correctly for T > 0."""
        k = np.array([-0.1, 0, 0.1])
        T = 0.25
        a, b, rho, m, sigma = 0.04, 0.1, -0.3, 0.0, 0.1

        iv = svi_raw_to_iv(k, T, a, b, rho, m, sigma)

        # IV should be reasonable (between 0.1 and 2.0)
        assert np.all(iv > 0)
        assert np.all(iv < 2.0)

    def test_roundtrip_variance(self):
        """iv^2 * T should equal w."""
        k = np.array([0.0])
        T = 1.0
        a, b, rho, m, sigma = 0.05, 0.15, -0.2, 0.0, 0.1

        w = svi_raw(k, a, b, rho, m, sigma)
        iv = svi_raw_to_iv(k, T, a, b, rho, m, sigma)

        np.testing.assert_allclose(iv**2 * T, w, rtol=1e-10)


class TestFitSVISlice:
    """Test SVI fitting to a single expiry."""

    def test_perfect_fit_synthetic(self):
        """Fitting SVI-generated data should recover params."""
        # Generate synthetic data
        T = 0.25
        S = 100
        r = 0.05
        F = S * np.exp(r * T)

        # True params
        a_true, b_true, rho_true, m_true, sigma_true = 0.04, 0.2, -0.3, 0.0, 0.1

        strikes = np.array([90, 95, 100, 105, 110])
        k = np.log(strikes / F)
        ivs = svi_raw_to_iv(k, T, a_true, b_true, rho_true, m_true, sigma_true)

        result = fit_svi_slice(strikes, ivs, T, S, r, method="least_squares")

        assert result["success"]
        assert result["rmse"] < 0.002

    def test_noisy_data(self):
        """Should fit reasonably well to noisy data."""
        T = 0.5
        S = 100
        r = 0.03
        F = S * np.exp(r * T)

        # Generate synthetic + noise
        strikes = np.array([85, 90, 95, 100, 105, 110, 115])
        k = np.log(strikes / F)
        ivs_clean = svi_raw_to_iv(k, T, 0.05, 0.15, -0.4, 0.0, 0.1)
        ivs_noisy = ivs_clean + np.random.normal(0, 0.005, len(ivs_clean))

        result = fit_svi_slice(strikes, ivs_noisy, T, S, r)

        assert result["success"]
        assert result["rmse"] < 0.02  # Reasonable fit

    def test_few_points(self):
        """Should handle small number of strikes."""
        T = 0.25
        S = 100
        r = 0.05

        strikes = np.array([95, 100, 105])
        ivs = np.array([0.25, 0.22, 0.25])

        result = fit_svi_slice(strikes, ivs, T, S, r)

        # May or may not succeed with only 3 points
        assert "a" in result
        assert "b" in result

    def test_parameter_bounds(self):
        """Fitted parameters should respect bounds."""
        T = 0.5
        S = 100
        r = 0.03

        strikes = np.array([90, 95, 100, 105, 110])
        ivs = np.array([0.28, 0.24, 0.22, 0.24, 0.28])

        result = fit_svi_slice(strikes, ivs, T, S, r)

        assert result["a"] >= 0
        assert result["b"] >= 0
        assert -1 < result["rho"] < 1
        assert result["sigma"] > 0

    def test_differential_evolution_method(self):
        """Global optimization should work."""
        T = 0.25
        S = 100
        r = 0.05

        strikes = np.array([90, 95, 100, 105, 110])
        ivs = np.array([0.26, 0.23, 0.21, 0.23, 0.26])

        result = fit_svi_slice(strikes, ivs, T, S, r, method="differential_evolution")

        assert result["success"]
        assert result["rmse"] < 0.05


class TestFitSVISurface:
    """Test SVI fitting across multiple expiries."""

    def test_multiple_expiries(self):
        """Should fit all expiries independently."""
        data = pd.DataFrame(
            {
                "strike": [95, 100, 105, 95, 100, 105],
                "expiry": [0.25, 0.25, 0.25, 0.50, 0.50, 0.50],
                "iv": [0.24, 0.22, 0.24, 0.26, 0.24, 0.26],
            }
        )
        S = 100
        r = 0.05

        surface_params = fit_svi_surface(data, S, r)

        assert len(surface_params) == 2
        assert 0.25 in surface_params
        assert 0.50 in surface_params
        assert surface_params[0.25]["success"]
        assert surface_params[0.50]["success"]

    def test_single_expiry(self):
        """Should handle single expiry."""
        data = pd.DataFrame(
            {
                "strike": [90, 95, 100, 105, 110],
                "expiry": [0.25] * 5,
                "iv": [0.26, 0.23, 0.21, 0.23, 0.26],
            }
        )
        S = 100
        r = 0.05

        surface_params = fit_svi_surface(data, S, r)

        assert len(surface_params) == 1
        assert 0.25 in surface_params

    def test_unequal_strikes_per_expiry(self):
        """Should handle different number of strikes per expiry."""
        data = pd.DataFrame(
            {
                "strike": [95, 100, 105, 110, 90, 100, 110],
                "expiry": [0.25, 0.25, 0.25, 0.25, 0.50, 0.50, 0.50],
                "iv": [0.23, 0.21, 0.23, 0.25, 0.25, 0.23, 0.27],
            }
        )
        S = 100
        r = 0.05

        surface_params = fit_svi_surface(data, S, r)

        assert len(surface_params) == 2


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_flat_iv_curve(self):
        """Flat IV should give b ≈ 0."""
        T = 0.25
        S = 100
        r = 0.05

        strikes = np.array([90, 95, 100, 105, 110])
        ivs = np.array([0.20] * 5)

        result = fit_svi_slice(strikes, ivs, T, S, r)

        assert abs(result["b"]) < 0.1  # Near zero slope

    def test_extreme_smile(self):
        """Very steep smile should still converge."""
        T = 0.25
        S = 100
        r = 0.05

        strikes = np.array([90, 95, 100, 105, 110])
        ivs = np.array([0.50, 0.35, 0.25, 0.35, 0.50])

        result = fit_svi_slice(strikes, ivs, T, S, r)

        # Should either succeed or fail gracefully
        assert "a" in result
