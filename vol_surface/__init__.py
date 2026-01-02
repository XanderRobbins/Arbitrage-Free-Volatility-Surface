"""
Arbitrage-free volatility surface construction and calibration toolkit.

Author: Alexander Robbins
"""

from .iv_solver import implied_vol_call, implied_vol_put, implied_vol
from .arbitrage import (
    check_put_call_parity,
    check_butterfly_arbitrage,
    check_calendar_arbitrage,
    check_all_arbitrage,
)
from .svi import svi_raw, fit_svi_slice, fit_svi_surface
from .heston import HestonModel, calibrate_heston
from .pricing import black_scholes_call, black_scholes_put, cos_pricer
from .surface import VolatilitySurface

__version__ = "0.1.0"

__all__ = [
    "implied_vol_call",
    "implied_vol_put",
    "implied_vol",
    "check_put_call_parity",
    "check_butterfly_arbitrage",
    "check_calendar_arbitrage",
    "check_all_arbitrage",
    "svi_raw",
    "fit_svi_slice",
    "fit_svi_surface",
    "HestonModel",
    "calibrate_heston",
    "black_scholes_call",
    "black_scholes_put",
    "cos_pricer",
    "VolatilitySurface",
]