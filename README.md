# 📈 Arbitrage Free Volatility Surface

**Arbitrage-free volatility surface construction, calibration, and analysis toolkit**

A production-grade Python library for computing implied volatilities, enforcing no-arbitrage constraints, fitting parametric models (SVI), and calibrating stochastic volatility models (Heston) to market data.

---

## 🚀 Features

- **Robust IV Computation**: Newton-Raphson + Brent's method fallback for deep ITM/OTM options
- **Static Arbitrage Checks**: Put-call parity, butterfly spreads, calendar arbitrage
- **SVI Parameterization**: Fit smooth, arbitrage-free volatility smiles across expiries
- **Heston Calibration**: Fast calibration using COS method (Fang & Oosterlee 2008)
- **Visualization**: 3D surface plots, smile comparisons, model diagnostics
- **Clean API**: Modular design with comprehensive tests and documentation

---

## 📊 Example Visualizations

### Volatility Smile
![Volatility Smile](images/Volitility_Smile.png)

### 3D Surface
![3D Surface](images/Volitility_Surface.png)

---

## 📦 Installation

### From Source (Development)

```bash
git clone https://github.com/XanderRobbins/Arbitrage-Free-Volitility-Surface.git
cd volatility-surface-lab
pip install -e .

**⚠️ Python Version:** This project requires **Python 3.11** for maximum stability. Python 3.13+ has known compatibility issues with NumPy/SciPy.
```

### With Optional Dependencies

```bash
# Development tools (pytest, jupyter, black)
pip install -e ".[dev]"

# Fast computation (numba JIT)
pip install -e ".[fast]"
```

---

## 🎯 Quick Start

### Minimal Example

```python
import pandas as pd
from vol_surface import VolatilitySurface

# Your market data (sample format)
data = pd.DataFrame({
    'strike': [95, 100, 105, 110],
    'expiry': [0.25, 0.25, 0.25, 0.25],
    'option_type': ['call', 'call', 'call', 'call'],
    'price': [8.5, 5.2, 2.8, 1.1]
})

# Initialize surface
surface = VolatilitySurface(S=100, r=0.02)

# Full pipeline
surface.load_data(data) \
       .compute_ivs() \
       .check_arbitrage() \
       .fit_svi() \
       .calibrate_heston()

# Visualize
surface.plot_smile(expiry=0.25)
surface.plot_surface_3d(model='heston')
surface.summary()
```

---

## 📊 Complete Workflow

### 1. Load Market Data

```python
# From CSV
import pandas as pd
data = pd.read_csv('data/spy_options.csv')

# Required columns: strike, expiry, option_type, price
surface = VolatilitySurface(S=450.0, r=0.03)
surface.load_data(data)
```

### 2. Compute Implied Volatilities

```python
# Computes IVs using Newton-Raphson with Brent fallback
surface.compute_ivs()

# Access computed IVs
print(surface.market_data[['strike', 'expiry', 'iv']])
```

### 3. Check for Arbitrage

```python
# Runs put-call parity, butterfly, and calendar checks
surface.check_arbitrage(tol=1e-3)

# View violations
violations = surface.arbitrage_violations
print(f"Butterfly violations: {len(violations['butterfly'])}")
```

### 4. Fit SVI Parameterization

```python
# Fits SVI to each expiry slice
surface.fit_svi(method='least_squares')

# View parameters
for T, params in surface.svi_params.items():
    print(f"T={T:.3f}: a={params['a']:.4f}, b={params['b']:.4f}, ρ={params['rho']:.4f}")
```

### 5. Calibrate Heston Model

```python
# Global calibration across all expiries
surface.calibrate_heston(method='local')

# View parameters
print(surface.heston_params)
# Output: {'kappa': 2.15, 'theta': 0.042, 'xi': 0.31, 'rho': -0.58, 'v0': 0.039}
```

### 6. Compare Models

```python
# Compare SVI vs Heston fit quality
df = surface.compare_models(metric='rmse')

#   expiry  n_options  SVI_rmse  Heston_rmse
# 0   0.25         20    0.0023       0.0041
# 1   0.50         18    0.0019       0.0036
```

### 7. Visualize

```python
# Single expiry smile
surface.plot_smile(expiry=0.25, include_svi=True, include_heston=True)

# 3D surface plot
surface.plot_surface_3d(model='market')  # or 'svi' or 'heston'

# Summary statistics
surface.summary()
```

---

## 🏗️ Project Structure

```
volatility-surface-lab/
│
├── vol_surface/              # Main package
│   ├── __init__.py
│   ├── iv_solver.py          # IV computation (Newton + Brent)
│   ├── arbitrage.py          # No-arbitrage checks
│   ├── svi.py                # SVI parameterization
│   ├── heston.py             # Heston model + COS pricing
│   ├── pricing.py            # Black-Scholes, COS, FFT
│   └── surface.py            # Main VolatilitySurface class
│
├── tests/                    # Unit tests (pytest)
│   ├── test_iv_solver.py
│   ├── test_arbitrage.py
│   ├── test_svi.py
│   ├── test_heston.py
│   └── test_pricing.py
│
├── data/                     # Sample data
│   └── spy_options.csv
│
├── requirements.txt          # Dependencies
├── setup.py                  # Package setup
└── README.md                 # This file
```

---

## 🧪 Testing

```bash
# Run all tests
pytest

# With coverage report
pytest --cov=vol_surface --cov-report=html

# Run specific test file
pytest tests/test_iv_solver.py -v
```

---

## 📚 Mathematical Background

### Implied Volatility

Given market option price C_market, solve for σ:

```
C_BS(S, K, T, r, σ) = C_market
```

Using Newton-Raphson with vega as the derivative.

### SVI Parameterization

Total implied variance as a function of log-moneyness k = log(K/F):

```
w(k) = a + b(ρ(k - m) + √((k - m)² + σ²))
```

Parameters: a, b, ρ, m, σ

### Heston Model

Asset price and variance dynamics:

```
dS_t = rS_t dt + √(v_t) S_t dW¹_t
dv_t = κ(θ - v_t)dt + ξ√(v_t) dW²_t
dW¹_t · dW²_t = ρ dt
```

Parameters: κ (mean reversion), θ (long-term var), ξ (vol-of-vol), ρ (correlation), v₀ (initial var)

Pricing: COS method (Fourier-cosine series expansion)

---

## 🔬 Example Use Cases

### 1. Market Making Desk
- Detect mispriced options via arbitrage checks
- Price exotic derivatives using calibrated Heston model
- Monitor volatility surface evolution in real-time

### 2. Quantitative Research
- Compare parametric models (SVI vs Heston vs SABR)
- Analyze term structure of volatility
- Study volatility skew dynamics

### 3. Risk Management
- Compute Greeks using calibrated models
- Stress-test portfolios under extreme vol scenarios
- Validate pricing models against market data

---

## 🛠️ Advanced Usage

### Custom Initial Guess for Heston

```python
initial_guess = {
    'kappa': 1.5,
    'theta': 0.05,
    'xi': 0.4,
    'rho': -0.7,
    'v0': 0.04
}

surface.calibrate_heston(method='global', initial_guess=initial_guess)
```

### Global Optimization (Slower but More Robust)

```python
# Uses scipy's differential_evolution
surface.fit_svi(method='differential_evolution')
surface.calibrate_heston(method='global')
```

### Access Underlying Models

```python
# SVI evaluation at any log-moneyness
from vol_surface.svi import svi_raw_to_iv

k = np.linspace(-0.3, 0.3, 50)  # log-moneyness
iv = svi_raw_to_iv(k, T=0.25, **surface.svi_params[0.25])

# Heston pricing for custom strikes
model = surface.heston_params['model']
price = model.price_call_cos(S=100, K=105, T=0.5, r=0.02, N=128)
```

---

## 📖 References

- Gatheral & Jacquier (2014): "Arbitrage-free SVI volatility surfaces"
- Fang & Oosterlee (2008): "A Novel Pricing Method for European Options Based on Fourier-Cosine Series Expansions"
- Heston (1993): "A Closed-Form Solution for Options with Stochastic Volatility"
- Carr & Madan (1999): "Option Valuation Using the Fast Fourier Transform"

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Add tests for new functionality
4. Ensure `pytest` and `flake8` pass
5. Submit a pull request

---

## 📄 License

MIT License - see LICENSE file for details.

---

## 👤 Author

**Alexander Robbins**  
University of Florida | Math, CS, Economics  
📧 robbins.a@ufl.edu  
🔗 [GitHub](https://github.com/XanderRobbins) | [LinkedIn](https://www.linkedin.com/in/alexander-robbins-a1086a248/) | [Website](https://xanderrobbins.github.io/)

---

## 🙏 Acknowledgments

- Inspired by Jim Gatheral's work on volatility surfaces
- COS method implementation based on Fang & Oosterlee (2008)

---

## 📈 Performance

- **IV computation**: ~1ms per option (Newton-Raphson)
- **SVI fitting**: ~50ms per expiry slice (L-BFGS-B)
- **Heston calibration**: ~5-10s for 100 options (local), ~60s (global)
- **COS pricing**: ~0.5ms per option (N=128 terms)

*Benchmarked on Windows, Python 3.11*

---

## 🐛 Known Issues

- Very deep ITM/OTM options may fail IV convergence (returns NaN)
- Heston calibration sensitive to initial guess (use `method='global'` if stuck)
- SVI fitting may violate Gatheral no-arb conditions for extreme parameters (future work)

---

## 🗺️ Roadmap

- [ ] Add SABR model calibration
- [ ] Implement local volatility surface (Dupire)
- [ ] Add time-dependent Heston parameters
- [ ] GPU acceleration for batch pricing (CuPy)
- [ ] Real-time data integration (CBOE, Interactive Brokers)
- [ ] Greeks computation via adjoint differentiation
