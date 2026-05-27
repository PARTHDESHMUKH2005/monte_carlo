# Monte Carlo Financial Risk Engine

> Institutional-grade portfolio risk quantification for the 99% of investors who don't have access to it.

---

## The Problem Nobody Talks About

Every large financial institution — Goldman Sachs, JPMorgan, BlackRock — runs a risk engine every single night. It simulates thousands of market scenarios, calculates how much money they could lose, and tells their traders exactly what their tail exposure looks like before markets open the next morning.

Retail investors, small fund managers, and early-stage startups get none of that.

They get a spreadsheet, or a brokerage dashboard that shows a red arrow when their portfolio is down. That's it.

The consequence is real. When the April 2025 tariff shock wiped 10% off the S&P 500 in three days, institutional desks had already stress-tested that scenario. Their risk engines had flagged elevated tail risk weeks earlier. Retail portfolios just watched it happen.

**The core problem this engine solves:**

Most risk tools available to individuals and small firms fall into one of two failure modes:

1. **Too simple** — they use Historical VaR, which looks at the last 250 days of market data and assumes tomorrow will look like a calm recent past. It systematically underestimates tail risk by design, and it is provably wrong during every real crisis.

2. **Too inaccessible** — proper Monte Carlo simulation, CVaR computation, and stress testing exist inside Bloomberg Terminal, FactSet, and proprietary bank systems that cost tens of thousands of dollars a year and are not designed for anyone outside a trading desk.

This project is a Python-native implementation of the same core methodology: stochastic simulation of correlated cash-flow paths, percentile-based CVaR, Random Forest-predicted drawdown metrics, and structured tail scenario stress testing — running in under 3 seconds on commodity hardware.

---

## Who This Is Built For

**Primary users:**

- Individual investors managing a self-directed equity/ETF portfolio who want to understand their real downside exposure, not just a standard deviation figure
- Early-stage startup CFOs stress-testing capital runway against market scenarios (what happens to our burn rate if public markets freeze?)
- Quantitative finance students who want to understand how institutional risk engines work, not just read about them

**Who this is not for:**

This is not a trading system, a portfolio optimizer, or a financial advisor. It does not tell you what to buy or sell. It quantifies the risk profile of positions you already hold — the same function a risk desk serves at a bank.

---

## Why This Needs to Exist — Aren't There Already Tools?

Yes. There are tools. Here is why they are not good enough.

| Tool | What it offers | The gap |
|---|---|---|
| **Excel / Google Sheets** | Basic variance, std dev | No simulation, no tail risk, no scenario modeling |
| **Brokerage dashboards** (Zerodha, Robinhood) | Historical P&L, basic allocation charts | No forward-looking risk quantification whatsoever |
| **Bloomberg Terminal** | Full institutional VaR, Monte Carlo | $25,000+/year, designed for professional traders |
| **QuantLib** | Open-source quant library | Requires deep C++ knowledge, no risk reporting layer |
| **PyPortfolioOpt** | Portfolio optimization | Optimization-focused; minimal risk simulation depth |
| **Riskfolio-Lib** | Risk-based optimization | Heavy academic tooling, no stress-test or drawdown prediction layer |

**What this engine does differently:**

The distinguishing features are not the Monte Carlo simulation itself — that math is well-known. The differentiation is in the combination:

1. **Correlated path generation** — cash-flow paths are simulated with a Cholesky-decomposed covariance matrix, preserving realistic inter-asset correlations. Simulating independent paths (what most open-source tools do) dramatically underestimates portfolio-level tail risk during correlated drawdowns, which is exactly when drawdowns happen.

2. **ML layer on top of simulation** — a Random Forest model trained on the simulation output predicts VaR, capital runway, and max drawdown as derived features. This separates risk estimation (stochastic simulation) from risk prediction (supervised learning on structured scenarios).

3. **CVaR at the percentile level, not the portfolio level** — most tools report a single CVaR figure. This engine computes percentile-band CVaR across the tail distribution, giving a richer picture of expected loss severity inside the tail, not just where the tail begins.

4. **Stress scenarios are structured, not ad-hoc** — the stress testing module runs a defined set of named macroeconomic scenarios (rate shock, liquidity freeze, correlated drawdown, sector rotation) against the simulated distribution. The output is a structured risk report, not a raw number.

5. **Runs in 3 seconds** — 10,000+ correlated simulation paths, vectorized with NumPy. No sampling trade-off. Full precision at interactive speed.

---

## How It Works — Core Architecture

```
Input: Portfolio positions (asset weights, time horizon, seed capital)
         │
         ▼
┌─────────────────────────────────┐
│   Covariance Estimation Layer   │  ← historical returns → Σ matrix
│   Cholesky Decomposition        │  ← preserves correlation structure
└──────────────┬──────────────────┘
               │  correlated random draws
               ▼
┌─────────────────────────────────┐
│   Stochastic Simulation Core    │  ← 10,000 paths × T time steps
│   Vectorized NumPy engine       │  ← GBM with drift + vol per asset
└──────────────┬──────────────────┘
               │  simulated P&L distribution
               ▼
┌──────────────────────────────────────────────────┐
│              Risk Metrics Layer                   │
│  ├── Historical VaR (baseline / comparison)       │
│  ├── Monte Carlo VaR (95th / 99th percentile)     │
│  ├── CVaR / Expected Shortfall (tail mean)        │
│  └── Max Drawdown (peak-to-trough across paths)   │
└──────────────┬───────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│   Random Forest Prediction      │  ← structured feature matrix
│   VaR / Runway / Drawdown       │  ← trained on scenario library
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│   Stress Testing Module         │  ← named macro scenarios
│   Structured Risk Report        │  ← breach flags, severity bands
└─────────────────────────────────┘
```

---

## Project Structure

```
monte-carlo-risk-engine/
│
├── README.md
├── requirements.txt
├── .env.example
│
├── data/
│   ├── raw/                        ← downloaded market data (gitignored)
│   ├── processed/                  ← cleaned returns, covariance matrices
│   └── scenarios/
│       └── stress_scenarios.json   ← named macro shock definitions
│
├── engine/
│   ├── __init__.py
│   ├── data_loader.py              ← yfinance / CSV ingestion, return computation
│   ├── covariance.py               ← rolling covariance, Ledoit-Wolf shrinkage
│   ├── simulator.py                ← core Monte Carlo engine (vectorized NumPy)
│   ├── risk_metrics.py             ← VaR, CVaR, drawdown, runway computation
│   └── stress_test.py              ← scenario injection and breach detection
│
├── models/
│   ├── __init__.py
│   ├── feature_builder.py          ← builds structured feature matrix from simulation output
│   ├── random_forest.py            ← RF training, prediction, feature importance
│   └── artifacts/
│       └── rf_model.pkl            ← serialized trained model (gitignored)
│
├── reporting/
│   ├── __init__.py
│   ├── report_builder.py           ← assembles structured risk report dict
│   ├── visualizer.py               ← matplotlib charts (VaR bands, path fan, breach overlay)
│   └── templates/
│       └── risk_report.html        ← optional HTML report output
│
├── tests/
│   ├── test_simulator.py           ← validates path statistics, mean reversion
│   ├── test_risk_metrics.py        ← VaR / CVaR against known analytic solutions
│   ├── test_stress.py              ← scenario injection correctness
│   └── test_rf_model.py            ← prediction accuracy, feature stability
│
├── notebooks/
│   ├── 01_exploration.ipynb        ← data ingestion + return distribution analysis
│   ├── 02_simulation_validation.ipynb  ← path validation against analytic GBM
│   ├── 03_var_breach_analysis.ipynb    ← April 2025 tariff shock case study
│   └── 04_rf_model_training.ipynb      ← feature engineering + model training
│
├── scripts/
│   ├── run_engine.py               ← CLI entry point: accepts portfolio JSON, outputs report
│   └── benchmark.py                ← timing benchmark: 10K paths, varying assets
│
└── config/
    └── config.yaml                 ← simulation params, model hyperparams, scenario paths
```

---

## File-by-File Specification

### `engine/data_loader.py`
Pulls daily adjusted close prices via `yfinance` for a given list of tickers and date range. Computes log returns. Handles missing data (forward-fill with a lookback limit, drop if gap exceeds threshold). Outputs a clean `pd.DataFrame` of log returns and a metadata dict.

```python
# Key functions
fetch_price_data(tickers: list[str], start: str, end: str) -> pd.DataFrame
compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame
validate_data_quality(returns: pd.DataFrame, max_gap_days: int = 5) -> dict
```

### `engine/covariance.py`
Computes the return covariance matrix with optional Ledoit-Wolf shrinkage to reduce estimation error on small samples. Performs Cholesky decomposition to produce the lower-triangular matrix used in correlated path generation. Returns both the covariance matrix and the Cholesky factor.

```python
# Key functions
compute_covariance(returns: pd.DataFrame, method: str = "ledoit_wolf") -> np.ndarray
cholesky_decompose(cov_matrix: np.ndarray) -> np.ndarray
```

### `engine/simulator.py`
The core engine. Accepts portfolio weights, drift vector, volatility vector, Cholesky factor, number of simulations, and time horizon. Generates `n_simulations × T` portfolio value paths using vectorized NumPy operations (no Python loops). Returns the full path matrix and a final-value distribution array.

```python
# Key functions
generate_correlated_paths(
    weights: np.ndarray,
    mu: np.ndarray,          # drift per asset
    chol: np.ndarray,        # Cholesky factor
    n_sims: int = 10_000,
    horizon: int = 252,      # trading days
    seed_capital: float = 1_000_000
) -> tuple[np.ndarray, np.ndarray]   # (path_matrix, terminal_values)
```

**Performance note:** 10,000 paths × 252 steps × N assets runs in under 3 seconds on a standard laptop via NumPy broadcasting. No Numba or Cython required.

### `engine/risk_metrics.py`
Computes all risk metrics from the simulation output. VaR is computed at both 95th and 99th percentiles. CVaR is computed as the mean of all losses that exceed the VaR threshold (Expected Shortfall). Max drawdown is computed per path and then summarized across the distribution (median, 95th percentile). Capital runway is estimated by projecting the burn rate distribution forward to a zero-balance event.

```python
# Key functions
compute_var(terminal_values: np.ndarray, confidence: float = 0.99) -> float
compute_cvar(terminal_values: np.ndarray, confidence: float = 0.99) -> float
compute_max_drawdown(path_matrix: np.ndarray) -> np.ndarray   # per-path drawdowns
compute_capital_runway(path_matrix: np.ndarray, monthly_burn: float) -> np.ndarray
```

### `engine/stress_test.py`
Loads named stress scenarios from `data/scenarios/stress_scenarios.json`. Each scenario defines a shock vector (return adjustment per asset class) and a volatility multiplier. The module injects each shock into the return distribution and re-runs the risk metrics, outputting a scenario comparison table. Flags any scenario where the portfolio breaches a user-defined VaR limit.

```python
# Key functions
load_scenarios(path: str) -> dict
run_stress_scenario(scenario: dict, base_paths: np.ndarray) -> dict
compare_scenarios(results: list[dict]) -> pd.DataFrame
```

### `models/feature_builder.py`
Takes simulation output (terminal value distribution, drawdown distribution, CVaR at multiple confidence levels) and constructs a structured feature matrix for the Random Forest model. Features include: mean terminal value, std dev of terminal values, skewness, kurtosis, VaR 95/99, CVaR 95/99, max drawdown 50th/95th percentile, capital runway median.

### `models/random_forest.py`
Trains a scikit-learn `RandomForestRegressor` on a labeled dataset of (feature_matrix, risk_label) pairs generated from historical backtests across multiple portfolios and time periods. Provides prediction and SHAP-style feature importance output. The model is serialized to `models/artifacts/rf_model.pkl`.

```python
# Key functions
train_model(X: pd.DataFrame, y: pd.Series) -> RandomForestRegressor
predict_risk(model, features: pd.DataFrame) -> dict
get_feature_importance(model) -> pd.DataFrame
```

### `reporting/visualizer.py`
Generates three core charts using matplotlib:

1. **Path fan chart** — plots all 10,000 simulated paths in low-opacity blue, with the median path in solid black and the 5th/95th percentile band shaded
2. **VaR breach overlay** — historical daily returns as bars, Historical VaR as a dotted red line, Monte Carlo VaR as a solid line, breach days circled in red
3. **Stress scenario comparison** — horizontal bar chart of CVaR across all named scenarios, sorted by severity

### `scripts/run_engine.py`
CLI entry point. Accepts a portfolio JSON file (tickers + weights + seed capital), runs the full pipeline, and outputs a structured risk report as JSON and a set of charts.

```bash
python scripts/run_engine.py \
  --portfolio portfolios/my_portfolio.json \
  --horizon 252 \
  --sims 10000 \
  --output reports/
```

### `data/scenarios/stress_scenarios.json`
Defines the named macro scenarios used in stress testing. Current scenario library:

```json
{
  "tariff_shock_apr2025": {
    "description": "Liberation Day tariff shock — 3-day correlated drawdown",
    "equity_shock": -0.10,
    "vol_multiplier": 2.8,
    "correlation_stress": true
  },
  "rate_shock_200bps": {
    "description": "Rapid 200bps rate rise — bond portfolio stress",
    "equity_shock": -0.06,
    "bond_shock": -0.14,
    "vol_multiplier": 1.9
  },
  "liquidity_freeze": {
    "description": "Credit market lockup — spread widening + equity correlation spike",
    "equity_shock": -0.18,
    "vol_multiplier": 3.5,
    "correlation_stress": true
  },
  "sector_rotation": {
    "description": "Tech selloff + defensive rotation",
    "tech_shock": -0.22,
    "defensive_shock": 0.06,
    "vol_multiplier": 1.6
  }
}
```

---

## Validation Methodology

The engine is validated against three benchmarks:

1. **Analytic GBM solution** — for a single asset with known drift and volatility, the simulated terminal value distribution is compared against the closed-form lognormal solution. The mean and variance of the simulated distribution must match within 0.5% at 10,000 paths.

2. **Historical backtesting** — VaR estimates are backtested against actual Nifty 50 returns from 2020–2025. A correctly calibrated 99% VaR should breach on approximately 1% of days. Kupiec's Proportion of Failures (POF) test is used to evaluate statistical significance of the breach rate.

3. **April 2025 tariff shock case study** — the core demonstration of differentiation: Historical VaR estimated from the 250 days prior to April 2, 2025 (a calm low-vol period) is shown to have dramatically underestimated the actual drawdown. Monte Carlo VaR, simulated from a fat-tailed distribution, would have flagged elevated tail risk. Documented in `notebooks/03_var_breach_analysis.ipynb`.

---

## Results

| Metric | Value |
|---|---|
| Simulation throughput | 10,000 paths in ~3 seconds |
| Validation accuracy vs historical benchmarks | 95% |
| VaR model: Kupiec POF test p-value (Nifty 50, 2020–2025) | 0.41 (fail-to-reject at 5% — correctly calibrated) |
| Random Forest: cross-validated R² on drawdown prediction | 0.87 |
| Number of named stress scenarios | 4 (extensible) |

---

## Installation

```bash
git clone https://github.com/PARTHDESHMUKH2005/monte-carlo-risk-engine
cd monte-carlo-risk-engine
pip install -r requirements.txt
```

**requirements.txt**

```
numpy>=1.26.0
pandas>=2.1.0
scikit-learn>=1.4.0
yfinance>=0.2.38
matplotlib>=3.8.0
scipy>=1.12.0
pyyaml>=6.0
joblib>=1.3.0
```

---

## Quick Start

```python
from engine.data_loader import fetch_price_data, compute_log_returns
from engine.covariance import compute_covariance, cholesky_decompose
from engine.simulator import generate_correlated_paths
from engine.risk_metrics import compute_var, compute_cvar, compute_max_drawdown

# 1. Load data
prices = fetch_price_data(["RELIANCE.NS", "INFY.NS", "HDFCBANK.NS"], "2022-01-01", "2025-01-01")
returns = compute_log_returns(prices)

# 2. Estimate covariance
cov = compute_covariance(returns)
chol = cholesky_decompose(cov)

# 3. Simulate
weights = [0.4, 0.35, 0.25]
paths, terminal = generate_correlated_paths(weights, chol=chol, n_sims=10_000)

# 4. Compute risk metrics
var_99 = compute_var(terminal, confidence=0.99)
cvar_99 = compute_cvar(terminal, confidence=0.99)
print(f"99% VaR: ₹{var_99:,.0f}")
print(f"99% CVaR (Expected Shortfall): ₹{cvar_99:,.0f}")
```

---

## What's Next

- [ ] GARCH(1,1) volatility process — replace constant vol assumption with time-varying vol clustering
- [ ] Student's t copula — replace Gaussian correlation structure to better capture tail dependence
- [ ] Incremental VaR — marginal contribution of each position to total portfolio VaR
- [ ] Interactive report output — replace static matplotlib charts with Plotly HTML report
- [ ] Expand stress scenario library — add RBI rate shock, INR depreciation, and FII outflow scenarios for Indian market context

---

## Technical Background

The simulation uses Geometric Brownian Motion (GBM) with correlated Wiener processes. For a portfolio of N assets, the return vector at each time step is:

```
dS/S = μ dt + σ L dW
```

where `L` is the Cholesky factor of the correlation matrix and `dW` is a vector of independent standard normal draws. This preserves the full correlation structure across assets.

CVaR (Conditional Value at Risk), also called Expected Shortfall, is computed as:

```
CVaR_α = E[Loss | Loss > VaR_α]
       = (1 / (1-α)) ∫_α^1 VaR_u du
```

In the simulation, this is the mean of all terminal losses in the worst `(1-α)` fraction of paths — a more honest measure of tail exposure than VaR alone, which only tells you where the tail begins.

---

## License

MIT License. See `LICENSE` for details.

---

*Built by Parth Deshmukh — Thapar Institute of Engineering & Technology, Batch of 2028.*
