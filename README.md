# TwinTrade

Monte-Carlo simulation of portfolio rebalancing strategies for correlated asset pairs.

TwinTrade simulates the daily price trajectories of two stocks with configurable growth rates, volatilities, and correlations. At the end of each day, if the allocation divergence exceeds a specified threshold, the portfolio is automatically rebalanced net of transaction costs, and compared against a passive Buy & Hold benchmark across $N$ simulations.

---

## Features

- **Correlated Asset Generator**: Simulates correlated daily price movements using Geometric Brownian Motion (GBM) with Cholesky matrix decomposition.
- **Exact Rebalancing Math**: Solves analytically for the exact dollar amount to trade to achieve equal position parity (50/50) after accounting for both fixed ($) and percentage (%) transaction fees.
- **Monte-Carlo Simulation**: Runs hundreds or thousands of randomized market paths to compute empirical distributions.
- **Benchmark Comparison**: Measures win rate, CAGR, median/mean returns, percentiles, and drawdown metrics relative to a passive Buy & Hold strategy.
- **Rich Terminal UI & Visualizations**: Outputs formatted comparison tables in terminal and optionally exports matplotlib charts and JSON reports.

---

## Installation

### Prerequisites
- Python 3.9 or higher

### Install locally
```bash
git clone https://github.com/sebastiengilbert73/twintrade.git
cd twintrade

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies and package in editable mode
pip install -e .
```

---

## Usage

### 1. Command Line Interface (CLI)

Run a basic simulation with default parameters (1,000 runs of 252 trading days):
```bash
twintrade
```

#### Simulating Correlated Assets
```bash
# Negatively correlated assets (rho = -0.8): assets tend to move in opposite directions
twintrade --correlation -0.8 --num-sims 1000

# Positively correlated assets (rho = 0.8): assets move together
twintrade --correlation 0.8 --num-sims 1000
```

#### Customizing Stock Properties & Market Fees
```bash
twintrade \
  --stock-a-name "Tech Stock" --growth-a 0.12 --vol-a 0.025 \
  --stock-b-name "Bond Index" --growth-b 0.04 --vol-b 0.008 \
  --fee-percent 0.001 --fee-fixed 2.0 \
  --threshold 0.05
```

#### Saving Visual Charts and JSON Reports
```bash
twintrade \
  --num-sims 500 \
  --correlation -0.5 \
  --plot-output simulation_chart.png \
  --json-output simulation_report.json
```

---

### 2. Command Line Options

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--correlation` | Correlation coefficient between daily returns ($-1.0$ to $1.0$) | `0.0` |
| `--threshold` | Rebalancing trigger threshold (relative ratio or absolute $) | `0.05` (5%) |
| `--threshold-type` | Threshold calculation mode (`relative` or `absolute`) | `relative` |
| `--fee-percent` | Transaction fee percentage on traded value (e.g. `0.001` = 0.1%) | `0.001` |
| `--fee-fixed` | Fixed dollar fee per transaction (e.g. `$2.00`) | `0.0` |
| `--stock-a-name` | Display name for Stock A | `"Stock A"` |
| `--growth-a` | Annual growth rate for Stock A (e.g. `0.08` = 8%) | `0.08` |
| `--vol-a` | Daily volatility (standard deviation) for Stock A | `0.015` |
| `--price-a` | Initial price per share for Stock A | `100.0` |
| `--stock-b-name` | Display name for Stock B | `"Stock B"` |
| `--growth-b` | Annual growth rate for Stock B | `0.08` |
| `--vol-b` | Daily volatility for Stock B | `0.015` |
| `--price-b` | Initial price per share for Stock B | `100.0` |
| `--initial-value` | Starting portfolio capital (split 50/50) | `1000.0` |
| `--num-days` | Number of trading days per simulation | `252` |
| `--num-sims` | Number of Monte-Carlo simulation runs | `1000` |
| `--seed` | Random seed for exact reproducibility | `None` |
| `--plot-output` | File path to save generated Matplotlib chart | `None` |
| `--json-output` | File path to export summary JSON report | `None` |

---

### 3. Python API Usage

You can also import `twintrade` as a Python library inside custom scripts or Jupyter notebooks:

```python
from twintrade.models import SimulationConfig, StockPairConfig, StockConfig, MarketConfig
from twintrade.engine import run_batch_simulation
from twintrade.stats import compute_summary_stats
from twintrade.visualization import plot_simulation_results

# Define simulation parameters
config = SimulationConfig(
    pair=StockPairConfig(
        stock_a=StockConfig(name="Asset A", annual_growth_rate=0.10, daily_volatility=0.02),
        stock_b=StockConfig(name="Asset B", annual_growth_rate=0.06, daily_volatility=0.01),
        correlation=-0.4,
    ),
    market=MarketConfig(fee_percent=0.001, fee_fixed=1.0),
    threshold=0.05,
    num_days=252,
    num_simulations=500,
    seed=42,
)

# Run Monte-Carlo simulations
results = run_batch_simulation(config)

# Compute performance statistics
stats = compute_summary_stats(results)

print(f"Win Rate vs Buy & Hold: {stats.win_rate_percent:.1f}%")
print(f"Mean Rebalanced Value: ${stats.rebalanced_stats.mean:,.2f}")
print(f"Mean Buy & Hold Value: ${stats.buy_and_hold_stats.mean:,.2f}")

# Generate and save chart
plot_simulation_results(results, save_path="twintrade_summary.png")
```

---

## Model Specifications

### Price Dynamics
Each stock price $P_{i, t}$ evolves according to Geometric Brownian Motion:
$$P_{i, t} = P_{i, t-1} \cdot \exp\left( \left(\mu_i - \frac{1}{2}\sigma_i^2\right) + \sigma_i Z_{i, t} \right)$$
where $Z_A \sim \mathcal{N}(0, 1)$ and $Z_B = \rho Z_A + \sqrt{1 - \rho^2} Z_{\text{indep}}$.

### Rebalancing Rule & Cost Equation
When the value divergence $|V_A - V_B| / V_{\text{total}} > \text{threshold}$, the higher-value position $V_{\text{high}}$ is sold to buy $V_{\text{low}}$. Selling $X$ of stock A and buying with the net proceeds of stock B is treated as **two transactions**, incurring fixed costs $c_{\$}$ and percentage costs $c_{\%}$. The required trade amount $X$ to achieve exact equality $V_A' = V_B'$ post-fees is:
$$X = \frac{V_{\text{high}} - V_{\text{low}} + (2 - c_{\%}) c_{\$}}{2 - 2c_{\%} + c_{\%}^2}$$

---

## Running Tests

Execute unit tests using `pytest`:
```bash
pytest
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
