"""
Data models and configuration dataclasses for TwinTrade.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional
import numpy as np


class ThresholdType(str, Enum):
    RELATIVE = "relative"  # Ratio: |Va - Vb| / (Va + Vb) > threshold
    ABSOLUTE = "absolute"  # Dollar amount: |Va - Vb| > threshold


@dataclass
class StockConfig:
    """Properties for a single stock asset."""
    name: str = "Stock"
    annual_growth_rate: float = 0.08  # 8% annual growth rate
    daily_volatility: float = 0.015   # 1.5% daily standard deviation of return
    initial_price: float = 100.0      # Starting share price in $


@dataclass
class StockPairConfig:
    """Properties for the pair of stocks."""
    stock_a: StockConfig = field(default_factory=lambda: StockConfig(name="Stock A"))
    stock_b: StockConfig = field(default_factory=lambda: StockConfig(name="Stock B"))
    correlation: float = 0.0  # -1.0 to 1.0 correlation between daily returns


@dataclass
class MarketConfig:
    """Properties of the market (transaction costs)."""
    fee_percent: float = 0.001  # 0.1% transaction fee on traded value
    fee_fixed: float = 0.0      # Fixed dollar cost per transaction ($)


@dataclass
class SimulationConfig:
    """Complete parameters for a trading simulation run."""
    pair: StockPairConfig = field(default_factory=StockPairConfig)
    market: MarketConfig = field(default_factory=MarketConfig)
    initial_portfolio_value: float = 1000.0  # Equal allocation: 50% A, 50% B
    threshold: float = 0.05  # Rebalancing threshold (e.g. 0.05 = 5% relative diff)
    threshold_type: ThresholdType = ThresholdType.RELATIVE
    num_days: int = 252       # Duration of simulation in days
    num_simulations: int = 1000  # Number of simulation runs
    trading_days_per_year: int = 252
    seed: Optional[int] = None


@dataclass
class SingleSimulationResult:
    """Detailed time-series results for one simulation run."""
    portfolio_values: np.ndarray        # Array of total portfolio value over time (num_days + 1)
    buy_and_hold_values: np.ndarray     # Array of un-rebalanced portfolio value (num_days + 1)
    stock_a_prices: np.ndarray          # Price path for Stock A
    stock_b_prices: np.ndarray          # Price path for Stock B
    stock_a_shares: np.ndarray          # Shares of Stock A over time
    stock_b_shares: np.ndarray          # Shares of Stock B over time
    rebalance_days: List[int]           # Indices of days when rebalancing occurred
    total_fees_paid: float              # Total dollar fees paid over the run
    final_portfolio_value: float        # Final value of rebalanced portfolio
    final_buy_and_hold_value: float     # Final value of buy-and-hold portfolio


@dataclass
class BatchSimulationResult:
    """Aggregated statistics across multiple simulation runs."""
    config: SimulationConfig
    final_portfolio_values: np.ndarray   # 1D array of final values for rebalanced portfolios
    final_buy_and_hold_values: np.ndarray  # 1D array of final values for buy-and-hold portfolios
    rebalance_counts: np.ndarray         # 1D array of rebalance counts per run
    total_fees_paid: np.ndarray          # 1D array of total fees paid per run
    empirical_correlations: np.ndarray   # 1D array of realized correlation coefficients per run
    sample_runs: List[SingleSimulationResult] = field(default_factory=list)
