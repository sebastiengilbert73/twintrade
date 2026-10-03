"""
Unit tests for single and batch simulation engine.
"""

import pytest
import numpy as np
from twintrade.models import (
    SimulationConfig,
    StockPairConfig,
    StockConfig,
    MarketConfig,
    ThresholdType,
)
from twintrade.engine import (
    generate_price_paths,
    run_single_simulation,
    run_batch_simulation,
)


def test_single_simulation_no_rebalance_high_threshold():
    config = SimulationConfig(
        pair=StockPairConfig(
            stock_a=StockConfig(annual_growth_rate=0.05, daily_volatility=0.01),
            stock_b=StockConfig(annual_growth_rate=0.05, daily_volatility=0.01),
            correlation=0.0,
        ),
        market=MarketConfig(fee_percent=0.0, fee_fixed=0.0),
        initial_portfolio_value=1000.0,
        threshold=10.0,  # Unreachable threshold
        threshold_type=ThresholdType.RELATIVE,
        num_days=100,
        seed=123,
    )

    prices_a, prices_b = generate_price_paths(config, num_simulations=1)
    res = run_single_simulation(config, prices_a[0], prices_b[0])

    assert len(res.rebalance_days) == 0
    assert res.total_fees_paid == 0.0
    # Without rebalancing, rebalanced portfolio values must match buy and hold values exactly
    np.testing.assert_allclose(res.portfolio_values, res.buy_and_hold_values, rtol=1e-6)


def test_batch_simulation_runs():
    config = SimulationConfig(
        num_days=50,
        num_simulations=10,
        threshold=0.02,
        seed=42,
    )

    batch_res = run_batch_simulation(config, store_sample_runs=3)

    assert len(batch_res.final_portfolio_values) == 10
    assert len(batch_res.final_buy_and_hold_values) == 10
    assert len(batch_res.sample_runs) == 3
    assert np.all(batch_res.final_portfolio_values > 0)
