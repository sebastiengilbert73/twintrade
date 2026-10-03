"""
Unit tests for mathematical derivations and core algorithms.
"""

import pytest
import numpy as np
from twintrade.models import SimulationConfig, StockPairConfig, StockConfig
from twintrade.engine import calculate_rebalance, generate_price_paths


def test_calculate_rebalance_zero_fees():
    v_high = 600.0
    v_low = 400.0
    fee_percent = 0.0
    fee_fixed = 0.0

    x_sell, val_added, fee = calculate_rebalance(v_high, v_low, fee_percent, fee_fixed)

    assert pytest.approx(x_sell, abs=1e-6) == 100.0
    assert pytest.approx(val_added, abs=1e-6) == 100.0
    assert pytest.approx(fee, abs=1e-6) == 0.0
    assert pytest.approx(v_high - x_sell, abs=1e-6) == v_low + val_added


def test_calculate_rebalance_percent_fees():
    v_high = 600.0
    v_low = 400.0
    fee_percent = 0.01  # 1%
    fee_fixed = 0.0

    x_sell, val_added, fee = calculate_rebalance(v_high, v_low, fee_percent, fee_fixed)

    va_final = v_high - x_sell
    vb_final = v_low + val_added

    # Check parity after rebalance
    assert pytest.approx(va_final, abs=1e-4) == vb_final
    assert fee > 0.0


def test_calculate_rebalance_fixed_and_percent_fees():
    v_high = 1000.0
    v_low = 500.0
    fee_percent = 0.002  # 0.2%
    fee_fixed = 5.0      # $5 fixed fee per transaction

    x_sell, val_added, fee = calculate_rebalance(v_high, v_low, fee_percent, fee_fixed)

    va_final = v_high - x_sell
    vb_final = v_low + val_added

    assert pytest.approx(va_final, abs=1e-4) == vb_final


def test_correlated_price_paths_correlation():
    # Test that empirical correlation matches input correlation over large N
    config = SimulationConfig(
        pair=StockPairConfig(
            stock_a=StockConfig(daily_volatility=0.02),
            stock_b=StockConfig(daily_volatility=0.02),
            correlation=-0.8,
        ),
        num_days=1000,
        num_simulations=50,
        seed=42,
    )

    prices_a, prices_b = generate_price_paths(config)

    # Compute daily log returns across all runs
    returns_a = np.diff(np.log(prices_a), axis=1).flatten()
    returns_b = np.diff(np.log(prices_b), axis=1).flatten()

    emp_corr = np.corrcoef(returns_a, returns_b)[0, 1]

    # Empirical correlation should be close to -0.8
    assert pytest.approx(emp_corr, abs=0.05) == -0.8
