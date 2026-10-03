"""
Unit tests for summary statistics and metrics calculations.
"""

import numpy as np
from twintrade.models import SimulationConfig, BatchSimulationResult
from twintrade.stats import compute_summary_stats


def test_compute_summary_stats():
    config = SimulationConfig(
        initial_portfolio_value=1000.0,
        num_days=252,
        num_simulations=5,
    )

    result = BatchSimulationResult(
        config=config,
        final_portfolio_values=np.array([1100.0, 1200.0, 1050.0, 1300.0, 950.0]),
        final_buy_and_hold_values=np.array([1000.0, 1150.0, 1100.0, 1200.0, 900.0]),
        rebalance_counts=np.array([2, 5, 1, 4, 3]),
        total_fees_paid=np.array([1.5, 3.2, 0.8, 2.9, 2.1]),
        empirical_correlations=np.array([0.1, -0.2, 0.05, 0.3, -0.1]),
    )

    stats = compute_summary_stats(result)

    assert stats.num_simulations == 5
    assert stats.initial_value == 1000.0
    # Win count: [1100>1000 (T), 1200>1150 (T), 1050>1100 (F), 1300>1200 (T), 950>900 (T)] => 4 wins / 5 = 80%
    assert stats.win_rate_percent == 80.0
    assert stats.mean_rebalances == 3.0
    assert stats.rebalanced_stats.mean == 1120.0
    assert stats.buy_and_hold_stats.mean == 1070.0
    assert abs(stats.mean_empirical_correlation - 0.03) < 1e-6
    assert stats.std_empirical_correlation > 0.0
