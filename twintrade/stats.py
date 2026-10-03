"""
Statistical calculation module for analyzing simulation results.
"""

from dataclasses import dataclass
from typing import Dict, Any
import numpy as np

from twintrade.models import BatchSimulationResult


@dataclass
class PerformanceMetrics:
    mean: float
    median: float
    std_dev: float
    min_val: float
    max_val: float
    cagr_mean: float       # Compound Annual Growth Rate (%)
    percentile_5: float
    percentile_25: float
    percentile_75: float
    percentile_95: float


@dataclass
class SummaryStats:
    num_simulations: int
    num_days: int
    initial_value: float
    rebalanced_stats: PerformanceMetrics
    buy_and_hold_stats: PerformanceMetrics
    win_rate_percent: float          # % of runs where rebalanced > buy & hold
    mean_outperformance: float       # Dollar difference (rebalanced - buy_and_hold)
    mean_outperformance_pct: float   # % difference
    mean_rebalances: float
    min_rebalances: int
    max_rebalances: int
    mean_total_fees: float
    max_total_fees: float
    mean_empirical_correlation: float
    std_empirical_correlation: float


def _calc_metrics(
    values: np.ndarray, initial_value: float, num_days: int, trading_days: int
) -> PerformanceMetrics:
    years = num_days / trading_days
    # CAGR = (V_final / V_initial) ** (1 / years) - 1
    returns = values / initial_value
    cagrs = np.where(returns > 0, (returns ** (1.0 / max(years, 1e-6))) - 1.0, -1.0)

    p5, p25, median, p75, p95 = np.percentile(values, [5, 25, 50, 75, 95])

    return PerformanceMetrics(
        mean=float(np.mean(values)),
        median=float(median),
        std_dev=float(np.std(values)),
        min_val=float(np.min(values)),
        max_val=float(np.max(values)),
        cagr_mean=float(np.mean(cagrs)) * 100.0,
        percentile_5=float(p5),
        percentile_25=float(p25),
        percentile_75=float(p75),
        percentile_95=float(p95),
    )


def compute_summary_stats(result: BatchSimulationResult) -> SummaryStats:
    """
    Compute comprehensive summary statistics for a batch simulation result.
    """
    config = result.config
    initial_val = config.initial_portfolio_value
    num_days = config.num_days
    trading_days = config.trading_days_per_year

    reb_metrics = _calc_metrics(result.final_portfolio_values, initial_val, num_days, trading_days)
    bh_metrics = _calc_metrics(result.final_buy_and_hold_values, initial_val, num_days, trading_days)

    wins = result.final_portfolio_values > result.final_buy_and_hold_values
    win_rate = (np.sum(wins) / len(wins)) * 100.0

    outperformance = result.final_portfolio_values - result.final_buy_and_hold_values
    mean_outperf = float(np.mean(outperformance))
    mean_outperf_pct = float(np.mean((outperformance / result.final_buy_and_hold_values) * 100.0))

    mean_corr = float(np.mean(result.empirical_correlations)) if len(result.empirical_correlations) > 0 else 0.0
    std_corr = float(np.std(result.empirical_correlations)) if len(result.empirical_correlations) > 0 else 0.0

    return SummaryStats(
        num_simulations=config.num_simulations,
        num_days=config.num_days,
        initial_value=initial_val,
        rebalanced_stats=reb_metrics,
        buy_and_hold_stats=bh_metrics,
        win_rate_percent=win_rate,
        mean_outperformance=mean_outperf,
        mean_outperformance_pct=mean_outperf_pct,
        mean_rebalances=float(np.mean(result.rebalance_counts)),
        min_rebalances=int(np.min(result.rebalance_counts)),
        max_rebalances=int(np.max(result.rebalance_counts)),
        mean_total_fees=float(np.mean(result.total_fees_paid)),
        max_total_fees=float(np.max(result.total_fees_paid)),
        mean_empirical_correlation=mean_corr,
        std_empirical_correlation=std_corr,
    )
