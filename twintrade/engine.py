"""
Simulation engine for generating correlated asset prices and running trading simulations.
"""

from typing import List, Tuple, Optional
import numpy as np

from twintrade.models import (
    SimulationConfig,
    SingleSimulationResult,
    BatchSimulationResult,
    ThresholdType,
)


def generate_price_paths(
    config: SimulationConfig,
    num_simulations: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate correlated daily price paths for Stock A and Stock B using
    Geometric Brownian Motion (GBM).

    Returns:
        Tuple of (prices_a, prices_b) each of shape (num_simulations, num_days + 1).
    """
    if num_simulations is None:
        num_simulations = config.num_simulations

    if rng is None:
        if config.seed is not None:
            rng = np.random.default_rng(config.seed)
        else:
            rng = np.random.default_rng()

    num_days = config.num_days
    trading_days = config.trading_days_per_year

    # Daily drift (log returns) and volatility for Stock A & B
    mu_a = np.log(1.0 + config.pair.stock_a.annual_growth_rate) / trading_days
    sigma_a = config.pair.stock_a.daily_volatility

    mu_b = np.log(1.0 + config.pair.stock_b.annual_growth_rate) / trading_days
    sigma_b = config.pair.stock_b.daily_volatility

    rho = config.pair.correlation
    # Clamp correlation to [-1.0, 1.0]
    rho = max(-1.0, min(1.0, rho))

    # Generate independent standard normal random variables: shape (num_simulations, num_days, 2)
    z_indep = rng.standard_normal(size=(num_simulations, num_days, 2))

    # Apply Cholesky decomposition to create correlated normals Z_A, Z_B
    z_a = z_indep[:, :, 0]
    z_b = rho * z_indep[:, :, 0] + np.sqrt(max(0.0, 1.0 - rho ** 2)) * z_indep[:, :, 1]

    # Calculate log returns per day
    # dS/S = exp((mu - 0.5 * sigma^2) + sigma * Z)
    ret_a = (mu_a - 0.5 * sigma_a ** 2) + sigma_a * z_a
    ret_b = (mu_b - 0.5 * sigma_b ** 2) + sigma_b * z_b

    # Cumulative log returns
    cum_ret_a = np.zeros((num_simulations, num_days + 1))
    cum_ret_b = np.zeros((num_simulations, num_days + 1))

    cum_ret_a[:, 1:] = np.cumsum(ret_a, axis=1)
    cum_ret_b[:, 1:] = np.cumsum(ret_b, axis=1)

    # Prices path
    prices_a = config.pair.stock_a.initial_price * np.exp(cum_ret_a)
    prices_b = config.pair.stock_b.initial_price * np.exp(cum_ret_b)

    return prices_a, prices_b


def calculate_rebalance(
    v_high: float,
    v_low: float,
    fee_percent: float,
    fee_fixed: float,
) -> Tuple[float, float, float]:
    """
    Calculate the dollar amount X to sell from higher-value stock to equalize values
    with lower-value stock after deducting sell and buy transaction fees.

    Returns:
        Tuple of (amount_sold_X, value_added_to_low_stock, total_fees_paid).
    """
    if v_high <= v_low:
        return 0.0, 0.0, 0.0

    c_pct = max(0.0, fee_percent)
    c_fix = max(0.0, fee_fixed)

    # Formula derived from: V_high - X = V_low + net_added_to_low
    denominator = 2.0 - 2.0 * c_pct + (c_pct ** 2)
    if denominator <= 0:
        return 0.0, 0.0, 0.0

    numerator = (v_high - v_low) + (2.0 - c_pct) * c_fix
    x_sell = numerator / denominator

    # Cap X sell to available value in v_high
    x_sell = max(0.0, min(v_high, x_sell))

    # Sell transaction
    fee_sell = c_pct * x_sell + c_fix
    net_sell = max(0.0, x_sell - fee_sell)

    # Buy transaction
    fee_buy = c_pct * net_sell + c_fix
    value_added = max(0.0, net_sell - fee_buy)

    total_fee = (x_sell - net_sell) + (net_sell - value_added)

    return x_sell, value_added, total_fee


def run_single_simulation(
    config: SimulationConfig,
    prices_a: np.ndarray,
    prices_b: np.ndarray,
) -> SingleSimulationResult:
    """
    Run trading simulation along a single pre-generated price path pair.
    """
    num_days = len(prices_a) - 1

    # Initial allocation: 50% in Stock A, 50% in Stock B
    initial_v_a = config.initial_portfolio_value / 2.0
    initial_v_b = config.initial_portfolio_value / 2.0

    p_a0 = prices_a[0]
    p_b0 = prices_b[0]

    shares_a = initial_v_a / p_a0
    shares_b = initial_v_b / p_b0

    # Buy and hold strategy uses fixed initial shares
    bh_shares_a = shares_a
    bh_shares_b = shares_b

    portfolio_values = np.zeros(num_days + 1)
    buy_and_hold_values = np.zeros(num_days + 1)
    shares_a_history = np.zeros(num_days + 1)
    shares_b_history = np.zeros(num_days + 1)

    portfolio_values[0] = config.initial_portfolio_value
    buy_and_hold_values[0] = config.initial_portfolio_value
    shares_a_history[0] = shares_a
    shares_b_history[0] = shares_b

    rebalance_days: List[int] = []
    total_fees_paid = 0.0

    for day in range(1, num_days + 1):
        pa = prices_a[day]
        pb = prices_b[day]

        # Values before rebalance
        va = shares_a * pa
        vb = shares_b * pb
        v_total = va + vb

        # Buy & hold value
        bh_val = bh_shares_a * pa + bh_shares_b * pb
        buy_and_hold_values[day] = bh_val

        # Check threshold condition
        diff = abs(va - vb)
        if config.threshold_type == ThresholdType.RELATIVE:
            condition = (v_total > 0) and ((diff / v_total) > config.threshold)
        else:  # ABSOLUTE
            condition = diff > config.threshold

        if condition and v_total > 0:
            if va > vb:
                x_sell, val_added, fee = calculate_rebalance(
                    va, vb, config.market.fee_percent, config.market.fee_fixed
                )
                if x_sell > 0:
                    va_new = va - x_sell
                    vb_new = vb + val_added
                    shares_a = va_new / pa
                    shares_b = vb_new / pb
                    total_fees_paid += fee
                    rebalance_days.append(day)
            elif vb > va:
                x_sell, val_added, fee = calculate_rebalance(
                    vb, va, config.market.fee_percent, config.market.fee_fixed
                )
                if x_sell > 0:
                    vb_new = vb - x_sell
                    va_new = va + val_added
                    shares_b = vb_new / pb
                    shares_a = va_new / pa
                    total_fees_paid += fee
                    rebalance_days.append(day)

        portfolio_values[day] = shares_a * pa + shares_b * pb
        shares_a_history[day] = shares_a
        shares_b_history[day] = shares_b

    return SingleSimulationResult(
        portfolio_values=portfolio_values,
        buy_and_hold_values=buy_and_hold_values,
        stock_a_prices=prices_a,
        stock_b_prices=prices_b,
        stock_a_shares=shares_a_history,
        stock_b_shares=shares_b_history,
        rebalance_days=rebalance_days,
        total_fees_paid=total_fees_paid,
        final_portfolio_value=portfolio_values[-1],
        final_buy_and_hold_value=buy_and_hold_values[-1],
    )


def run_batch_simulation(
    config: SimulationConfig,
    store_sample_runs: int = 5,
) -> BatchSimulationResult:
    """
    Run batch simulations for defined number of runs and return aggregated results.
    """
    prices_a, prices_b = generate_price_paths(config)
    num_sims = config.num_simulations

    final_portfolio_values = np.zeros(num_sims)
    final_buy_and_hold_values = np.zeros(num_sims)
    rebalance_counts = np.zeros(num_sims, dtype=int)
    total_fees = np.zeros(num_sims)
    sample_runs: List[SingleSimulationResult] = []

    # Compute empirical correlation for each run (Pearson correlation of daily log-returns)
    returns_a = np.diff(np.log(prices_a), axis=1)
    returns_b = np.diff(np.log(prices_b), axis=1)
    dev_a = returns_a - np.mean(returns_a, axis=1, keepdims=True)
    dev_b = returns_b - np.mean(returns_b, axis=1, keepdims=True)
    denom = np.sqrt(np.sum(dev_a ** 2, axis=1) * np.sum(dev_b ** 2, axis=1))
    empirical_correlations = np.where(denom > 0, np.sum(dev_a * dev_b, axis=1) / denom, 0.0)

    for i in range(num_sims):
        res = run_single_simulation(config, prices_a[i], prices_b[i])
        final_portfolio_values[i] = res.final_portfolio_value
        final_buy_and_hold_values[i] = res.final_buy_and_hold_value
        rebalance_counts[i] = len(res.rebalance_days)
        total_fees[i] = res.total_fees_paid

        if i < store_sample_runs:
            sample_runs.append(res)

    return BatchSimulationResult(
        config=config,
        final_portfolio_values=final_portfolio_values,
        final_buy_and_hold_values=final_buy_and_hold_values,
        rebalance_counts=rebalance_counts,
        total_fees_paid=total_fees,
        empirical_correlations=empirical_correlations,
        sample_runs=sample_runs,
    )
