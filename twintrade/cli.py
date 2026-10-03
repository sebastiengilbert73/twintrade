"""
Command Line Interface (CLI) for running TwinTrade simulations.
"""

import argparse
import json
import sys
from typing import Optional

from twintrade import __version__
from twintrade.models import (
    SimulationConfig,
    StockConfig,
    StockPairConfig,
    MarketConfig,
    ThresholdType,
)
from twintrade.engine import run_batch_simulation
from twintrade.stats import compute_summary_stats
from twintrade.visualization import plot_simulation_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="twintrade",
        description="Simulate trading of correlated asset pairs with automated threshold rebalancing.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    # Stock A Parameters
    group_a = parser.add_argument_group("Stock A Parameters")
    group_a.add_argument("--stock-a-name", type=str, default="Stock A", help="Name of Stock A")
    group_a.add_argument("--growth-a", type=float, default=0.08, help="Annual growth rate for Stock A (default: 0.08 = 8%%)")
    group_a.add_argument("--vol-a", type=float, default=0.015, help="Daily volatility for Stock A (default: 0.015 = 1.5%%)")
    group_a.add_argument("--price-a", type=float, default=100.0, help="Initial price for Stock A (default: $100.0)")

    # Stock B Parameters
    group_b = parser.add_argument_group("Stock B Parameters")
    group_b.add_argument("--stock-b-name", type=str, default="Stock B", help="Name of Stock B")
    group_b.add_argument("--growth-b", type=float, default=0.08, help="Annual growth rate for Stock B (default: 0.08 = 8%%)")
    group_b.add_argument("--vol-b", type=float, default=0.015, help="Daily volatility for Stock B (default: 0.015 = 1.5%%)")
    group_b.add_argument("--price-b", type=float, default=100.0, help="Initial price for Stock B (default: $100.0)")

    # Pair & Market Parameters
    group_pair = parser.add_argument_group("Pair & Market Parameters")
    group_pair.add_argument("--correlation", type=float, default=0.0, help="Correlation between asset returns (-1.0 to 1.0, default: 0.0)")
    group_pair.add_argument("--fee-percent", type=float, default=0.001, help="Transaction fee percent (default: 0.001 = 0.1%%)")
    group_pair.add_argument("--fee-fixed", type=float, default=0.0, help="Fixed transaction fee in $ (default: $0.0)")

    # Strategy & Simulation Parameters
    group_sim = parser.add_argument_group("Simulation Parameters")
    group_sim.add_argument("--initial-value", type=float, default=1000.0, help="Initial total portfolio value (default: $1000.0)")
    group_sim.add_argument("--threshold", type=float, default=0.05, help="Rebalancing threshold (default: 0.05 = 5%%)")
    group_sim.add_argument("--threshold-type", choices=["relative", "absolute"], default="relative", help="Threshold calculation mode")
    group_sim.add_argument("--num-days", type=int, default=252, help="Number of trading days per simulation run (default: 252)")
    group_sim.add_argument("--num-sims", type=int, default=1000, help="Number of simulation runs (default: 1000)")
    group_sim.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")

    # Output Options
    group_out = parser.add_argument_group("Output Options")
    group_out.add_argument("--plot-output", type=str, default=None, help="File path to save result chart (e.g. plot.png)")
    group_out.add_argument("--show-plot", action="store_true", help="Display interactive plot window")
    group_out.add_argument("--json-output", type=str, default=None, help="File path to save summary JSON report")

    return parser


def print_rich_report(stats) -> None:
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel

        console = Console()

        panel_content = (
            f"[bold]Simulations:[/] {stats.num_simulations:,} runs | "
            f"[bold]Days:[/] {stats.num_days} | "
            f"[bold]Initial Value:[/] ${stats.initial_value:,.2f}\n"
            f"[bold]Rebalanced Win Rate vs Buy & Hold:[/] [green]{stats.win_rate_percent:.1f}%[/]"
        )
        console.print(Panel(panel_content, title="TwinTrade Simulation Summary", expand=False))

        table = Table(title="Performance Comparison", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Rebalanced Strategy", justify="right")
        table.add_column("Buy & Hold Strategy", justify="right")

        table.add_row("Mean Final Value", f"${stats.rebalanced_stats.mean:,.2f}", f"${stats.buy_and_hold_stats.mean:,.2f}")
        table.add_row("Median Final Value", f"${stats.rebalanced_stats.median:,.2f}", f"${stats.buy_and_hold_stats.median:,.2f}")
        table.add_row("Std Dev", f"${stats.rebalanced_stats.std_dev:,.2f}", f"${stats.buy_and_hold_stats.std_dev:,.2f}")
        table.add_row("Min Final Value", f"${stats.rebalanced_stats.min_val:,.2f}", f"${stats.buy_and_hold_stats.min_val:,.2f}")
        table.add_row("Max Final Value", f"${stats.rebalanced_stats.max_val:,.2f}", f"${stats.buy_and_hold_stats.max_val:,.2f}")
        table.add_row("Mean Annualized Return (CAGR)", f"{stats.rebalanced_stats.cagr_mean:.2f}%", f"{stats.buy_and_hold_stats.cagr_mean:.2f}%")
        table.add_row("5th Percentile", f"${stats.rebalanced_stats.percentile_5:,.2f}", f"${stats.buy_and_hold_stats.percentile_5:,.2f}")
        table.add_row("95th Percentile", f"${stats.rebalanced_stats.percentile_95:,.2f}", f"${stats.buy_and_hold_stats.percentile_95:,.2f}")

        console.print(table)

        fee_table = Table(title="Trading & Transaction Stats", show_header=True, header_style="bold yellow")
        fee_table.add_column("Metric", style="cyan")
        fee_table.add_column("Value", justify="right")

        fee_table.add_row("Mean Rebalance Events / Run", f"{stats.mean_rebalances:.1f}")
        fee_table.add_row("Rebalance Events Range", f"{stats.min_rebalances} - {stats.max_rebalances}")
        fee_table.add_row("Mean Total Transaction Fees Paid", f"${stats.mean_total_fees:,.2f}")
        fee_table.add_row("Mean Outperformance vs B&H ($)", f"${stats.mean_outperformance:+,.2f}")
        fee_table.add_row("Mean Outperformance vs B&H (%)", f"{stats.mean_outperformance_pct:+.2f}%")
        fee_table.add_row("Empirical Correlation (Mean)", f"{stats.mean_empirical_correlation:+.4f}")
        fee_table.add_row("Empirical Correlation (Std Dev)", f"{stats.std_empirical_correlation:.4f}")

        console.print(fee_table)

    except ImportError:
        # Plain text fallback
        print("=" * 60)
        print("TWINTRADE SIMULATION SUMMARY")
        print("=" * 60)
        print(f"Simulations: {stats.num_simulations} | Days: {stats.num_days} | Initial Value: ${stats.initial_value:,.2f}")
        print(f"Win Rate vs Buy & Hold: {stats.win_rate_percent:.1f}%")
        print("-" * 60)
        print(f"Rebalanced Mean Final Value: ${stats.rebalanced_stats.mean:,.2f}")
        print(f"Buy & Hold Mean Final Value: ${stats.buy_and_hold_stats.mean:,.2f}")
        print(f"Mean Outperformance:         ${stats.mean_outperformance:+,.2f} ({stats.mean_outperformance_pct:+.2f}%)")
        print(f"Empirical Correlation:       {stats.mean_empirical_correlation:+.4f} ± {stats.std_empirical_correlation:.4f}")
        print(f"Mean Rebalance Events:       {stats.mean_rebalances:.1f}")
        print(f"Mean Total Fees Paid:        ${stats.mean_total_fees:,.2f}")
        print("=" * 60)


def run_cli(args_list: Optional[list] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(args_list)

    stock_a = StockConfig(
        name=args.stock_a_name,
        annual_growth_rate=args.growth_a,
        daily_volatility=args.vol_a,
        initial_price=args.price_a,
    )

    stock_b = StockConfig(
        name=args.stock_b_name,
        annual_growth_rate=args.growth_b,
        daily_volatility=args.vol_b,
        initial_price=args.price_b,
    )

    pair = StockPairConfig(
        stock_a=stock_a,
        stock_b=stock_b,
        correlation=args.correlation,
    )

    market = MarketConfig(
        fee_percent=args.fee_percent,
        fee_fixed=args.fee_fixed,
    )

    config = SimulationConfig(
        pair=pair,
        market=market,
        initial_portfolio_value=args.initial_value,
        threshold=args.threshold,
        threshold_type=ThresholdType(args.threshold_type),
        num_days=args.num_days,
        num_simulations=args.num_sims,
        seed=args.seed,
    )

    batch_result = run_batch_simulation(config)
    stats = compute_summary_stats(batch_result)

    print_rich_report(stats)

    if args.json_output:
        report_data = {
            "num_simulations": stats.num_simulations,
            "num_days": stats.num_days,
            "initial_value": stats.initial_value,
            "win_rate_percent": stats.win_rate_percent,
            "mean_outperformance": stats.mean_outperformance,
            "mean_outperformance_pct": stats.mean_outperformance_pct,
            "mean_rebalances": stats.mean_rebalances,
            "mean_total_fees": stats.mean_total_fees,
            "empirical_correlation": {
                "mean": stats.mean_empirical_correlation,
                "std_dev": stats.std_empirical_correlation,
            },
            "rebalanced": {
                "mean": stats.rebalanced_stats.mean,
                "median": stats.rebalanced_stats.median,
                "std_dev": stats.rebalanced_stats.std_dev,
                "cagr_mean": stats.rebalanced_stats.cagr_mean,
            },
            "buy_and_hold": {
                "mean": stats.buy_and_hold_stats.mean,
                "median": stats.buy_and_hold_stats.median,
                "std_dev": stats.buy_and_hold_stats.std_dev,
                "cagr_mean": stats.buy_and_hold_stats.cagr_mean,
            }
        }
        with open(args.json_output, "w") as f:
            json.dump(report_data, f, indent=2)
        print(f"JSON summary report written to {args.json_output}")

    if args.plot_output or args.show_plot:
        plot_simulation_results(batch_result, save_path=args.plot_output, show_plot=args.show_plot)

    return 0
