"""
Visualization module for generating charts of simulation results using matplotlib.
"""

from typing import Optional
import matplotlib.pyplot as plt
import numpy as np

from twintrade.models import BatchSimulationResult


def plot_simulation_results(
    result: BatchSimulationResult,
    save_path: Optional[str] = None,
    show_plot: bool = False,
):
    """
    Generate and display or save visual charts summarizing the batch simulation.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"TwinTrade Simulation ({result.config.num_simulations} Runs, {result.config.num_days} Days, "
        f"Correlation = {result.config.pair.correlation})",
        fontsize=14,
        fontweight="bold",
    )

    # 1. Sample Price Trajectory (Run 0)
    if result.sample_runs:
        run = result.sample_runs[0]
        days = np.arange(len(run.stock_a_prices))
        ax0 = axes[0, 0]
        ax0.plot(days, run.stock_a_prices, label=f"{result.config.pair.stock_a.name}", color="blue")
        ax0.plot(days, run.stock_b_prices, label=f"{result.config.pair.stock_b.name}", color="orange")
        ax0.set_title("Stock Price Trajectories (Sample Run)")
        ax0.set_xlabel("Day")
        ax0.set_ylabel("Price ($)")
        ax0.grid(True, linestyle="--", alpha=0.6)
        ax0.legend()

        # 2. Portfolio Value Comparison (Sample Run)
        ax1 = axes[0, 1]
        ax1.plot(days, run.portfolio_values, label="Rebalanced Portfolio", color="green", linewidth=2)
        ax1.plot(days, run.buy_and_hold_values, label="Buy & Hold", color="gray", linestyle="--", linewidth=2)
        if run.rebalance_days:
            reb_y = [run.portfolio_values[d] for d in run.rebalance_days]
            ax1.scatter(run.rebalance_days, reb_y, color="red", marker="o", s=30, zorder=5, label="Rebalance Event")
        ax1.set_title("Portfolio Value Over Time (Sample Run)")
        ax1.set_xlabel("Day")
        ax1.set_ylabel("Portfolio Value ($)")
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend()

    # 3. Distribution of Final Values (All Runs)
    ax2 = axes[1, 0]
    ax2.hist(
        result.final_portfolio_values,
        bins=30,
        alpha=0.6,
        color="green",
        label="Rebalanced",
        edgecolor="black",
    )
    ax2.hist(
        result.final_buy_and_hold_values,
        bins=30,
        alpha=0.4,
        color="gray",
        label="Buy & Hold",
        edgecolor="black",
    )
    ax2.axvline(np.mean(result.final_portfolio_values), color="darkgreen", linestyle="--", label="Mean Rebalanced")
    ax2.axvline(np.mean(result.final_buy_and_hold_values), color="black", linestyle="--", label="Mean Buy & Hold")
    ax2.set_title("Final Portfolio Value Distribution")
    ax2.set_xlabel("Final Value ($)")
    ax2.set_ylabel("Frequency")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    # 4. Outperformance vs Fee Scatter / Boxplot
    ax3 = axes[1, 1]
    diffs = result.final_portfolio_values - result.final_buy_and_hold_values
    ax3.hist(diffs, bins=30, color="purple", alpha=0.7, edgecolor="black")
    ax3.axvline(0, color="red", linestyle="-", label="Zero Difference")
    ax3.axvline(np.mean(diffs), color="yellow", linestyle="--", label=f"Mean Outperformance (${np.mean(diffs):.2f})")
    ax3.set_title("Outperformance (Rebalanced - Buy & Hold)")
    ax3.set_xlabel("Dollar Difference ($)")
    ax3.set_ylabel("Frequency")
    ax3.grid(True, linestyle="--", alpha=0.6)
    ax3.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Chart saved to {save_path}")

    if show_plot:
        plt.show()

    plt.close(fig)
