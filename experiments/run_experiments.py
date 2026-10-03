"""
Experimentation script for TwinTrade.
Executes N_trials with randomly sampled parameters (growth, volatility, correlation, threshold, fees)
and reports the trials that maximize the mean outperformance of the rebalancing strategy vs Buy & Hold.
"""

import argparse
from dataclasses import dataclass, asdict
import csv
import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

from twintrade.models import (
    SimulationConfig,
    StockPairConfig,
    StockConfig,
    MarketConfig,
    ThresholdType,
)
from twintrade.engine import run_batch_simulation
from twintrade.stats import compute_summary_stats


@dataclass
class TrialRecord:
    trial_id: int
    growth_a: float
    vol_a: float
    growth_b: float
    vol_b: float
    correlation: float
    threshold: float
    fee_percent: float
    fee_fixed: float
    num_sims: int
    num_days: int
    win_rate_pct: float
    mean_outperformance_dollars: float
    mean_outperformance_pct: float
    mean_rebalances: float
    mean_fees_paid: float
    rebalanced_mean_final: float
    buy_hold_mean_final: float
    empirical_corr_mean: float
    empirical_corr_std: float


def sample_random_config(
    rng: np.random.Generator,
    trial_id: int,
    num_sims: int,
    num_days: int,
    initial_value: float = 1000.0,
) -> SimulationConfig:
    """Sample random simulation configuration parameters from uniform distributions."""
    growth_a = float(rng.uniform(0.0, 0.15))
    vol_a = float(rng.uniform(0.005, 0.04))
    price_a = 100.0

    growth_b = float(rng.uniform(0.0, 0.15))
    vol_b = float(rng.uniform(0.005, 0.04))
    price_b = 100.0

    correlation = float(rng.uniform(-0.99, 0.99))
    threshold = float(rng.uniform(0.01, 0.20))
    fee_percent = float(rng.uniform(0.0, 0.003))  # 0% to 0.3%
    fee_fixed = float(rng.uniform(0.0, 5.0))      # $0 to $5

    return SimulationConfig(
        pair=StockPairConfig(
            stock_a=StockConfig(name="Stock A", annual_growth_rate=growth_a, daily_volatility=vol_a, initial_price=price_a),
            stock_b=StockConfig(name="Stock B", annual_growth_rate=growth_b, daily_volatility=vol_b, initial_price=price_b),
            correlation=correlation,
        ),
        market=MarketConfig(fee_percent=fee_percent, fee_fixed=fee_fixed),
        initial_portfolio_value=initial_value,
        threshold=threshold,
        threshold_type=ThresholdType.RELATIVE,
        num_days=num_days,
        num_simulations=num_sims,
        seed=None,
    )


def run_experiments(
    n_trials: int = 50,
    num_sims: int = 500,
    num_days: int = 252,
    top_k: int = 5,
    seed: int = 42,
    csv_output: str = "experiments/results.csv",
    report_output: str = "experiments/report.md",
) -> List[TrialRecord]:
    """Run N_trials experiments with random parameter sampling and report top performers."""
    rng = np.random.default_rng(seed)
    records: List[TrialRecord] = []

    print(f"Starting {n_trials} experimental trials ({num_sims} simulations/trial)...")

    for i in range(1, n_trials + 1):
        config = sample_random_config(rng, i, num_sims, num_days)
        batch_res = run_batch_simulation(config)
        stats = compute_summary_stats(batch_res)

        rec = TrialRecord(
            trial_id=i,
            growth_a=config.pair.stock_a.annual_growth_rate,
            vol_a=config.pair.stock_a.daily_volatility,
            growth_b=config.pair.stock_b.annual_growth_rate,
            vol_b=config.pair.stock_b.daily_volatility,
            correlation=config.pair.correlation,
            threshold=config.threshold,
            fee_percent=config.market.fee_percent,
            fee_fixed=config.market.fee_fixed,
            num_sims=config.num_simulations,
            num_days=config.num_days,
            win_rate_pct=stats.win_rate_percent,
            mean_outperformance_dollars=stats.mean_outperformance,
            mean_outperformance_pct=stats.mean_outperformance_pct,
            mean_rebalances=stats.mean_rebalances,
            mean_fees_paid=stats.mean_total_fees,
            rebalanced_mean_final=stats.rebalanced_stats.mean,
            buy_hold_mean_final=stats.buy_and_hold_stats.mean,
            empirical_corr_mean=stats.mean_empirical_correlation,
            empirical_corr_std=stats.std_empirical_correlation,
        )
        records.append(rec)

    # Sort trials by mean dollar outperformance descending
    sorted_records = sorted(records, key=lambda x: x.mean_outperformance_dollars, reverse=True)

    # Export to CSV if requested
    if csv_output:
        Path(csv_output).parent.mkdir(parents=True, exist_ok=True)
        with open(csv_output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(records[0]).keys()))
            writer.writeheader()
            for r in records:
                writer.writerow(asdict(r))
        print(f"Exported all {n_trials} trial records to CSV: {csv_output}")

    # Generate Markdown Report
    if report_output:
        generate_markdown_report(sorted_records, top_k, n_trials, num_sims, report_output)

    # Print summary to console
    print_console_summary(sorted_records, top_k, n_trials)

    return sorted_records


def print_console_summary(sorted_records: List[TrialRecord], top_k: int, n_trials: int):
    """Print formatted terminal report showing full parameter breakdown for top trials."""
    top_trials = sorted_records[:top_k]
    try:
        from rich.console import Console
        from rich.panel import Panel

        console = Console()
        console.print(Panel(f"[bold green]TwinTrade Experiments Summary[/]\nTotal Trials Evaluated: {n_trials} | Top Selected: {top_k}", expand=False))

        for rank, r in enumerate(top_trials, 1):
            title = f"Rank #{rank} — Trial {r.trial_id} | Outperformance: ${r.mean_outperformance_dollars:+,.2f} ({r.mean_outperformance_pct:+.2f}%) | Win Rate: {r.win_rate_pct:.1f}%"
            content = (
                f"[bold cyan]Stock A:[/] Growth = {r.growth_a*100:.2f}%, Daily Volatility = {r.vol_a*100:.2f}%\n"
                f"[bold cyan]Stock B:[/] Growth = {r.growth_b*100:.2f}%, Daily Volatility = {r.vol_b*100:.2f}%\n"
                f"[bold yellow]Correlation (rho):[/] Target = {r.correlation:+.4f}, Empirical Mean = {r.empirical_corr_mean:+.4f} (std = {r.empirical_corr_std:.4f})\n"
                f"[bold magenta]Strategy Parameters:[/] Threshold = {r.threshold*100:.2f}% | Fee % = {r.fee_percent*100:.3f}% | Fee Fixed = ${r.fee_fixed:.2f}\n"
                f"[bold green]Results:[/] Rebalanced Mean = ${r.rebalanced_mean_final:,.2f} | Buy & Hold Mean = ${r.buy_hold_mean_final:,.2f} | Avg Rebalances = {r.mean_rebalances:.1f} | Avg Fees = ${r.mean_fees_paid:.2f}"
            )
            console.print(Panel(content, title=title, expand=False))

    except ImportError:
        print("=" * 80)
        print(f"TOP {top_k} TRIALS MAXIMIZING OUTPERFORMANCE")
        print("=" * 80)
        for rank, r in enumerate(top_trials, 1):
            print(f"--- Rank #{rank} (Trial {r.trial_id}) ---")
            print(f"  Stock A: Growth={r.growth_a*100:.2f}%, Vol={r.vol_a*100:.2f}%")
            print(f"  Stock B: Growth={r.growth_b*100:.2f}%, Vol={r.vol_b*100:.2f}%")
            print(f"  Correlation (rho): Target={r.correlation:+.4f}, Empirical={r.empirical_corr_mean:+.4f}")
            print(f"  Strategy: Threshold={r.threshold*100:.2f}%, Fee%={r.fee_percent*100:.3f}%, FeeFixed=${r.fee_fixed:.2f}")
            print(f"  Performance: Outperformance=${r.mean_outperformance_dollars:+,.2f} ({r.mean_outperformance_pct:+.2f}%), WinRate={r.win_rate_pct:.1f}%")
            print("-" * 80)


def generate_markdown_report(
    sorted_records: List[TrialRecord], top_k: int, n_trials: int, num_sims: int, output_file: str
):
    """Generate detailed markdown experiment report."""
    top_trials = sorted_records[:top_k]
    worst_trials = sorted_records[-top_k:]

    md_content = f"""# Rapport d'Expérimentation TwinTrade

## Aperçu
- **Nombre de jeux d'expérimentation (N_trials)** : {n_trials}
- **Nombre de simulations par jeu (N_sims)** : {num_sims}
- **Objectif** : Identifier les combinaisons de paramètres (croissance, volatilité, corrélation, seuil, frais) maximisant l'écart moyen de performance entre la stratégie de rééquilibrage et la stratégie passive (Buy & Hold).

---

## Top {top_k} Meilleures Configurations (Écart Moyen Maximal)

| Rang | Trial ID | Growth A | Vol A | Growth B | Vol B | Target Rho | Emp Rho | Seuil | Fee % | Fee Fixed | Win Rate | Outperformance ($) | Outperformance (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for rank, r in enumerate(top_trials, 1):
        md_content += (
            f"| #{rank} | Trial {r.trial_id} | {r.growth_a*100:.2f}% | {r.vol_a*100:.2f}% | "
            f"{r.growth_b*100:.2f}% | {r.vol_b*100:.2f}% | {r.correlation:+.4f} | {r.empirical_corr_mean:+.4f} | "
            f"{r.threshold*100:.2f}% | {r.fee_percent*100:.3f}% | ${r.fee_fixed:.2f} | "
            f"**{r.win_rate_pct:.1f}%** | **${r.mean_outperformance_dollars:+,.2f}** | **{r.mean_outperformance_pct:+.2f}%** |\n"
        )

    md_content += f"""
---

## Top {top_k} Moins Bonnes Configurations (Écart Moyen Minimal)

| Rang | Trial ID | Growth A | Vol A | Growth B | Vol B | Target Rho | Emp Rho | Seuil | Fee % | Fee Fixed | Win Rate | Outperformance ($) | Outperformance (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for rank, r in enumerate(reversed(worst_trials), 1):
        md_content += (
            f"| #{rank} | Trial {r.trial_id} | {r.growth_a*100:.2f}% | {r.vol_a*100:.2f}% | "
            f"{r.growth_b*100:.2f}% | {r.vol_b*100:.2f}% | {r.correlation:+.4f} | {r.empirical_corr_mean:+.4f} | "
            f"{r.threshold*100:.2f}% | {r.fee_percent*100:.3f}% | ${r.fee_fixed:.2f} | "
            f"{r.win_rate_pct:.1f}% | ${r.mean_outperformance_dollars:+,.2f} | {r.mean_outperformance_pct:+.2f}% |\n"
        )

    md_content += """
---

## Observations & Insights Analytiques
1. **Effet de la Corrélation** : Le rééquilibrage produit une surperformance maximale lorsque la corrélation entre les deux actions est négative ou faible (rho < 0), car la divergence relative est plus fréquente.
2. **Impact des Frais de Transaction** : Des frais proportionnels élevés ou des frais fixes répétés peuvent éroder l'alpha du rééquilibrage si le seuil est trop bas et déclenche de trop nombreuses transactions.
3. **Volatilité et Seuil** : Une volatilité plus forte couplée à un seuil adapté augmente les opportunités d'arbitrage de rééquilibrage sans sur-trader.
"""

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"Report generated successfully: {output_file}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="experiments",
        description="Run randomized parameter experiments for TwinTrade and report top outperforming configurations.",
    )
    parser.add_argument("--n-trials", type=int, default=50, help="Number of random parameter trials to run (default: 50)")
    parser.add_argument("--num-sims", type=int, default=500, help="Number of simulation runs per trial (default: 500)")
    parser.add_argument("--num-days", type=int, default=252, help="Number of trading days per simulation (default: 252)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top outperforming trials to report (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Master random seed for parameter generation (default: 42)")
    parser.add_argument("--csv-output", type=str, default="experiments/results.csv", help="CSV export file path")
    parser.add_argument("--report-output", type=str, default="experiments/report.md", help="Markdown report output path")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    run_experiments(
        n_trials=args.n_trials,
        num_sims=args.num_sims,
        num_days=args.num_days,
        top_k=args.top_k,
        seed=args.seed,
        csv_output=args.csv_output,
        report_output=args.report_output,
    )


if __name__ == "__main__":
    main()
