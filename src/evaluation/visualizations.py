"""
Visualization module for execution analysis.

Generates publication-quality plots for:
  1. Strategy comparison bar charts
  2. Implementation shortfall distributions
  3. A-C efficient frontier (the canonical plot)
  4. Execution trajectory visualization
  5. Regime comparison heatmap

All plots save to reports/figures/ by default.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Lazy import matplotlib (heavy dependency)
_plt = None
_sns = None


def _get_plt():
    global _plt
    if _plt is None:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend (works on servers)
        import matplotlib.pyplot as plt
        plt.style.use("seaborn-v0_8-whitegrid")
        _plt = plt
    return _plt


def _get_sns():
    global _sns
    if _sns is None:
        import seaborn as sns
        _sns = sns
    return _sns


def save_fig(fig, filename: str, output_dir: str = "reports/figures"):
    """Save figure to disk."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    filepath = path / filename
    fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
    logger.info(f"Saved figure: {filepath}")
    return filepath


def plot_strategy_comparison(
    results_df: pd.DataFrame,
    metric: str = "is_mean_bps",
    title: str = "Strategy Comparison: Implementation Shortfall",
    output_dir: str = "reports/figures",
) -> Path:
    """Bar chart comparing strategies on a single metric."""
    plt = _get_plt()

    fig, ax = plt.subplots(figsize=(12, 6))

    strategies = results_df["strategy"].tolist()
    values = results_df[metric].tolist()

    # Color: green for low cost, red for high cost
    colors = ["#2ecc71" if v == min(values) else
              "#e74c3c" if v == max(values) else
              "#3498db" for v in values]

    bars = ax.barh(strategies, values, color=colors, edgecolor="white", linewidth=0.5)

    # Add error bars if std is available
    std_col = metric.replace("mean", "std")
    if std_col in results_df.columns:
        stds = results_df[std_col].tolist()
        ax.errorbar(values, range(len(strategies)), xerr=stds,
                     fmt="none", color="black", capsize=3, linewidth=1)

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(val + max(abs(v) for v in values) * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.2f}", va="center", fontsize=10, fontweight="bold")

    ax.set_xlabel("Implementation Shortfall (bps)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    return save_fig(fig, "strategy_comparison.png", output_dir)


def plot_is_distributions(
    all_results: dict,
    output_dir: str = "reports/figures",
) -> Path:
    """Histogram/KDE of implementation shortfall for each strategy.

    all_results: dict of {name: list[ExecutionResult]}
    """
    plt = _get_plt()

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
              "#1abc9c", "#e67e22", "#34495e"]

    for i, (name, results) in enumerate(all_results.items()):
        is_vals = [r.implementation_shortfall_bps for r in results]
        color = colors[i % len(colors)]
        ax.hist(is_vals, bins=50, alpha=0.4, label=name, color=color, density=True)
        # Add mean line
        mean_val = np.mean(is_vals)
        ax.axvline(mean_val, color=color, linestyle="--", linewidth=2, alpha=0.8)

    ax.set_xlabel("Implementation Shortfall (bps)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Distribution of Implementation Shortfall by Strategy", fontsize=14,
                 fontweight="bold")
    ax.legend(fontsize=10)

    plt.tight_layout()
    return save_fig(fig, "is_distributions.png", output_dir)


def plot_efficient_frontier(
    lambda_results: list[tuple[float, float, float]],
    output_dir: str = "reports/figures",
) -> Path:
    """Plot the Almgren-Chriss efficient frontier.

    THIS IS THE CANONICAL PLOT in execution research.
    X-axis: Expected cost (IS mean)
    Y-axis: Cost variance (IS std)
    Each point = one value of λ

    lambda_results: list of (lambda, is_mean, is_std)
    """
    plt = _get_plt()

    fig, ax = plt.subplots(figsize=(10, 7))

    lambdas = [r[0] for r in lambda_results]
    is_means = [r[1] for r in lambda_results]
    is_stds = [r[2] for r in lambda_results]

    # Plot the frontier
    scatter = ax.scatter(is_means, is_stds, c=np.log10(lambdas),
                          cmap="viridis", s=120, edgecolors="black", linewidth=1,
                          zorder=5)

    # Connect points to show the frontier
    sorted_idx = np.argsort(is_means)
    ax.plot([is_means[i] for i in sorted_idx],
            [is_stds[i] for i in sorted_idx],
            color="gray", linewidth=1, alpha=0.5, linestyle="--")

    # Label each point with its λ value
    for lam, mean, std in lambda_results:
        ax.annotate(f"λ={lam}", (mean, std),
                     textcoords="offset points", xytext=(8, 8),
                     fontsize=9, fontweight="bold",
                     arrowprops=dict(arrowstyle="-", color="gray", alpha=0.5))

    # Mark the extremes
    ax.annotate("← TWAP\n(patient)", xy=(is_means[0], is_stds[0]),
                xytext=(-80, 30), textcoords="offset points",
                fontsize=10, color="#2ecc71", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#2ecc71"))

    ax.annotate("Immediate →\n(aggressive)", xy=(is_means[-1], is_stds[-1]),
                xytext=(20, -40), textcoords="offset points",
                fontsize=10, color="#e74c3c", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#e74c3c"))

    cbar = plt.colorbar(scatter, ax=ax, label="log₁₀(λ)")
    ax.set_xlabel("Expected Implementation Shortfall (bps)", fontsize=12)
    ax.set_ylabel("Implementation Shortfall Std Dev (bps)", fontsize=12)
    ax.set_title("Almgren-Chriss Efficient Frontier\n"
                 "Risk-Return Trade-off in Execution",
                 fontsize=14, fontweight="bold")

    plt.tight_layout()
    return save_fig(fig, "efficient_frontier.png", output_dir)


def plot_execution_trajectory(
    child_orders: list,
    title: str = "Execution Trajectory",
    output_dir: str = "reports/figures",
    filename: str = "trajectory.png",
) -> Path:
    """Plot a single execution trajectory showing price path and fills."""
    plt = _get_plt()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[2, 1],
                                    sharex=True)

    timestamps = [c.timestamp for c in child_orders]
    mid_prices = [c.mid_price for c in child_orders]
    exec_prices = [c.exec_price for c in child_orders]
    quantities = [c.quantity for c in child_orders]

    # Top: Price path with execution points
    ax1.plot(timestamps, mid_prices, color="#3498db", linewidth=1, label="Mid price",
             alpha=0.7)
    ax1.scatter(timestamps, exec_prices, c="#e74c3c", s=30, zorder=5,
                label="Execution price", alpha=0.8)

    ax1.set_ylabel("Price ($)", fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)

    # Bottom: Quantity executed per bar
    ax2.bar(timestamps, quantities, width=0.0005, color="#2ecc71", alpha=0.7)
    ax2.set_ylabel("Quantity", fontsize=12)
    ax2.set_xlabel("Time", fontsize=12)

    plt.tight_layout()
    return save_fig(fig, filename, output_dir)


def plot_regime_comparison(
    regime_results: dict,
    metric: str = "is_mean",
    output_dir: str = "reports/figures",
) -> Path:
    """Grouped bar chart comparing strategies across regimes."""
    plt = _get_plt()

    regimes = list(regime_results.keys())
    strategies = list(next(iter(regime_results.values())).keys())

    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(regimes))
    width = 0.8 / len(strategies)
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
              "#1abc9c", "#e67e22", "#34495e"]

    for i, strategy in enumerate(strategies):
        values = []
        for regime in regimes:
            stats = regime_results[regime].get(strategy)
            values.append(getattr(stats, metric, 0) if stats else 0)

        offset = (i - len(strategies) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width * 0.9,
                       label=strategy, color=colors[i % len(colors)], alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([r.replace("_", " ").title() for r in regimes], fontsize=12)
    ax.set_ylabel(f"{metric.replace('_', ' ').title()}", fontsize=12)
    ax.set_title("Strategy Performance Across Volatility Regimes",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    return save_fig(fig, "regime_comparison.png", output_dir)
