import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, List, Dict


def plot_cumulative_returns(
    returns: pd.DataFrame,
    title: str = "Cumulative Returns",
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """Plot cumulative returns for multiple assets"""
    fig, ax = plt.subplots(figsize=figsize)
    
    cumulative = (1 + returns).cumprod() - 1
    
    for col in cumulative.columns:
        ax.plot(cumulative.index, cumulative[col], label=col, linewidth=2)
    
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_drawdown(
    returns: pd.Series,
    title: str = "Drawdown",
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """Plot underwater drawdown chart"""
    fig, ax = plt.subplots(figsize=figsize)
    
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    
    ax.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color="red")
    ax.plot(drawdown.index, drawdown, color="red", linewidth=2)
    
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)
    
    max_dd = drawdown.min()
    ax.text(
        0.02, 0.98,
        f"Max Drawdown: {max_dd:.2%}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    title: str = "Correlation Matrix",
    figsize: tuple = (10, 8),
) -> plt.Figure:
    """Plot correlation heatmap"""
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    
    plt.tight_layout()
    return fig


def plot_efficient_frontier(
    frontier_points: List[Dict],
    current_portfolio: Optional[Dict] = None,
    title: str = "Efficient Frontier",
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """Plot efficient frontier"""
    fig, ax = plt.subplots(figsize=figsize)
    
    volatilities = [p["volatility"] for p in frontier_points]
    returns = [p["return"] for p in frontier_points]
    sharpes = [p["sharpe_ratio"] for p in frontier_points]
    
    scatter = ax.scatter(
        volatilities,
        returns,
        c=sharpes,
        cmap="RdYlGn",
        s=50,
        alpha=0.6,
        edgecolors="black",
        linewidth=0.5,
    )
    
    if current_portfolio is not None:
        ax.scatter(
            current_portfolio["volatility"],
            current_portfolio["return"],
            color="red",
            s=200,
            marker="*",
            edgecolors="black",
            linewidth=2,
            label="Current Portfolio",
            zorder=5,
        )
    
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Volatility (Risk)")
    ax.set_ylabel("Expected Return")
    ax.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Sharpe Ratio")
    
    if current_portfolio is not None:
        ax.legend()
    
    plt.tight_layout()
    return fig


def plot_monte_carlo_fan(
    price_paths: np.ndarray,
    percentiles: List[int] = [5, 25, 50, 75, 95],
    title: str = "Monte Carlo Simulation",
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """Plot Monte Carlo fan chart with percentile bands"""
    fig, ax = plt.subplots(figsize=figsize)
    
    time_steps = np.arange(price_paths.shape[0])
    
    colors = ["#d62728", "#ff7f0e", "#2ca02c", "#ff7f0e", "#d62728"]
    alphas = [0.2, 0.3, 0.5, 0.3, 0.2]
    
    for i, p in enumerate(percentiles):
        percentile_path = np.percentile(price_paths, p, axis=1)
        ax.plot(
            time_steps,
            percentile_path,
            label=f"{p}th percentile",
            linewidth=2,
            alpha=0.8,
            color=colors[i],
        )
    
    for i in range(len(percentiles) - 1):
        lower = np.percentile(price_paths, percentiles[i], axis=1)
        upper = np.percentile(price_paths, percentiles[i + 1], axis=1)
        ax.fill_between(
            time_steps,
            lower,
            upper,
            alpha=alphas[i],
            color=colors[i],
        )
    
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Days")
    ax.set_ylabel("Portfolio Value")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
