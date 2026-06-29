"""Create a public preview image from actual favar model outputs."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from favar import FAVAR
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from favar import FAVAR


def simulate_public_example(
    nobs: int = 180,
    nseries: int = 48,
    k_factors: int = 2,
    n_slow: int = 26,
    seed: int = 2026,
):
    """Generate a small reproducible macro-style panel."""
    rng = np.random.default_rng(seed)
    transition = np.array(
        [
            [0.60, 0.12, -0.04],
            [0.08, 0.52, 0.10],
            [0.18, 0.22, 0.58],
        ]
    )
    shocks = np.array(
        [
            [0.70, 0.00, 0.00],
            [0.20, 0.55, 0.00],
            [-0.10, 0.18, 0.45],
        ]
    )

    burn = 160
    states = np.zeros((nobs + burn, k_factors + 1))
    for t in range(1, len(states)):
        states[t] = transition @ states[t - 1] + shocks @ rng.normal(size=k_factors + 1)
    states = states[burn:]

    factors = states[:, :k_factors]
    policy = states[:, 2]
    factor_loadings = rng.normal(size=(nseries, k_factors))
    policy_loadings = rng.normal(scale=0.5, size=nseries)
    policy_loadings[:n_slow] = 0.0

    panel = (
        factors @ factor_loadings.T
        + np.outer(policy, policy_loadings)
        + rng.normal(scale=0.45, size=(nobs, nseries))
    )
    index = pd.period_range("2010-01", periods=nobs, freq="M")
    X = pd.DataFrame(
        panel,
        index=index,
        columns=[f"indicator_{i + 1:02d}" for i in range(nseries)],
    )
    Y = pd.DataFrame(
        {
            "Activity": 0.85 * factors[:, 0]
            + 0.10 * factors[:, 1]
            + rng.normal(scale=0.15, size=nobs),
            "Inflation": 0.25 * factors[:, 0]
            + 0.65 * factors[:, 1]
            + rng.normal(scale=0.15, size=nobs),
            "Policy Rate": policy + rng.normal(scale=0.08, size=nobs),
        },
        index=index,
    )
    return X, Y, list(X.columns[:n_slow])


def plot_forecast(ax, y: pd.Series, forecast: pd.DataFrame):
    hist = y.iloc[-54:]
    x_hist = np.arange(len(hist))
    x_fc = np.arange(len(hist), len(hist) + len(forecast))

    ax.plot(x_hist, hist.to_numpy(), color="#1f2a44", linewidth=1.9, label="observed")
    ax.plot(x_fc, forecast["Activity"], color="#1f6feb", linewidth=2.0, label="forecast")
    ax.fill_between(
        x_fc,
        forecast["Activity_lower"].to_numpy(),
        forecast["Activity_upper"].to_numpy(),
        color="#1f6feb",
        alpha=0.18,
        linewidth=0.0,
        label="90% interval",
    )
    ax.axvline(len(hist) - 0.5, color="#667085", linewidth=1.0, linestyle="--")
    ax.set_title("Forecast with confidence interval")
    ax.set_xlabel("months")
    ax.set_ylabel("Activity")
    ax.legend(loc="upper left", frameon=False, fontsize=8)


def plot_irf(ax, irf: pd.DataFrame):
    colors = ["#1f6feb", "#b42318", "#2f7d32"]
    for col, color in zip(irf.columns, colors):
        ax.plot(irf.index, irf[col], linewidth=1.9, label=col, color=color)
    ax.axhline(0, color="#344054", linewidth=0.8)
    ax.set_title("Orthogonalized impulse responses")
    ax.set_xlabel("periods after policy shock")
    ax.set_ylabel("response")
    ax.legend(loc="upper right", frameon=False, fontsize=8)


def plot_panel_irf(ax, panel_irf: pd.DataFrame):
    colors = ["#6941c6", "#0e766f", "#c2410c"]
    for col, color in zip(panel_irf.columns, colors):
        ax.plot(panel_irf.index, panel_irf[col], linewidth=1.9, label=col, color=color)
    ax.axhline(0, color="#344054", linewidth=0.8)
    ax.set_title("Panel-projected impulse responses")
    ax.set_xlabel("periods after policy shock")
    ax.set_ylabel("standardized response")
    ax.legend(loc="upper right", frameon=False, fontsize=8)


def plot_order_selection(ax, order_frame: pd.DataFrame, selected_orders: dict[str, int]):
    colors = {
        "AIC": "#1f6feb",
        "BIC": "#b42318",
        "FPE": "#2f7d32",
        "HQIC": "#6941c6",
    }
    normalized = order_frame[["AIC", "BIC", "FPE", "HQIC"]].copy()
    for col in normalized:
        values = normalized[col]
        span = values.max() - values.min()
        normalized[col] = 0.5 if np.isclose(span, 0.0) else (values - values.min()) / span
        ax.plot(
            normalized.index,
            normalized[col],
            marker="o",
            linewidth=1.7,
            markersize=4,
            color=colors[col],
            label=col,
        )
        selected = selected_orders[col.lower()]
        ax.scatter(
            [selected],
            [normalized.loc[selected, col]],
            s=72,
            facecolors="none",
            edgecolors=colors[col],
            linewidths=2.0,
        )
    ax.set_title("Lag-order selection criteria")
    ax.set_xlabel("lag order")
    ax.set_ylabel("normalized criterion")
    ax.set_xticks(order_frame.index)
    ax.legend(loc="upper right", frameon=False, fontsize=8)


def plot_residual_corr(ax, corr: pd.DataFrame):
    image = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title("Residual correlation matrix")
    ax.set_xticks(np.arange(len(corr.columns)), corr.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(corr.index)), corr.index)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(
                j,
                i,
                f"{corr.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if abs(corr.iloc[i, j]) > 0.45 else "#1d2939",
            )
    return image


def plot_residual_acf(fig, outer_cell, results, nlags: int = 10):
    axes = outer_cell.subgridspec(1, 3, wspace=0.28).subplots()
    acorr = results.acorr(nlags=nlags)[1:]
    bound = 2.0 / np.sqrt(results.nobs)
    x = np.arange(1, nlags + 1)
    for ax, name in zip(axes, results.y_names):
        pos = results.order.index(name)
        ax.bar(x, acorr[:, pos, pos], color="#d92d20", width=0.62)
        ax.axhline(0, color="#344054", linewidth=0.8)
        ax.axhline(bound, color="#344054", linestyle="--", linewidth=0.8)
        ax.axhline(-bound, color="#344054", linestyle="--", linewidth=0.8)
        ax.set_ylim(-1.0, 1.0)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("lag")
        ax.grid(axis="y", color="#e4e7ec", linewidth=0.7)
    axes[0].set_ylabel("ACF")
    axes[1].text(
        0.5,
        1.20,
        r"Residual autocorrelation with $2 / \sqrt{T}$ bounds",
        transform=axes[1].transAxes,
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )
    return axes


def create_preview(output_path: Path):
    X, Y, slow_columns = simulate_public_example()
    model = FAVAR(
        X=X,
        Y=Y,
        policy_var="Policy Rate",
        k_factors=2,
        slow_columns=slow_columns,
        standardize=True,
    )
    order_selection = model.select_order(maxlags=6)
    results = model.fit(lags=2)

    forecast = results.forecast(steps=18, confidence_level=0.90)
    irf = results.impulse_response(periods=36, impulse_size=0.25, include_factors=False)
    panel_irf = results.panel_impulse_response(
        periods=30,
        impulse_size=0.25,
        columns=["indicator_01", "indicator_12", "indicator_36"],
        scale="std",
    )
    residual_corr = pd.DataFrame(
        results.resid_corr,
        index=results.order,
        columns=results.order,
    ).loc[results.y_names, results.y_names]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    )
    fig = plt.figure(figsize=(14, 9), dpi=160, facecolor="white")
    grid = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.82], hspace=0.52, wspace=0.28)

    ax_forecast = fig.add_subplot(grid[0, 0])
    ax_irf = fig.add_subplot(grid[0, 1])
    ax_panel = fig.add_subplot(grid[1, 0])
    ax_order = fig.add_subplot(grid[1, 1])
    ax_corr = fig.add_subplot(grid[2, 0])

    plot_forecast(ax_forecast, Y["Activity"], forecast)
    plot_irf(ax_irf, irf)
    plot_panel_irf(ax_panel, panel_irf)
    plot_order_selection(ax_order, order_selection.to_frame(), order_selection.selected_orders)
    image = plot_residual_corr(ax_corr, residual_corr)
    plot_residual_acf(fig, grid[2, 1], results)

    cbar = fig.colorbar(image, ax=ax_corr, fraction=0.046, pad=0.04)
    cbar.set_label("correlation")

    for ax in fig.axes:
        if ax.get_visible():
            ax.grid(color="#e4e7ec", linewidth=0.7)
            for spine in ax.spines.values():
                spine.set_color("#d0d5dd")
                spine.set_linewidth(0.8)

    fig.suptitle(
        "FAVAR outputs from a synthetic macroeconomic panel",
        fontsize=16,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.956,
        "Forecasts, impulse responses, lag-order diagnostics, residual correlations, and residual autocorrelation plots",
        ha="center",
        va="top",
        fontsize=10,
        color="#475467",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    output = Path(__file__).resolve().parents[1] / "docs" / "assets" / "favar-public-preview.png"
    create_preview(output)
    print(output)
