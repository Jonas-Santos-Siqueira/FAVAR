"""Create an editorial-style preview image from actual favar outputs."""

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


PAPER = "#f4f1e8"
INK = "#202020"
MUTED = "#6f6b62"
GRID = "#d8d1c3"
RED = "#e62b1e"
RED_LIGHT = "#f7aaa2"
GREY = "#b7b5ac"
BLUE_GREY = "#4f6372"


def simulate_public_example(
    nobs: int = 228,
    nseries: int = 48,
    k_factors: int = 2,
    n_slow: int = 26,
    seed: int = 2026,
):
    """Generate a reproducible macro-style panel with a visible policy channel."""
    rng = np.random.default_rng(seed)
    transition = np.array(
        [
            [0.70, 0.08, -0.18],
            [0.08, 0.64, -0.10],
            [0.14, 0.20, 0.58],
        ]
    )
    shocks = np.array(
        [
            [0.56, 0.00, 0.00],
            [0.12, 0.44, 0.00],
            [-0.06, 0.16, 0.36],
        ]
    )

    burn = 220
    states = np.zeros((nobs + burn, k_factors + 1))
    for t in range(1, len(states)):
        states[t] = transition @ states[t - 1] + shocks @ rng.normal(size=k_factors + 1)
    states = states[burn:]

    factors = states[:, :k_factors]
    policy = states[:, 2]
    factor_loadings = rng.normal(size=(nseries, k_factors))
    policy_loadings = rng.normal(scale=0.40, size=nseries)
    policy_loadings[:n_slow] = 0.0

    panel = (
        factors @ factor_loadings.T
        + np.outer(policy, policy_loadings)
        + rng.normal(scale=0.40, size=(nobs, nseries))
    )
    index = pd.period_range("2006-01", periods=nobs, freq="M")
    X = pd.DataFrame(
        panel,
        index=index,
        columns=[f"indicator_{i + 1:02d}" for i in range(nseries)],
    )
    Y = pd.DataFrame(
        {
            "Activity": 0.94 * factors[:, 0]
            + 0.08 * factors[:, 1]
            + rng.normal(scale=0.13, size=nobs),
            "Inflation": 0.20 * factors[:, 0]
            + 0.74 * factors[:, 1]
            + rng.normal(scale=0.13, size=nobs),
            "Policy rate": policy + rng.normal(scale=0.06, size=nobs),
        },
        index=index,
    )
    return X, Y, list(X.columns[:n_slow])


def fit_example():
    X, Y, slow_columns = simulate_public_example()
    model = FAVAR(
        X=X,
        Y=Y,
        policy_var="Policy rate",
        k_factors=2,
        slow_columns=slow_columns,
        standardize=True,
    )
    return Y, model.fit(lags=2)


def style_axes(ax):
    ax.set_facecolor(PAPER)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="both", colors=INK, labelsize=8.5, length=3, width=0.9)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(INK)
        ax.spines[side].set_linewidth(0.9)
    ax.title.set_fontweight("bold")
    ax.title.set_color(INK)


def add_panel_title(ax, title: str, subtitle: str | None = None):
    ax.set_title(title, loc="left", fontsize=13.5, pad=10)
    if subtitle:
        ax.text(
            0.0,
            1.005,
            subtitle,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            color=MUTED,
            fontsize=8.8,
        )


def plot_forecast(ax, y: pd.DataFrame, results, steps: int = 18):
    forecast = results.forecast(steps=steps, confidence_level=0.95)
    series = "Activity"
    history = y[series].tail(42).to_numpy()
    mean = forecast[series].to_numpy()
    lower = forecast[f"{series}_lower"].to_numpy()
    upper = forecast[f"{series}_upper"].to_numpy()

    hist_x = np.arange(len(history))
    future_x = np.arange(len(history), len(history) + steps)
    bridge_x = np.r_[hist_x[-1], future_x]
    bridge_mean = np.r_[history[-1], mean]
    bridge_lower = np.r_[history[-1], lower]
    bridge_upper = np.r_[history[-1], upper]

    ax.plot(hist_x, history, color=INK, linewidth=2.0, zorder=3)
    ax.fill_between(
        bridge_x,
        bridge_lower,
        bridge_upper,
        color=RED_LIGHT,
        alpha=0.55,
        linewidth=0,
        zorder=1,
    )
    ax.plot(bridge_x, bridge_mean, color=RED, linewidth=2.7, zorder=4)
    ax.axvline(hist_x[-1], color=GRID, linewidth=1.0)

    ypad = (max(bridge_upper.max(), history.max()) - min(bridge_lower.min(), history.min())) * 0.14
    ax.set_ylim(min(bridge_lower.min(), history.min()) - ypad, max(bridge_upper.max(), history.max()) + ypad)
    ax.set_xlim(0, future_x[-1] + 1)
    ax.set_xticks([0, 20, 40, 59])
    ax.set_xticklabels(["t-41", "t-21", "t-1", "t+18"])
    ax.set_ylabel("Index")

    ax.text(hist_x[7], history[7] + 0.18, "observed", color=INK, fontsize=9, fontweight="bold")
    ax.text(future_x[-4], mean[-4] + 0.13, "forecast", color=RED, fontsize=9, fontweight="bold")
    ax.text(future_x[5], upper[5] + 0.08, "95% interval", color="#9a413b", fontsize=8.5, fontweight="bold")


def plot_irf(ax, results, periods: int = 24):
    irf = results.impulse_response(
        periods=periods,
        impulse_size=0.25,
        include_factors=False,
    )
    x = irf.index.to_numpy()

    ax.axhline(0, color=INK, linewidth=0.9)
    ax.plot(x, irf["Inflation"], color=GREY, linewidth=2.0, zorder=2)
    ax.plot(x, irf["Policy rate"], color=BLUE_GREY, linewidth=2.2, zorder=3)
    ax.plot(x, irf["Activity"], color=RED, linewidth=2.8, zorder=4)

    ymin = min(irf.min().min(), -0.05)
    ymax = max(irf.max().max(), 0.28)
    ax.set_ylim(ymin - 0.03, ymax + 0.04)
    ax.set_xlim(0, periods)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_xlabel("Periods after shock")
    ax.set_ylabel("Response")

    ax.text(1.0, irf["Policy rate"].iloc[1] + 0.025, "Policy rate", color=BLUE_GREY, fontsize=9, fontweight="bold")
    ax.text(4.0, irf["Activity"].iloc[4] - 0.028, "Activity", color=RED, fontsize=9, fontweight="bold")
    ax.text(6.0, irf["Inflation"].iloc[6] + 0.020, "Inflation", color=MUTED, fontsize=8.5, fontweight="bold")


def plot_acorr(ax, results, nlags: int = 12):
    acorr = results.acorr(nlags=nlags, resid=True)
    pos = results.order.index("Activity")
    vals = acorr[1:, pos, pos]
    lags = np.arange(1, nlags + 1)
    bound = 2.0 / np.sqrt(results.nobs)
    colors = np.where(np.abs(vals) > bound, RED, "#333333")

    ax.axhline(0, color=INK, linewidth=0.9)
    ax.axhline(bound, color=INK, linewidth=0.9, linestyle=(0, (4, 3)))
    ax.axhline(-bound, color=INK, linewidth=0.9, linestyle=(0, (4, 3)))
    ax.bar(lags, vals, width=0.55, color=colors, edgecolor=colors, zorder=3)

    ax.set_ylim(-0.34, 0.34)
    ax.set_xlim(0.25, nlags + 0.75)
    ax.set_xticks([1, 4, 8, 12])
    ax.set_xlabel("Lag")
    ax.set_ylabel("Correlation")
    ax.text(8.15, bound + 0.025, r"$2/\sqrt{T}$ bounds", color=INK, fontsize=8.4, fontweight="bold")
    ax.text(1.1, -0.285, "Activity residuals", color=MUTED, fontsize=8.8, fontweight="bold")


def create_preview(output_path: Path):
    y, results = fit_example()

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 13,
            "axes.labelsize": 8.8,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "figure.facecolor": PAPER,
            "savefig.facecolor": PAPER,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 5.2), dpi=180, facecolor=PAPER)
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.18, top=0.72, wspace=0.28)

    for ax in axes:
        style_axes(ax)

    add_panel_title(axes[0], "Forecast, h steps ahead", "Activity with a 95% prediction interval")
    add_panel_title(axes[1], "Impulse response", "Policy-rate shock, orthogonalized response")
    add_panel_title(axes[2], "Residual autocorrelation", r"Dashed lines are $2/\sqrt{T}$ bounds")

    plot_forecast(axes[0], y, results)
    plot_irf(axes[1], results)
    plot_acorr(axes[2], results)

    fig.text(
        0.055,
        0.91,
        "Factor-Augmented Vector Autoregression (FAVAR)",
        fontsize=22,
        fontweight="bold",
        color=INK,
        ha="left",
    )
    fig.text(
        0.055,
        0.855,
        "Three model outputs researchers commonly inspect: forecast uncertainty, dynamic responses and residual diagnostics.",
        fontsize=11.2,
        color="#3f3f3f",
        ha="left",
    )
    fig.text(
        0.055,
        0.075,
        "Source: favar Python package; synthetic macroeconomic panel.",
        fontsize=8.5,
        color=MUTED,
        ha="left",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    output = Path(__file__).resolve().parents[1] / "docs" / "assets" / "favar-public-preview.png"
    create_preview(output)
    print(output)
