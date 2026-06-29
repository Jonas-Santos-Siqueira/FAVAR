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


def simulate_public_example(
    nobs: int = 220,
    nseries: int = 48,
    k_factors: int = 2,
    n_slow: int = 26,
    seed: int = 2026,
):
    """Generate a reproducible macro-style panel with a policy channel."""
    rng = np.random.default_rng(seed)
    transition = np.array(
        [
            [0.66, 0.10, -0.22],
            [0.06, 0.62, -0.12],
            [0.18, 0.22, 0.56],
        ]
    )
    shocks = np.array(
        [
            [0.62, 0.00, 0.00],
            [0.16, 0.46, 0.00],
            [-0.08, 0.18, 0.42],
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
    policy_loadings = rng.normal(scale=0.45, size=nseries)
    policy_loadings[:n_slow] = 0.0

    panel = (
        factors @ factor_loadings.T
        + np.outer(policy, policy_loadings)
        + rng.normal(scale=0.42, size=(nobs, nseries))
    )
    index = pd.period_range("2006-01", periods=nobs, freq="M")
    X = pd.DataFrame(
        panel,
        index=index,
        columns=[f"indicator_{i + 1:02d}" for i in range(nseries)],
    )
    Y = pd.DataFrame(
        {
            "Activity": 0.90 * factors[:, 0]
            + 0.08 * factors[:, 1]
            + rng.normal(scale=0.14, size=nobs),
            "Inflation": 0.24 * factors[:, 0]
            + 0.70 * factors[:, 1]
            + rng.normal(scale=0.14, size=nobs),
            "Policy rate": policy + rng.normal(scale=0.07, size=nobs),
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
    return model.fit(lags=2)


def direct_label(ax, x, y, label, color, dy=0.0):
    ax.text(
        x,
        y + dy,
        label,
        color=color,
        fontsize=10,
        fontweight="bold",
        va="center",
        ha="left",
    )


def create_preview(output_path: Path):
    results = fit_example()
    observed_irf = results.impulse_response(
        periods=36,
        impulse_size=0.25,
        include_factors=False,
    )
    panel_irf = results.panel_impulse_response(
        periods=36,
        impulse_size=0.25,
        columns=list(results.x_names[:18]),
        scale="std",
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )

    fig = plt.figure(figsize=(11.5, 7.2), dpi=180, facecolor="#f7f2e8")
    ax = fig.add_axes([0.105, 0.18, 0.78, 0.54], facecolor="#f7f2e8")

    for col in panel_irf.columns:
        ax.plot(
            panel_irf.index,
            panel_irf[col],
            color="#9aa0a6",
            linewidth=0.9,
            alpha=0.26,
            zorder=1,
        )

    colors = {
        "Activity": "#1f4e79",
        "Inflation": "#7b3f00",
        "Policy rate": "#c00000",
    }
    for col in ["Activity", "Inflation", "Policy rate"]:
        ax.plot(
            observed_irf.index,
            observed_irf[col],
            color=colors[col],
            linewidth=2.6,
            zorder=3,
        )

    direct_label(ax, 1.35, observed_irf["Policy rate"].iloc[1], "Policy rate", colors["Policy rate"], dy=0.010)
    direct_label(ax, 2.55, observed_irf["Activity"].iloc[2], "Activity", colors["Activity"], dy=-0.012)
    direct_label(ax, 2.15, observed_irf["Inflation"].iloc[2], "Inflation", colors["Inflation"], dy=0.012)

    ax.axhline(0, color="#202020", linewidth=0.9)
    ax.axvline(0, color="#202020", linewidth=0.9)
    ax.set_xlim(0, 39)
    ymin = min(panel_irf.min().min(), observed_irf.min().min()) - 0.018
    ymax = max(panel_irf.max().max(), observed_irf.max().max()) + 0.025
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Months after shock")
    ax.set_ylabel("Response")
    ax.set_title("Estimated responses to a 25-basis-point policy-rate shock", loc="left", pad=12)
    ax.grid(axis="y", color="#ddd5c7", linewidth=0.8)
    ax.grid(axis="x", visible=False)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#202020")
        ax.spines[side].set_linewidth(0.9)

    fig.text(
        0.105,
        0.925,
        "What a FAVAR extracts from a large macro panel",
        fontsize=22,
        fontweight="bold",
        color="#111111",
        ha="left",
    )
    fig.text(
        0.105,
        0.875,
        "A synthetic example: many indicators are compressed into a few factors, then used to trace a monetary-policy shock.",
        fontsize=11.5,
        color="#3f3f3f",
        ha="left",
    )

    flow_y = 0.795
    flow_items = [
        ("48 indicators", "X"),
        ("2 latent factors", "F"),
        ("FAVAR system", "VAR"),
        ("policy responses", "IRF"),
    ]
    x_positions = [0.115, 0.330, 0.525, 0.735]
    for (label, symbol), xpos in zip(flow_items, x_positions):
        fig.text(
            xpos,
            flow_y,
            symbol,
            fontsize=10.5,
            fontweight="bold",
            color="white",
            ha="center",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.34,rounding_size=0.10",
                "facecolor": "#c00000",
                "edgecolor": "#c00000",
            },
        )
        fig.text(
            xpos + 0.033,
            flow_y,
            label,
            fontsize=10.5,
            color="#202020",
            ha="left",
            va="center",
        )
    for xpos in [0.285, 0.480, 0.690]:
        fig.text(xpos, flow_y, "→", fontsize=15, color="#202020", va="center", ha="center")

    fig.text(
        0.105,
        0.092,
        "Note: grey lines show selected panel-projected responses; coloured lines show observed variables in the FAVAR system. Data are synthetic.",
        fontsize=8.8,
        color="#4f4f4f",
        ha="left",
    )
    fig.text(
        0.105,
        0.055,
        "Source: favar Python package, synthetic demonstration",
        fontsize=8.8,
        color="#4f4f4f",
        ha="left",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    output = Path(__file__).resolve().parents[1] / "docs" / "assets" / "favar-public-preview.png"
    create_preview(output)
    print(output)
