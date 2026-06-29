"""Self-contained synthetic FAVAR example."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from favar import FAVAR
except ModuleNotFoundError:  # Allow running from a fresh checkout.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from favar import FAVAR


def simulate_favar(nobs=180, nseries=40, k_factors=2, n_slow=20, seed=123):
    rng = np.random.default_rng(seed)
    state_dim = k_factors + 1

    transition = np.zeros((state_dim, state_dim))
    transition[:k_factors, :k_factors] = 0.35
    transition[:k_factors, :k_factors][np.diag_indices(k_factors)] = 0.55
    transition[:k_factors, k_factors] = 0.12
    transition[k_factors, :k_factors] = 0.25
    transition[k_factors, k_factors] = 0.65
    spectral_radius = np.max(np.abs(np.linalg.eigvals(transition)))
    if spectral_radius >= 0.95:
        transition *= 0.95 / spectral_radius

    shocks = np.tril(rng.normal(size=(state_dim, state_dim)))
    shocks[np.diag_indices(state_dim)] = np.abs(np.diag(shocks)) + 0.5

    burn = 200
    states = np.zeros((nobs + burn, state_dim))
    for t in range(1, len(states)):
        states[t] = transition @ states[t - 1] + shocks @ rng.normal(size=state_dim)
    states = states[burn:]

    factors = states[:, :k_factors]
    policy = states[:, k_factors]
    factor_loadings = rng.normal(size=(nseries, k_factors))
    policy_loadings = rng.normal(size=nseries)
    policy_loadings[:n_slow] = 0.0

    x = (
        factors @ factor_loadings.T
        + np.outer(policy, policy_loadings)
        + rng.normal(scale=0.5, size=(nobs, nseries))
    )
    index = pd.period_range("2000-01", periods=nobs, freq="M")
    X = pd.DataFrame(x, index=index, columns=[f"x{i}" for i in range(nseries)])
    Y = pd.DataFrame(
        {"FFR": policy + rng.normal(scale=0.1, size=nobs)},
        index=index,
    )
    slow_columns = [f"x{i}" for i in range(n_slow)]
    return X, Y, slow_columns


def main():
    X, Y, slow_columns = simulate_favar()
    model = FAVAR(
        X,
        Y,
        policy_var="FFR",
        k_factors=2,
        slow_columns=slow_columns,
        standardize=True,
    )
    results = model.fit(lags=2)

    print(results.summary())
    print("\nForecast")
    print(results.forecast(steps=4, confidence_level=0.95).round(3))
    print("\nPolicy IRF for observed variables")
    print(results.impulse_response(periods=6, include_factors=False).round(3))
    print("\nProjected IRF for first X variables")
    print(results.panel_impulse_response(periods=6, columns=["x0", "x1", "x20"]).round(3))


if __name__ == "__main__":
    main()
