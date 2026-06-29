from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from favar import FAVAR


def simulate_favar(nobs=120, nseries=30, k_factors=2, n_slow=15, seed=1234):
    rng = np.random.default_rng(seed)
    state_dim = k_factors + 1

    transition = np.zeros((state_dim, state_dim))
    transition[:k_factors, :k_factors] = 0.2
    transition[:k_factors, :k_factors][np.diag_indices(k_factors)] = 0.5
    transition[:k_factors, k_factors] = 0.1
    transition[k_factors, :k_factors] = 0.3
    transition[k_factors, k_factors] = 0.6
    spectral_radius = np.max(np.abs(np.linalg.eigvals(transition)))
    if spectral_radius >= 0.95:
        transition *= 0.95 / spectral_radius

    shocks = np.tril(rng.normal(size=(state_dim, state_dim)))
    shocks[np.diag_indices(state_dim)] = np.abs(np.diag(shocks)) + 0.5

    burn = 100
    states = np.zeros((nobs + burn, state_dim))
    for t in range(1, nobs + burn):
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
        + rng.normal(scale=0.4, size=(nobs, nseries))
    )
    y = np.column_stack([factors[:, 0], policy])

    index = pd.period_range("2010-01", periods=nobs, freq="M")
    X = pd.DataFrame(x, index=index, columns=[f"x{i}" for i in range(nseries)])
    Y = pd.DataFrame(y, index=index, columns=["Activity", "R"])
    slow_columns = [f"x{i}" for i in range(n_slow)]
    return X, Y, slow_columns


def test_favar_fit_summary_forecast_and_irf():
    X, Y, slow_columns = simulate_favar()
    results = FAVAR(
        X,
        Y,
        policy_var="R",
        k_factors=2,
        slow_columns=slow_columns,
    ).fit(lags=2)

    assert results.order[-1] == "R"
    assert results.K == 2
    assert results.N == X.shape[1]
    assert results.is_stable()

    summary = str(results.summary())
    assert "Summary of FAVAR Regression Results" in summary
    assert "Model:" in summary
    assert "FAVAR" in summary
    assert "FAVAR Model Information" in summary
    assert "Results for equation F1" in summary
    assert "Correlation matrix of residuals" in summary

    forecast = results.forecast(steps=3)
    assert forecast.shape == (3, 6)
    assert list(forecast.columns) == [
        "Activity",
        "Activity_lower",
        "Activity_upper",
        "R",
        "R_lower",
        "R_upper",
    ]

    irf_y = results.impulse_response(periods=4, include_factors=False)
    assert irf_y.shape == (5, 2)
    assert list(irf_y.columns) == ["Activity", "R"]

    irf_x = results.panel_impulse_response(
        periods=4, columns=["x0", "x20"], scale="std"
    )
    assert irf_x.shape == (5, 2)
    assert list(irf_x.columns) == ["x0", "x20"]


def test_order_selection_api():
    X, Y, slow_columns = simulate_favar()
    model = FAVAR(
        X,
        Y,
        policy_var="R",
        k_factors=2,
        slow_columns=slow_columns,
    )
    order_selection = model.select_order(maxlags=2)
    order_table = str(order_selection.summary())

    assert "FAVAR Lag Order Selection" in order_table
    assert "*" in order_table
    assert set(order_selection.to_frame().columns) == {"AIC", "BIC", "FPE", "HQIC"}

    results = model.fit(select_order="aic", maxlags=2)

    assert results.order[-1] == "R"
    assert results.forecast(steps=1).shape == (1, 6)


def test_plot_acorr_returns_figure():
    pytest.importorskip("matplotlib")
    X, Y, slow_columns = simulate_favar()
    results = FAVAR(
        X,
        Y,
        policy_var="R",
        k_factors=2,
        slow_columns=slow_columns,
    ).fit(lags=2)

    fig = results.plot_acorr(nlags=4)
    assert len(fig.axes) == len(results.order) ** 2


def test_validation_rejects_unknown_slow_columns():
    X, Y, slow_columns = simulate_favar()
    with pytest.raises(ValueError, match="unknown columns"):
        FAVAR(
            X,
            Y,
            policy_var="R",
            k_factors=2,
            slow_columns=slow_columns + ["missing"],
        ).fit(lags=2)


def test_irf_scaling_targets_policy_impact_response():
    X, Y, slow_columns = simulate_favar()
    results = FAVAR(
        X,
        Y,
        policy_var="R",
        k_factors=2,
        slow_columns=slow_columns,
    ).fit(lags=2)

    irf = results.impulse_response(periods=2, impulse_size=0.25)
    assert np.isclose(irf.loc[0, "R"], 0.25)
