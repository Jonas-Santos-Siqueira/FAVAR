"""FAVAR model classes."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

from .factor import (
    align_panels,
    as_float_dataframe,
    principal_components,
    standardize_frame,
    validate_slow_columns,
)
from .order_selection import FAVAROrderSelection
from .results import FAVARResults


class FAVAR:
    """Factor-Augmented VAR estimated with the BBE two-step procedure.

    Parameters
    ----------
    X : DataFrame or array_like
        Large information panel with shape ``(nobs, nseries)``.
    Y : DataFrame or array_like
        Observed variables included directly in the VAR. Must contain
        ``policy_var``.
    policy_var : str
        Policy instrument ordered last for recursive identification.
    k_factors : int, default 3
        Number of factors extracted from ``X``.
    slow_columns : sequence[str], optional
        Columns of ``X`` treated as slow-moving. If omitted, all X columns are
        used. Supplying this list is recommended for monetary-policy studies.
    standardize : bool, default True
        Standardize ``X`` before principal components.
    """

    def __init__(
        self,
        X,
        Y,
        policy_var: str,
        k_factors: int = 3,
        slow_columns: list[str] | None = None,
        standardize: bool = True,
    ):
        self.X = as_float_dataframe(X, "X")
        self.Y = as_float_dataframe(Y, "Y")
        self.X, self.Y = align_panels(self.X, self.Y)

        if policy_var not in self.Y.columns:
            raise ValueError("policy_var must be a column of Y.")
        self.policy_var = policy_var
        self.k_factors = int(k_factors)
        self.slow_columns = (
            list(slow_columns) if slow_columns is not None else None
        )
        self.standardize = bool(standardize)
        self._results: FAVARResults | None = None

    def _prepare_system(self):
        """Run the factor steps and build the augmented FAVAR system."""
        X = self.X
        Y = self.Y
        nobs, nseries = X.shape

        if self.standardize:
            Xs, x_mean, x_std = standardize_frame(X)
        else:
            Xs = X.copy()
            x_mean = np.zeros(nseries)
            x_std = np.ones(nseries)

        slow_columns = validate_slow_columns(
            X.columns, self.slow_columns, self.k_factors
        )
        factor_names = [f"F{i + 1}" for i in range(self.k_factors)]

        f0, pc_loadings, pc_var_ratio = principal_components(
            Xs.to_numpy(), self.k_factors
        )
        fslow, slow_pc_loadings, slow_pc_var_ratio = principal_components(
            Xs.loc[:, slow_columns].to_numpy(), self.k_factors
        )

        policy = Y[self.policy_var].to_numpy(dtype=float)
        cleaning_design = np.column_stack([np.ones(nobs), policy, fslow])
        cleaning_coef, *_ = np.linalg.lstsq(cleaning_design, f0, rcond=None)
        policy_cleaning_coef = cleaning_coef[1, :]
        factors = f0 - np.outer(policy, policy_cleaning_coef)

        y_order = [col for col in Y.columns if col != self.policy_var]
        y_order.append(self.policy_var)
        order = factor_names + y_order
        y_ordered = Y.loc[:, y_order]
        var_data = pd.DataFrame(
            np.column_stack([factors, y_ordered.to_numpy(dtype=float)]),
            columns=order,
            index=X.index,
        )

        measurement_design = np.column_stack(
            [np.ones(nobs), factors, y_ordered.to_numpy(dtype=float)]
        )
        measurement_coef, *_ = np.linalg.lstsq(
            measurement_design, Xs.to_numpy(dtype=float), rcond=None
        )

        return {
            "var_data": var_data,
            "order": order,
            "factor_names": factor_names,
            "y_names": y_order,
            "x_names": list(X.columns),
            "slow_columns": slow_columns,
            "factors": factors,
            "principal_components": f0,
            "slow_principal_components": fslow,
            "pc_loadings": pc_loadings,
            "slow_pc_loadings": slow_pc_loadings,
            "explained_variance_ratio": pc_var_ratio,
            "slow_explained_variance_ratio": slow_pc_var_ratio,
            "cleaning_coefficients": cleaning_coef,
            "policy_cleaning_coefficients": policy_cleaning_coef,
            "measurement_intercept": measurement_coef[0, :],
            "measurement_loadings": measurement_coef[1:, :],
            "x_mean": x_mean,
            "x_std": x_std,
        }

    def select_order(self, maxlags: int = 12, trend: str = "c"):
        """Select the lag order of the augmented FAVAR system.

        Returns
        -------
        FAVAROrderSelection
            Object with selected orders, a DataFrame representation, and a
            printable summary table.
        """
        prepared = self._prepare_system()
        var_model = VAR(prepared["var_data"])
        order_results = var_model.select_order(maxlags=maxlags, trend=trend)
        return FAVAROrderSelection(
            order_results,
            k_factors=self.k_factors,
            n_x=len(prepared["x_names"]),
            n_slow=len(prepared["slow_columns"]),
            policy_var=self.policy_var,
            maxlags=maxlags,
        )

    def fit(
        self,
        lags: int | None = 13,
        select_order: str | None = None,
        maxlags: int | None = None,
        trend: str = "c",
        verbose: bool = False,
    ):
        """Estimate the FAVAR.

        Parameters
        ----------
        lags : int, default 13
            Fixed VAR lag order. Ignored when ``select_order`` is provided.
        select_order : {"aic", "bic", "hqic", "fpe"}, optional
            Information criterion used to select the VAR lag order.
        maxlags : int, optional
            Maximum lag considered when ``select_order`` is used. If omitted,
            ``lags`` is used.
        trend : {"c", "ct", "ctt", "n"}, default "c"
            Deterministic terms in the VAR step.
        verbose : bool, default False
            Print lag-selection details when available.
        """
        prepared = self._prepare_system()
        var_model = VAR(prepared["var_data"])
        if select_order is None:
            var_results = var_model.fit(
                maxlags=lags, ic=None, trend=trend, verbose=verbose
            )
        else:
            lag_cap = maxlags if maxlags is not None else lags
            var_results = var_model.fit(
                maxlags=lag_cap, ic=select_order, trend=trend, verbose=verbose
            )

        self._results = FAVARResults(
            model=self,
            var_results=var_results,
            var_data=prepared["var_data"],
            order=prepared["order"],
            factor_names=prepared["factor_names"],
            y_names=prepared["y_names"],
            x_names=prepared["x_names"],
            slow_columns=prepared["slow_columns"],
            policy_var=self.policy_var,
            factors=prepared["factors"],
            principal_components=prepared["principal_components"],
            slow_principal_components=prepared["slow_principal_components"],
            pc_loadings=prepared["pc_loadings"],
            slow_pc_loadings=prepared["slow_pc_loadings"],
            explained_variance_ratio=prepared["explained_variance_ratio"],
            slow_explained_variance_ratio=prepared["slow_explained_variance_ratio"],
            cleaning_coefficients=prepared["cleaning_coefficients"],
            policy_cleaning_coefficients=prepared["policy_cleaning_coefficients"],
            measurement_intercept=prepared["measurement_intercept"],
            measurement_loadings=prepared["measurement_loadings"],
            x_mean=prepared["x_mean"],
            x_std=prepared["x_std"],
            index=self.X.index,
            standardize=self.standardize,
        )
        return self._results

    def _check_results(self) -> FAVARResults:
        if self._results is None:
            raise RuntimeError("Call fit() before accessing results.")
        return self._results

    def summary(self):
        return self._check_results().summary()

    def forecast(self, steps: int, alpha: float = 0.05, confidence_level=None):
        return self._check_results().forecast(
            steps, alpha=alpha, confidence_level=confidence_level
        )

    def impulse_response(self, periods: int = 10, shock: str | None = None, **kwargs):
        return self._check_results().impulse_response(periods, shock=shock, **kwargs)

    def panel_impulse_response(
        self, periods: int = 10, shock: str | None = None, **kwargs
    ):
        return self._check_results().panel_impulse_response(
            periods, shock=shock, **kwargs
        )
