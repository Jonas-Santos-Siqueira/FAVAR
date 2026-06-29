"""Results containers for fitted FAVAR models."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .summary import FAVARSummary


class FAVARResults:
    """Estimation results for a fitted FAVAR."""

    _model_type = "FAVAR"

    def __init__(
        self,
        model,
        var_results,
        var_data: pd.DataFrame,
        order: list[str],
        factor_names: list[str],
        y_names: list[str],
        x_names: list[str],
        slow_columns: list[str],
        policy_var: str,
        factors: np.ndarray,
        principal_components: np.ndarray,
        slow_principal_components: np.ndarray,
        pc_loadings: np.ndarray,
        slow_pc_loadings: np.ndarray,
        explained_variance_ratio: np.ndarray,
        slow_explained_variance_ratio: np.ndarray,
        cleaning_coefficients: np.ndarray,
        policy_cleaning_coefficients: np.ndarray,
        measurement_intercept: np.ndarray,
        measurement_loadings: np.ndarray,
        x_mean: np.ndarray,
        x_std: np.ndarray,
        index,
        standardize: bool,
    ):
        self.model = model
        self.var_results = var_results
        self.varresults_ = var_results
        self._var_results = getattr(var_results, "_results", var_results)
        self.var_data = var_data
        self.var_data_ = var_data
        self.order = order
        self.order_ = order
        self.factor_names = factor_names
        self.factor_names_ = factor_names
        self.y_names = y_names
        self.y_names_ = y_names
        self.x_names = x_names
        self.x_names_ = x_names
        self.slow_columns = slow_columns
        self.slow_columns_ = slow_columns
        self.policy_var = policy_var
        self.policy_var_ = policy_var
        self.factors = factors
        self.factors_ = factors
        self.principal_components = principal_components
        self.principal_components_ = principal_components
        self.slow_principal_components = slow_principal_components
        self.slow_principal_components_ = slow_principal_components
        self.pc_loadings = pc_loadings
        self.pc_loadings_ = pc_loadings
        self.slow_pc_loadings = slow_pc_loadings
        self.slow_pc_loadings_ = slow_pc_loadings
        self.explained_variance_ratio = explained_variance_ratio
        self.explained_variance_ratio_ = explained_variance_ratio
        self.slow_explained_variance_ratio = slow_explained_variance_ratio
        self.slow_explained_variance_ratio_ = slow_explained_variance_ratio
        self.cleaning_coefficients = cleaning_coefficients
        self.cleaning_coefficients_ = cleaning_coefficients
        self.policy_cleaning_coefficients = policy_cleaning_coefficients
        self.policy_cleaning_coefficients_ = policy_cleaning_coefficients
        self.measurement_intercept = measurement_intercept
        self.measurement_intercept_ = measurement_intercept
        self.measurement_loadings = measurement_loadings
        self.measurement_loadings_ = measurement_loadings
        self.loadings_ = measurement_loadings
        self.x_mean = x_mean
        self.x_mean_ = x_mean
        self.x_std = x_std
        self.x_std_ = x_std
        self.index = index
        self.index_ = index
        self.standardize = standardize
        self.standardize_ = standardize
        self.K = len(factor_names)
        self.K_ = self.K
        self.M = len(y_names)
        self.M_ = self.M
        self.N = len(x_names)
        self.N_ = self.N
        self.policy_position = order.index(policy_var)
        self._policy_pos = self.policy_position

    def __getattr__(self, name):
        """Delegate standard VAR result attributes to the fitted VAR object."""
        return getattr(self._var_results, name)

    def __repr__(self):
        return (
            f"<FAVARResults K={self.K} p={self.k_ar} "
            f"M={self.M} N={self.N}>"
        )

    @property
    def names(self):
        return self.order

    @property
    def endog_names(self):
        return self.order

    def summary(self):
        """Return a text summary object."""
        return FAVARSummary(self)

    def _future_index(self, steps: int):
        idx = self.index
        if isinstance(idx, pd.DatetimeIndex) and idx.freq is not None:
            return pd.date_range(idx[-1] + idx.freq, periods=steps, freq=idx.freq)
        if isinstance(idx, pd.PeriodIndex):
            return pd.period_range(idx[-1] + 1, periods=steps)
        return pd.RangeIndex(len(idx), len(idx) + steps)

    def forecast_interval(
        self,
        steps: int,
        alpha: float = 0.05,
        confidence_level: float | None = None,
        include_factors: bool = False,
    ):
        """Forecast the fitted FAVAR system and return DataFrame intervals."""
        if confidence_level is not None:
            alpha = 1.0 - confidence_level
        y_last = self.var_data.to_numpy()[-self.k_ar :]
        mid, lower, upper = self._var_results.forecast_interval(
            y_last, steps, alpha=alpha
        )
        cols = self.order if include_factors else self.y_names
        positions = [self.order.index(col) for col in cols]
        idx = self._future_index(steps)
        return (
            pd.DataFrame(mid[:, positions], columns=cols, index=idx),
            pd.DataFrame(lower[:, positions], columns=cols, index=idx),
            pd.DataFrame(upper[:, positions], columns=cols, index=idx),
        )

    def forecast(
        self,
        steps: int,
        alpha: float = 0.05,
        confidence_level: float | None = None,
        include_factors: bool = False,
    ):
        """Forecast observed Y variables with lower and upper intervals."""
        mid, lower, upper = self.forecast_interval(
            steps,
            alpha=alpha,
            confidence_level=confidence_level,
            include_factors=include_factors,
        )
        out = {}
        for name in mid.columns:
            out[name] = mid[name]
            out[f"{name}_lower"] = lower[name]
            out[f"{name}_upper"] = upper[name]

        ordered_cols = []
        for name in mid.columns:
            ordered_cols.extend([name, f"{name}_lower", f"{name}_upper"])
        return pd.DataFrame(out, index=mid.index)[ordered_cols]

    def _orth_irf_to_shock(
        self,
        periods: int,
        shock: str | None = None,
        impulse_size: float | None = None,
        cumulative: bool = False,
    ) -> np.ndarray:
        if shock is None:
            shock = self.policy_var
        if shock not in self.order:
            raise ValueError(f"Unknown shock variable: {shock}")

        shock_pos = self.order.index(shock)
        irf = self._var_results.irf(periods)
        responses = irf.orth_irfs[:, :, shock_pos].copy()

        if impulse_size is not None:
            impact = responses[0, shock_pos]
            if np.isclose(impact, 0.0):
                raise ValueError("Cannot scale IRF because impact response is zero.")
            responses *= impulse_size / impact

        if cumulative:
            responses = responses.cumsum(axis=0)
        return responses

    def impulse_response(
        self,
        periods: int = 10,
        shock: str | None = None,
        impulse_size: float | None = None,
        cumulative: bool = False,
        include_factors: bool = True,
    ) -> pd.DataFrame:
        """Orthogonalized impulse responses for factors and observed variables."""
        resp = self._orth_irf_to_shock(
            periods,
            shock=shock,
            impulse_size=impulse_size,
            cumulative=cumulative,
        )
        cols = self.order if include_factors else self.y_names
        positions = [self.order.index(col) for col in cols]
        df = pd.DataFrame(resp[:, positions], columns=cols)
        df.index.name = "period"
        return df

    def panel_impulse_response(
        self,
        periods: int = 10,
        shock: str | None = None,
        columns: list[str] | None = None,
        scale: str = "original",
        impulse_size: float | None = None,
        cumulative: bool = False,
    ) -> pd.DataFrame:
        """Project system IRFs back to the information panel X."""
        if scale not in {"original", "std"}:
            raise ValueError("scale must be either 'original' or 'std'.")

        resp = self._orth_irf_to_shock(
            periods,
            shock=shock,
            impulse_size=impulse_size,
            cumulative=cumulative,
        )
        irf_x = resp @ self.measurement_loadings
        if scale == "original":
            irf_x = irf_x * self.x_std

        df = pd.DataFrame(irf_x, columns=self.x_names)
        df.index.name = "period"
        if columns is not None:
            missing = [col for col in columns if col not in df.columns]
            if missing:
                raise ValueError(f"columns contains unknown columns: {missing}")
            df = df.loc[:, columns]
        return df

    def impulse_response_X(self, periods: int = 10, shock: str | None = None, X_cols=None, **kwargs):
        """Compatibility alias for :meth:`panel_impulse_response`."""
        if X_cols is not None and "columns" not in kwargs:
            kwargs["columns"] = X_cols
        return self.panel_impulse_response(periods=periods, shock=shock, **kwargs)

    def is_stable(self, verbose: bool = False) -> bool:
        """Return whether the estimated VAR dynamics are stable."""
        return bool(self._var_results.is_stable(verbose=verbose))

    def acorr(self, nlags: int = 10, resid: bool = True) -> np.ndarray:
        """Autocorrelation matrices for residuals or fitted system data.

        Parameters
        ----------
        nlags : int, default 10
            Number of lags to compute, including lag zero in the returned
            array.
        resid : bool, default True
            If True, use residual autocorrelations. If False, use sample
            autocorrelations of the augmented FAVAR system.
        """
        if resid:
            return self._var_results.resid_acorr(nlags=nlags)
        return self._var_results.sample_acorr(nlags=nlags)

    def plot_acorr(
        self,
        nlags: int = 10,
        resid: bool = True,
        linewidth: float = 5.0,
        figsize: tuple[float, float] | None = None,
    ):
        """Plot autocorrelation matrices with ``2 / sqrt(T)`` bounds.

        The default plot uses residuals of the augmented FAVAR system. The
        diagonal panels are residual autocorrelations, while off-diagonal
        panels are residual cross-correlations at each lag.
        """
        import matplotlib.pyplot as plt

        acorrs = self.acorr(nlags=nlags, resid=resid)[1:]
        names = self.order
        k = len(names)
        x = np.arange(1, nlags + 1)
        bound = 2.0 / np.sqrt(self.nobs)
        if figsize is None:
            figsize = (max(7.0, 3.2 * k), max(6.0, 3.0 * k))

        fig, axes = plt.subplots(k, k, figsize=figsize, squeeze=False)
        for i, row_name in enumerate(names):
            for j, col_name in enumerate(names):
                ax = axes[i, j]
                ax.axhline(0, color="black", linewidth=1.0)
                ax.axhline(bound, color="black", linestyle="--", linewidth=1.0)
                ax.axhline(-bound, color="black", linestyle="--", linewidth=1.0)
                ax.bar(
                    x,
                    acorrs[:, i, j],
                    width=0.35,
                    color="#e24a33",
                    linewidth=linewidth,
                )
                ax.set_ylim(-1.0, 1.0)
                ax.set_xlim(0, nlags + 1)
                ax.set_title(f"{row_name} / {col_name}", fontsize=9)
                if i == k - 1:
                    ax.set_xlabel("lag")
        kind = "residual" if resid else "system"
        fig.suptitle(
            rf"FAVAR {kind} ACF plots with $2 / \sqrt{{T}}$ bounds",
            y=0.995,
        )
        fig.tight_layout()
        return fig
