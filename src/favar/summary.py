"""Text summaries for FAVAR results."""

from __future__ import annotations

import time
from io import StringIO

import numpy as np
import pandas as pd


def _fmt(value, precision=6):
    if isinstance(value, (str, bool)):
        return str(value)
    if isinstance(value, (int, np.integer)):
        return str(value)
    try:
        value = float(value)
    except Exception:
        return str(value)
    if np.isnan(value):
        return "nan"
    if abs(value) != 0 and (abs(value) >= 1e5 or abs(value) < 1e-4):
        return f"{value:.5e}"
    return f"{value:.{precision}f}"


def _fmt_prob(value):
    try:
        value = float(value)
    except Exception:
        return str(value)
    if np.isnan(value):
        return "nan"
    return f"{value:.3f}"


class FAVARSummary:
    """Console summary for a fitted FAVAR model."""

    def __init__(self, results):
        self.results = results
        self.summary = self.make()

    def __repr__(self):
        return self.summary

    def __str__(self):
        return self.summary

    def make(self):
        buf = StringIO()
        buf.write(self._header())
        buf.write("\n")
        buf.write(self._system_stats())
        buf.write("\n")
        buf.write(self._favar_info())
        buf.write("\n")
        buf.write(self._coef_tables())
        buf.write("\n")
        buf.write(self._residual_corr())
        return buf.getvalue()

    def _header(self):
        now = time.localtime()
        rows = [
            ("Model:", "FAVAR"),
            ("Estimator:", "Two-step PCA"),
            ("VAR method:", "OLS"),
            ("Date:", time.strftime("%a, %d, %b, %Y", now)),
            ("Time:", time.strftime("%H:%M:%S", now)),
        ]
        width = 38
        lines = ["  Summary of FAVAR Regression Results   ", "=" * width]
        lines.extend(f"{stub:<14}{value:>24}" for stub, value in rows)
        return "\n".join(lines)

    def _system_stats(self):
        r = self.results
        pairs = [
            ("No. of Equations:", r.neqs, "BIC:", r.bic),
            ("Nobs:", r.nobs, "HQIC:", r.hqic),
            ("Log likelihood:", r.llf, "FPE:", r.fpe),
            ("AIC:", r.aic, "Det(Omega_mle):", r.detomega),
        ]
        lines = ["-" * 68]
        for left_stub, left_value, right_stub, right_value in pairs:
            lines.append(
                f"{left_stub:<21}{_fmt(left_value, 5):>12}    "
                f"{right_stub:<16}{_fmt(right_value, 5):>14}"
            )
        lines.append("-" * 68)
        return "\n".join(lines)

    def _favar_info(self):
        r = self.results
        explained = ", ".join(_fmt(x, 3) for x in r.explained_variance_ratio)
        rows = [
            ("No. of factors:", r.K),
            ("No. of X variables:", r.N),
            ("No. of observed Y variables:", r.M),
            ("No. of slow-moving variables:", len(r.slow_columns)),
            ("Policy variable:", r.policy_var),
            ("Policy position:", r.policy_position + 1),
            ("Lag order:", r.k_ar),
            ("Standardized X:", r.standardize),
            ("PC variance shares:", explained),
        ]
        lines = ["FAVAR Model Information", "=" * 68]
        lines.extend(f"{stub:<31}{_fmt(value, 5):>37}" for stub, value in rows)
        lines.append("-" * 68)
        lines.append(
            "Identification: recursive policy shock with the policy variable ordered last."
        )
        return "\n".join(lines)

    def _coef_tables(self):
        r = self.results
        var = r._var_results
        params = np.asarray(var.params)
        stderr = np.asarray(var.stderr)
        tvalues = np.asarray(var.tvalues)
        pvalues = np.asarray(var.pvalues)
        exog_names = list(var.exog_names)
        name_width = max(12, max(len(name) for name in exog_names) + 2)
        width = name_width + 64

        chunks = []
        for i, eq_name in enumerate(r.order):
            lines = [f"Results for equation {eq_name}", "=" * width]
            lines.append(
                f"{'':<{name_width}}"
                f"{'coefficient':>15}"
                f"{'std. error':>17}"
                f"{'t-stat':>15}"
                f"{'prob':>14}"
            )
            lines.append("-" * width)
            for row_name, coef, se, tval, pval in zip(
                exog_names, params[:, i], stderr[:, i], tvalues[:, i], pvalues[:, i]
            ):
                lines.append(
                    f"{row_name:<{name_width}}"
                    f"{_fmt(coef):>15}"
                    f"{_fmt(se):>17}"
                    f"{_fmt(tval):>15}"
                    f"{_fmt_prob(pval):>14}"
                )
            lines.append("=" * width)
            chunks.append("\n".join(lines))
        return "\n\n".join(chunks)

    def _residual_corr(self):
        r = self.results
        corr = pd.DataFrame(r.resid_corr, index=r.order, columns=r.order)
        return (
            "Correlation matrix of residuals\n"
            + corr.to_string(float_format=lambda value: f"{value: .6f}")
            + "\n"
        )
