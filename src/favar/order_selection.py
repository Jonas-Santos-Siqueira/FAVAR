"""Lag-order selection results for FAVAR models."""

from __future__ import annotations

import pandas as pd
from statsmodels.iolib.table import SimpleTable


class FAVAROrderSelection:
    """Lag-order selection table for the augmented FAVAR system."""

    def __init__(
        self,
        order_results,
        k_factors: int,
        n_x: int,
        n_slow: int,
        policy_var: str,
        maxlags: int,
    ):
        self.order_results = order_results
        self.ics = order_results.ics
        self.selected_orders = dict(order_results.selected_orders)
        self.aic = self.selected_orders["aic"]
        self.bic = self.selected_orders["bic"]
        self.fpe = self.selected_orders["fpe"]
        self.hqic = self.selected_orders["hqic"]
        self.k_factors = k_factors
        self.n_x = n_x
        self.n_slow = n_slow
        self.policy_var = policy_var
        self.maxlags = maxlags

    def __repr__(self):
        return str(self.summary())

    def __str__(self):
        return (
            "<FAVAROrderSelection: "
            f"AIC={self.aic}, BIC={self.bic}, FPE={self.fpe}, HQIC={self.hqic}>"
        )

    def to_frame(self, mark_min: bool = False) -> pd.DataFrame:
        """Return the information criteria as a DataFrame."""
        columns = ["aic", "bic", "fpe", "hqic"]
        nrows = len(self.ics[columns[0]])
        p_min = self.maxlags - nrows + 1
        index = range(p_min, self.maxlags + 1)
        frame = pd.DataFrame(
            {col.upper(): self.ics[col] for col in columns}, index=index
        )
        if mark_min:
            out = pd.DataFrame(index=frame.index, columns=frame.columns, dtype=object)
            for row in frame.index:
                for col in frame.columns:
                    out.loc[row, col] = f"{frame.loc[row, col]:#.4g}"
            for col, selected in self.selected_orders.items():
                out.loc[selected, col.upper()] = f"{frame.loc[selected, col.upper()]:#.4g}*"
            return out
        return frame

    def summary(self):
        """Return a printable order-selection table."""
        frame = self.to_frame(mark_min=True)
        table = SimpleTable(
            frame.to_numpy(dtype=object),
            headers=list(frame.columns),
            stubs=[str(i) for i in frame.index],
            title="FAVAR Lag Order Selection (* highlights the minimums)",
        )
        return table
