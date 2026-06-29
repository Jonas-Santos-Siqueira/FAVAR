"""Factor extraction and input preparation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def as_float_dataframe(data, name: str) -> pd.DataFrame:
    """Return *data* as a numeric DataFrame with unique column names."""
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    if data.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional.")
    if data.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one column.")
    if data.columns.has_duplicates:
        raise ValueError(f"{name} has duplicate column names.")

    out = data.copy()
    try:
        out = out.apply(pd.to_numeric, errors="raise").astype(float)
    except Exception as exc:  # pragma: no cover - pandas exception details vary
        raise ValueError(f"{name} must contain only numeric values.") from exc
    return out


def align_panels(x: pd.DataFrame, y: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align X and Y on the time index and reject missing values."""
    if not x.index.equals(y.index):
        x, y = x.align(y, join="inner", axis=0)
        if len(x) == 0:
            raise ValueError("X and Y do not share any index values.")

    if len(x) != len(y):
        raise ValueError("X and Y must have the same number of observations.")
    if x.isna().any().any():
        raise ValueError("X contains missing values after alignment.")
    if y.isna().any().any():
        raise ValueError("Y contains missing values after alignment.")
    return x, y


def standardize_frame(x: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Standardize columns with population standard deviations."""
    values = x.to_numpy(dtype=float)
    mean = values.mean(axis=0)
    std = values.std(axis=0, ddof=0)
    std[std == 0] = 1.0
    standardized = (values - mean) / std
    return pd.DataFrame(standardized, columns=x.columns, index=x.index), mean, std


def principal_components(z, k_factors: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract principal-component scores using the BBE normalization.

    Parameters
    ----------
    z : array_like
        Standardized information panel with shape ``(nobs, nseries)``.
    k_factors : int
        Number of principal components.

    Returns
    -------
    factors : ndarray
        Principal-component scores, normalized as ``U S / sqrt(N)``.
    loadings : ndarray
        Right singular vectors for the selected components.
    explained_variance_ratio : ndarray
        Share of panel variation captured by each selected component.
    """
    z = np.asarray(z, dtype=float)
    if z.ndim != 2:
        raise ValueError("z must be two-dimensional.")
    nobs, nseries = z.shape
    if k_factors < 1:
        raise ValueError("k_factors must be at least 1.")
    if k_factors > min(nobs, nseries):
        raise ValueError("k_factors cannot exceed min(nobs, nseries).")

    u, s, vt = np.linalg.svd(z, full_matrices=False)
    factors = u[:, :k_factors] * s[:k_factors] / np.sqrt(nseries)
    loadings = vt[:k_factors].T
    total = np.sum(s**2)
    ratio = (s[:k_factors] ** 2) / total if total > 0 else np.zeros(k_factors)
    return factors, loadings, ratio


def validate_slow_columns(
    x_columns: pd.Index, slow_columns: list[str] | None, k_factors: int
) -> list[str]:
    """Validate slow-moving column names and return the effective list."""
    if slow_columns is None:
        slow = list(x_columns)
    else:
        missing = [col for col in slow_columns if col not in x_columns]
        if missing:
            raise ValueError(f"slow_columns contains unknown columns: {missing}")
        slow = list(slow_columns)

    if len(slow) < k_factors:
        raise ValueError("slow_columns must contain at least k_factors columns.")
    return slow
