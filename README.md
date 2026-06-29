# Factor-Augmented Vector Autoregression (FAVAR)

[![PyPI version](https://img.shields.io/pypi/v/favar.svg)](https://pypi.org/project/favar/)
[![Python versions](https://img.shields.io/pypi/pyversions/favar.svg)](https://pypi.org/project/favar/)
[![CI](https://github.com/Jonas-Santos-Siqueira/FAVAR/actions/workflows/ci.yml/badge.svg)](https://github.com/Jonas-Santos-Siqueira/FAVAR/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)](https://github.com/Jonas-Santos-Siqueira/FAVAR)

`favar` is a Python package for estimating **Factor-Augmented Vector
Autoregressive** models. It is designed for empirical macroeconomic research,
monetary policy analysis, forecasting, and impulse-response analysis with large
information panels.

The package implements the two-step FAVAR procedure of Bernanke, Boivin, and
Eliasz (2005). It extracts latent factors from a large panel, removes the
contemporaneous policy component from the estimated factors, estimates an
augmented VAR system, and projects impulse responses back to any observable
series in the information panel.

## Contents

- [Installation](#installation)
- [Key Features](#key-features)
- [Quick Start](#quick-start)
- [Model Overview](#model-overview)
- [Data Requirements](#data-requirements)
- [Data Preparation and Transformations](#data-preparation-and-transformations)
- [Slow-Moving and Fast-Moving Variables](#slow-moving-and-fast-moving-variables)
- [Estimation Procedure](#estimation-procedure)
- [Forecasting](#forecasting)
- [Impulse Response Functions](#impulse-response-functions)
- [Residual Autocorrelation Diagnostics](#residual-autocorrelation-diagnostics)
- [Example Summary Output](#example-summary-output)
- [Examples and Notebook](#examples-and-notebook)
- [Public API](#public-api)
- [Project Status and Releases](#project-status-and-releases)
- [Citation](#citation)
- [References](#references)

## Installation

Install the latest release from PyPI:

```bash
pip install favar
```

## Key Features

- Two-step FAVAR estimation following Bernanke, Boivin, and Eliasz (2005).
- Principal-component factor extraction from large information panels.
- Slow-moving variable adjustment for recursive monetary policy identification.
- Forecasts for observed variables with confidence intervals.
- Orthogonalized impulse response functions for the augmented system.
- Panel-projected impulse responses for any selected series in $X_t$.
- Lag-order selection table with AIC, BIC, FPE, and HQIC.
- Residual autocorrelation diagnostics with $2 / \sqrt{T}$ bounds.
- Clean pandas-based interface.

## Quick Start

```python
from favar import FAVAR

model = FAVAR(
    X=X,
    Y=Y,
    policy_var="policy_rate",
    k_factors=3,
    slow_columns=slow_columns,
    standardize=True,
)

order_selection = model.select_order(maxlags=12)
print(order_selection.summary())

results = model.fit(lags=4)
print(results.summary())

forecast = results.forecast(steps=12, confidence_level=0.95)
irf_y = results.impulse_response(periods=48, impulse_size=0.25)
irf_x = results.panel_impulse_response(
    periods=48,
    columns=["ip_growth", "inflation_core"],
    impulse_size=0.25,
)
```

## Model Overview

A FAVAR combines a large information panel $X_t$ with a smaller set of observed
variables $Y_t$ that enter the VAR directly.

- $X_t$ is a large panel of economic indicators with dimension $T \times N$.
- $Y_t$ is a smaller set of observed variables with dimension $T \times M$.
- $F_t$ is a low-dimensional vector of latent factors extracted from $X_t$.
- $R_t$ is the policy instrument, supplied through `policy_var`.

The estimated system is:

$$Z_t = c + A_1 Z_{t-1} + \cdots + A_p Z_{t-p} + u_t.$$

$$X_t = d + \Lambda F_t + \Gamma Y_t + e_t.$$

where:

$$
Z_t =
\begin{bmatrix}
F_t \\
Y_t
\end{bmatrix}.
$$

The policy variable is ordered last in $Y_t$ for recursive identification.

## Data Requirements

Prepare two pandas `DataFrame` objects:

```python
X  # large information panel, shape T x N
Y  # observed VAR variables, shape T x M
```

Both inputs must:

- have a compatible time index;
- be observed at the same frequency, such as monthly or quarterly;
- contain only numeric columns;
- contain no missing values after transformations;
- be aligned over the same sample period;
- include `policy_var` as a column of `Y`.

Example layout:

```text
X
            ip_growth  employment_growth  inflation_core  credit_spread
2000-01         0.31               0.12            0.22           1.21
2000-02         0.28               0.10            0.19           1.18
```

```text
Y
            output_growth  inflation  policy_rate
2000-01             0.31       0.22         5.75
2000-02             0.28       0.19         5.80
```

## Data Preparation and Transformations

The package standardizes `X` internally when `standardize=True`, but it does
not decide the economic transformation of each raw series. Transformations
should be chosen before estimation.

Common transformations:

### Log growth

For real quantities such as production, employment, credit, or monetary
aggregates:

$$
\Delta \log(x_t) = 100 \left[\log(x_t) - \log(x_{t-1})\right].
$$

### Inflation

For price indexes:

$$
\pi_t = 100 \left[\log(P_t) - \log(P_{t-1})\right].
$$

### Interest rates and spreads

Interest rates, spreads, and percentages are often used in levels:

$$
r_t = R_t.
$$

They can also be differenced when the empirical design calls for changes:

$$
\Delta r_t = r_t - r_{t-1}.
$$

### Practical preprocessing checklist

Before fitting the model:

- seasonally adjust series when appropriate;
- apply logs before differencing strictly positive level series;
- avoid mixing levels and growth rates without an economic reason;
- document outlier treatment and sample restrictions;
- call `dropna()` after transformations;
- verify that `X.index.equals(Y.index)` is `True`.

Example:

```python
import numpy as np
import pandas as pd

raw = pd.read_csv("macro_panel.csv", parse_dates=["date"]).set_index("date")

X = pd.DataFrame(index=raw.index)
X["ip_growth"] = 100 * np.log(raw["industrial_production"]).diff()
X["employment_growth"] = 100 * np.log(raw["employment"]).diff()
X["inflation_core"] = 100 * np.log(raw["core_price_index"]).diff()
X["credit_spread"] = raw["credit_spread"]

Y = pd.DataFrame(index=raw.index)
Y["output_growth"] = X["ip_growth"]
Y["inflation"] = X["inflation_core"]
Y["policy_rate"] = raw["policy_rate"]

data = pd.concat([X, Y], axis=1).dropna()
X = data[X.columns]
Y = data[Y.columns]
```

## Slow-Moving and Fast-Moving Variables

For monetary policy applications, the information panel is divided into:

- **slow-moving variables**: variables assumed not to react contemporaneously to
  the policy shock within the period, such as output, employment, consumption,
  and some prices;
- **fast-moving variables**: variables allowed to react within the period, such
  as interest rates, spreads, asset prices, and financial indicators.

Pass the slow-moving columns through `slow_columns`:

```python
slow_columns = [
    "ip_growth",
    "employment_growth",
    "inflation_core",
]
```

If `slow_columns=None`, all columns in `X` are treated as slow-moving. This is
allowed, but explicit classification is recommended for monetary policy work.

## Estimation Procedure

Let $X^s$ denote the standardized information panel:

$$
X^s_{tj} = \frac{X_{tj} - \bar{X}_j}{s_j}.
$$

The implemented two-step estimator proceeds as follows.

### 1. Principal components from the full panel

Estimate $K$ principal components from $X^s$:

$$
\widehat{C}_t = \widehat{C}(F_t, Y_t).
$$

These components estimate the common space spanned by both latent factors and
observed variables.

### 2. Principal components from slow-moving variables

Estimate $K$ principal components from the slow-moving subset of the panel:

$$\widehat{C}^{\star}_t = \widehat{C}^{\star}(F_t).$$

These components are used to isolate the latent factor space from the
contemporaneous policy instrument.

### 3. Remove the contemporaneous policy component

Regress the full-panel principal components on a constant, the policy
instrument, and the slow-moving principal components:

$$\widehat{C}_t = a + b_R R_t + B_S \widehat{C}^{\star}_t + v_t.$$

The cleaned factor estimate is:

$$
\widehat{F}_t = \widehat{C}_t - \widehat{b}_R R_t.
$$

### 4. Estimate the augmented VAR

Stack the cleaned factors and observed variables:

$$
\widehat{Z}_t =
\begin{bmatrix}
\widehat{F}_t \\
Y_t
\end{bmatrix}.
$$

Estimate:

$$\widehat{Z}_t = c + A_1 \widehat{Z}_{t-1} + \cdots + A_p \widehat{Z}_{t-p} + u_t.$$

### 5. Estimate the measurement equation

Estimate the relationship between the standardized information panel and the
augmented state:

$$
X^s_t = d + \Theta \widehat{Z}_t + e_t.
$$

This measurement equation allows responses from the FAVAR system to be mapped
back to each series in $X$:

$$\mathrm{IRF}_{X}(h) = \mathrm{IRF}_{Z}(h)\widehat{\Theta}.$$

When `scale="original"`, projected panel responses are multiplied by the
stored standard deviation of each original `X` column.

## Basic Usage

```python
from favar import FAVAR

model = FAVAR(
    X=X,
    Y=Y,
    policy_var="policy_rate",
    k_factors=3,
    slow_columns=slow_columns,
    standardize=True,
)

results = model.fit(lags=13)
print(results.summary())
```

Main arguments:

- `X`: large information panel.
- `Y`: observed variables included directly in the FAVAR system.
- `policy_var`: policy instrument column in `Y`.
- `k_factors`: number of latent factors.
- `slow_columns`: slow-moving columns from `X`.
- `standardize`: whether to standardize `X` before factor extraction.
- `lags`: fixed lag order for the augmented VAR.

Lag order can also be selected by an information criterion:

```python
order_selection = model.select_order(maxlags=12)
print(order_selection.summary())

results = model.fit(select_order="aic", maxlags=12)
```

Accepted criteria are `"aic"`, `"bic"`, `"hqic"`, and `"fpe"`.

Compact order-selection example:

```text
FAVAR Lag Order Selection (* highlights the minimums)
==================================
    AIC     BIC     FPE      HQIC
----------------------------------
0  -1.262  -1.208   0.2832  -1.240
1 -3.985* -3.769* 0.01859* -3.897*
2  -3.979  -3.601  0.01870  -3.826
3  -3.927  -3.386  0.01971  -3.708
4  -3.881  -3.178  0.02066  -3.596
----------------------------------
```

## Forecasting

Use `forecast()` to forecast the observed variables in `Y`:

```python
forecast = results.forecast(steps=12, confidence_level=0.95)
print(forecast.head())
```

For each variable in `Y`, the output includes:

- point forecast;
- lower confidence bound;
- upper confidence bound.

Example column names:

```text
policy_rate  policy_rate_lower  policy_rate_upper
```

## Impulse Response Functions

Use `impulse_response()` for responses of the augmented FAVAR system.

```python
irf_system = results.impulse_response(
    periods=48,
    shock="policy_rate",
    impulse_size=0.25,
    include_factors=False,
)
print(irf_system.head())
```

`impulse_size=0.25` rescales the shock so that the impact response of the
policy variable is `0.25`. If the policy rate is measured in percentage
points, this corresponds to 25 basis points.

Use `panel_impulse_response()` to project responses back to selected series in
the information panel:

```python
irf_panel = results.panel_impulse_response(
    periods=48,
    shock="policy_rate",
    columns=["ip_growth", "inflation_core", "credit_spread"],
    scale="original",
    impulse_size=0.25,
)
print(irf_panel.head())
```

Use:

- `scale="original"` for projected responses in the transformed units supplied
  by the user;
- `scale="std"` for responses in standardized panel units.

## Residual Autocorrelation Diagnostics

Use `plot_acorr()` to inspect residual autocorrelations and cross-correlations
of the augmented FAVAR system:

```python
fig = results.plot_acorr(nlags=10)
```

The figure contains one panel for each pair of variables in the augmented
system. The dashed bands are $2 / \sqrt{T}$ bounds.

## Example Summary Output

`results.summary()` returns a text summary with overall fit statistics,
equation-by-equation coefficients, residual correlations, and FAVAR-specific
metadata.

Compact example:

```text
  Summary of FAVAR Regression Results
======================================
Model:                           FAVAR
Estimator:                Two-step PCA
VAR method:                        OLS
Date:               Sun, 28, Jun, 2026
Time:                         21:14:27
--------------------------------------------------------------------
No. of Equations:               3    BIC:                  -3.59224
Nobs:                         178    HQIC:                 -3.81539
Log likelihood:        -383.59539    FPE:                   0.01892
AIC:                     -3.96762    Det(Omega_mle):        0.01685
--------------------------------------------------------------------
FAVAR Model Information
====================================================================
No. of factors:                                                    2
No. of X variables:                                               40
No. of observed Y variables:                                       1
No. of slow-moving variables:                                     20
Policy variable:                                                 FFR
Policy position:                                                   3
Lag order:                                                         2
Standardized X:                                                 True
PC variance shares:                                     0.579, 0.248
--------------------------------------------------------------------
Identification: recursive policy shock with the policy variable ordered last.
Results for equation F1
============================================================================
                coefficient       std. error         t-stat          prob
----------------------------------------------------------------------------
const              0.024340         0.040805       0.596496         0.551
L1.F1              0.699803         0.078694       8.892735         0.000
L1.F2              0.055353         0.068085       0.813003         0.416
L1.FFR            -0.001386         0.028840      -0.048049         0.962
...

Correlation matrix of residuals
           F1        F2       FFR
F1   1.000000 -0.266766 -0.145655
F2  -0.266766  1.000000  0.674814
FFR -0.145655  0.674814  1.000000
```

## Examples and Notebook

Run the self-contained script:

```bash
python examples/synthetic_demo.py
```

Open the walkthrough notebook:

```text
notebooks/favar_synthetic_walkthrough.ipynb
```

The notebook demonstrates:

1. package import and installation check;
2. synthetic macroeconomic data generation;
3. preprocessing and transformations;
4. construction of `X`, `Y`, and `slow_columns`;
5. FAVAR estimation;
6. forecasts with confidence intervals;
7. impulse response functions;
8. panel-projected impulse responses.

## Public API

```python
from favar import FAVAR
```

Main methods:

- `FAVAR(...).fit(...)`: estimate the model.
- `results.summary()`: print the estimation summary.
- `model.select_order(maxlags=12)`: compare lag orders by information criteria.
- `results.forecast(steps, confidence_level=0.95)`: forecast observed
  variables.
- `results.impulse_response(periods, impulse_size=None)`: compute system IRFs.
- `results.panel_impulse_response(periods, columns=None)`: compute IRFs
  projected to the information panel.
- `results.plot_acorr(nlags=10)`: plot residual autocorrelations and
  cross-correlations.
- `results.is_stable()`: check dynamic stability of the augmented VAR.

## Final Checklist Before Estimation

- `X` and `Y` have the same time index.
- No missing values remain after transformations.
- All columns in `X` and `Y` are numeric.
- `policy_var` is a column of `Y`.
- Every item in `slow_columns` is a column of `X`.
- `k_factors <= min(T, N)`.
- The lag order is feasible for the available sample size.
- Transformations are economically justified and documented.

## Project Status and Releases

This GitHub repository is the official source repository for the `favar` PyPI
package. Public package releases are available at
[pypi.org/project/favar](https://pypi.org/project/favar/), and release history is
tracked in [CHANGELOG.md](CHANGELOG.md).

Development changes should be proposed through GitHub commits and pull requests.
See [CONTRIBUTING.md](CONTRIBUTING.md) for the local development workflow and
[RELEASING.md](RELEASING.md) for the PyPI release process.

## Citation

If you use `favar` in academic work, please cite:

```text
Siqueira, J. S. (2026). Factor-Augmented Vector Autoregression (FAVAR)
[Python package]. Zenodo. https://doi.org/10.5281/zenodo.21017290
```

Author ORCID: [0009-0009-8918-8946](https://orcid.org/0009-0009-8918-8946)

DOI: [10.5281/zenodo.21017290](https://doi.org/10.5281/zenodo.21017290)

BibTeX:

```bibtex
@software{siqueira_favar_2026,
  author = {Siqueira, J. S.},
  title = {Factor-Augmented Vector Autoregression (FAVAR)},
  year = {2026},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.21017290},
  url = {https://doi.org/10.5281/zenodo.21017290},
  orcid = {0009-0009-8918-8946}
}
```

## References

Bernanke, B. S., Boivin, J., & Eliasz, P. (2005). Measuring the Effects of
Monetary Policy: A Factor-Augmented Vector Autoregressive (FAVAR) Approach.
Quarterly Journal of Economics.

Lutkepohl, H. (2005). New Introduction to Multiple Time Series Analysis.
Springer.

Seabold, S., & Perktold, J. (2010). statsmodels: Econometric and Statistical
Modeling with Python. Proceedings of the 9th Python in Science Conference.

## License

MIT.
