# Factor-Augmented Vector Autoregression (FAVAR)

`favar` is a Python package for estimating Factor-Augmented Vector
Autoregression (FAVAR) models in empirical macroeconomic research.

It implements a two-step workflow inspired by Bernanke, Boivin, and Eliasz
(2005): extract latent factors from a large information panel, estimate an
augmented VAR, and compute forecasts and impulse-response functions.

## Installation

```bash
pip install favar
```

## Features

- Principal-component factor extraction from large macroeconomic panels.
- Slow-moving variable adjustment for monetary policy applications.
- Lag-order selection with AIC, BIC, FPE, and HQIC.
- FAVAR summary output with equation tables and residual correlations.
- Forecasts for observed variables with confidence intervals.
- Orthogonalized impulse-response functions.
- Panel-projected impulse responses for selected variables in the information
  panel.
- Residual autocorrelation diagnostics.

## Quick Example

```python
from favar import FAVAR

model = FAVAR(
    X=X,
    Y=Y,
    policy_var="policy_rate",
    k_factors=3,
    slow_columns=slow_columns,
)

results = model.fit(lags=4)

print(results.summary())

forecast = results.forecast(steps=12, confidence_level=0.95)
irf = results.impulse_response(periods=48, impulse_size=0.25)
panel_irf = results.panel_impulse_response(
    periods=48,
    columns=["industrial_production", "inflation"],
    impulse_size=0.25,
)
```

## Documentation

The full documentation, mathematical details, synthetic walkthrough notebook,
examples, and release notes are available in the GitHub repository:

https://github.com/Siqueira-J-S/FAVAR

## Citation

If you use `favar` in academic work, please cite:

```text
Siqueira, J. S. (2026). Factor-Augmented Vector Autoregression (FAVAR)
[Python package]. Zenodo. https://doi.org/10.5281/zenodo.21017290
```

ORCID: [0009-0009-8918-8946](https://orcid.org/0009-0009-8918-8946)

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

## Reference

Bernanke, B. S., Boivin, J., & Eliasz, P. (2005). Measuring the Effects of
Monetary Policy: A Factor-Augmented Vector Autoregressive (FAVAR) Approach.
Quarterly Journal of Economics.

## License

MIT.
