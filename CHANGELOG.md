# Changelog

All notable changes to this project will be documented in this file.

The project follows semantic versioning once the public API stabilizes. During
the alpha period, minor releases may still include interface adjustments that
are documented here.

## [0.1.0] - 2026-06-28

### Added

- Initial public release on PyPI as `favar`.
- `FAVAR` model class for two-step Factor-Augmented VAR estimation.
- Principal-component factor extraction and slow-moving variable adjustment.
- Lag-order selection with AIC, BIC, FPE, and HQIC.
- FAVAR-style summary output with equation tables and residual correlations.
- Forecasts for observed variables with confidence intervals.
- Orthogonalized impulse-response functions and panel-projected responses.
- Residual autocorrelation diagnostics and plotting helper.
- Synthetic data walkthrough notebook and runnable example script.
- Test suite covering model fitting, summaries, forecasts, IRFs, and diagnostics.
