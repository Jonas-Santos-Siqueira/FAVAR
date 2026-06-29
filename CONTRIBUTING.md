# Contributing

Thank you for your interest in improving `favar`.

This repository is the source of truth for the public PyPI package. Changes
should keep the package installable, tested, and documented for empirical
research users.

## Development Setup

Clone the repository and install the package in editable mode:

```bash
git clone https://github.com/Siqueira-J-S/FAVAR.git
cd FAVAR
python -m pip install -e ".[test]"
```

Run the test suite:

```bash
python -m pytest
```

Run the synthetic example:

```bash
python examples/synthetic_demo.py
```

## Pull Requests

Before opening a pull request:

- Keep public API changes intentional and documented.
- Add or update tests for model behavior, forecasts, impulse responses, or
  diagnostics.
- Update `README.md`, `CHANGELOG.md`, or the notebook when user-facing behavior
  changes.
- Run `python -m pytest` locally.

## Release Checklist

Package releases are published to PyPI by maintainers.

1. Update the version in `pyproject.toml` and `src/favar/__init__.py`.
2. Update `CHANGELOG.md`.
3. Run `python -m pytest`.
4. Build and check the distribution:

```bash
python -m build
python -m twine check dist/*
```

5. Upload with a PyPI API token:

```bash
python -m twine upload dist/*
```

See `RELEASING.md` for the full release workflow.
