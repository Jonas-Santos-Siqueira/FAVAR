# Releasing `favar` to PyPI

This project uses the modern `pyproject.toml` build flow.

## 1. Validate the package

```bash
python -m pytest
python -m build
python -m twine check dist/*
```

## 2. Create a PyPI API token

1. Sign in at https://pypi.org/.
2. Open account settings.
3. Create an API token.
4. For the first release of a new project, use an account-scoped token.
5. After the first release exists, replace it with a project-scoped token.

Do not commit tokens and do not paste them into notebooks or source files.

## 3. Upload to PyPI

PowerShell:

```powershell
$env:TWINE_USERNAME = "__token__"
$env:TWINE_PASSWORD = "pypi-your-token-here"
python -m twine upload dist/*
```

The package should then be available at:

```text
https://pypi.org/project/favar/
```

Users can install it with:

```bash
pip install favar
```

## 4. Optional TestPyPI dry run

TestPyPI uses a separate account and token:

```powershell
$env:TWINE_USERNAME = "__token__"
$env:TWINE_PASSWORD = "pypi-your-testpypi-token-here"
python -m twine upload --repository testpypi dist/*
```

Then test installation with:

```bash
python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple favar
```
