Release and Upload Instructions

1. Prepare environment

```bash
python -m pip install --upgrade pip build twine
```

2. Run tests locally

```bash
uv run pytest -q
```

3. Build distributions

```bash
python -m build
```

4. Upload to Test PyPI (verify account and API token)

```bash
twine upload --repository testpypi dist/*
```

5. Install from Test PyPI to verify

```bash
pip install -i https://test.pypi.org/simple/ multiple-minimum-monte-carlo
```

6. Upload to PyPI

```bash
twine upload dist/*
```

Notes
- Update `pyproject.toml` version before building a new release.
- Consider using GitHub Actions to publish automatically on semantic version tags.
