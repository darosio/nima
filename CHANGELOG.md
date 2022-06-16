# Changelog

## Version 0.5.4 - 2022-06-16
### Added
- [ci] Testpypi and pypi.
- [ci] Adding release workflow.

### Changed
- [docs] Switched to `README.md`.

## Version 0.5.3 - 2022-06-16
### Changed
- [refactor] Renamed to nima (rg, embark export, wgrep, replace-regex).

## Version 0.5.2 - 2022-06-15
### Changed
- [build] Updated dependencies.

## Version 0.5.1 - 2022-06-15
### Changed
- [test] Switched to click.testing.

### Fixed
- [test] Typeguard nima.

## Version 0.5.0 - 2022-06-15
Moved from bitbucket to GITHUB.

### Added
- [ci] Codecov from tests github action.

### Fixed
- [ci] Windows testing.

## Version 0.4.3 - 2022-06-14
### Added
- [build] New linters: flake8-rst-docstrings, pep8-naming, flake8-comprehensions,
flake8_eradicate and  flake8-pytest-style.
- [build] pyupgrade.
- [build] Typeguard.

### Removed
- pytest-cov.

### Changed
- Switched lint to pre-commit.
- Switched to pydata_sphinx_theme.
- Setting for coverage.

## Version 0.4.2 - 2022-06-13
### Added
- [build] pre-commit and pre-commit-hooks ad poetry dev dependencies.
- [build] Switched to isort.
- [build] Nox clean session.

### Removed
- [build] flake8-import-order.
- [build] flake8-black.

## Version 0.4.1 - 2022-06-13
### Changed
- `poetry version …` and use importlib.metadata.version(__name__) when needed.

## Version 0.4.0 - 2022-06-13
### Added
- [doc] Sphinx_click.

### Changed
- Click for the cli.
- Separated `nima` from `bias dark|flat`.

### Removed
- docopt.
- [build] flake8-import-order. Will use isort in pre-commit.

### Fixed
- [build] Remove mypy cache every run.

## Version 0.3.5 - 2022-06-10
### Added
- [Build] mypy checking (yet imperfect).

### Changed
- Some plot graphics have improved.

## Version 0.3.4 - 2022-06-05
### Added
- Markdown for sphinx.

### Changed
- Changelog and authors from rst to md.

### Fixed
- matplotlib version for python-3.10.

## Version 0.3.3
-   Move out of flake8-based linting bandit; use system wide as:
```bandit -r src tests```
-   Move out safety; when updating packages dependencies consider:
```
poetry run safety check
poetry show --outdated
```
-   Update all packages except matplotlib (I will fix its tests).
-   Dropped python-3.7.
-   Added python-3.10.
-   Changed noxfile to use nox_poetry.

## Version 0.3.1
-   Dropping Pyscaffold in favor of poetry.
-   Works with python \< 3.7.

## Version 0.3
-   Transition to Pyscaffold template for faster dev cycles.

## Version 0.2.3
-   Works for clophensor data.
-   Heavy on memory.
-   Flat and Dark not tested.
