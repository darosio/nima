# Changelog

## Version 0.5.6 - 2022-06-18
### What's Changed
- Bump nox-poetry from 0.9.0 to 1.0.0 in /.github/workflows by @dependabot in #7
- fixtestpypi by @darosio in #20

For some reason the automatic triggering of actions stopped working.

## Version 0.5.5 - 2022-06-17
### Changes
- bump v0.5.5 (#18) @darosio

### construction_worker Continuous Integration
- fix: pre-commit in noxfile (#17) @darosio

### books Documentation
- update sphinx (#16) @darosio

### Dependencies
- Bump sphinx to 5.0.2 (#15) @darosio
- Bump myst-parser 0.18.0 (#15) @darosio
- Bump babel from 2.10.2 to 2.10.3 (#12) @dependabot
- Bump ipykernel from 6.14.0 to 6.15.0 (#11) @dependabot
- Bump certifi from 2022.5.18.1 to 2022.6.15 (#10) @dependabot
- Bump traitlets from 5.2.2.post1 to 5.3.0 (#13) @dependabot
- Bump actions/setup-python from 3 to 4 (#6) @dependabot

## Version 0.5.4 - 2022-06-16
### Added
- [ci] Testpypi and pypi, release drafter and labeler.

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
