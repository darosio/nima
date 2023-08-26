# Changelog

## v0.7.3 (2023-08-26)

### Build

- **deps-dev**: bump sphinx from 7.2.2 to 7.2.3 (#279)
- **deps-dev**: bump zarr from 2.16.0 to 2.16.1 (#276)
- **deps-dev**: update scipy requirement from <1.11.2 to <1.11.3 (#273)
- **deps-dev**: bump sphinx from 7.1.2 to 7.2.2 (#274)
- **deps-dev**: bump ruff from 0.0.284 to 0.0.285 (#275)
- **deps-dev**: bump click from 8.1.6 to 8.1.7 (#272)
- **deps-dev**: bump typeguard from 4.1.1 to 4.1.2 (#271)
- **deps-dev**: bump sphinx-click from 4.4.0 to 5.0.1 (#270)
- **deps-dev**: bump mypy from 1.5.0 to 1.5.1 (#269)

### Refactor

- to respect ruff 0.0.285

### chore

- Adjust python-version

## v0.7.2 (2023-08-16)

### Fix

- mypy errors

### Build

- **deps-dev**: bump typeguard from 4.1.0 to 4.1.1 (#268)
- **deps-dev**: bump types-setuptools from 68.0.0.3 to 68.1.0.0 (#267)
- **deps-dev**: update bokeh requirement from <3.2.2 to <3.2.3 (#265)
- **deps-dev**: bump pandas-stubs from 2.0.2.230605 to 2.0.3.230814 (#264)
- **deps-dev**: update tifffile requirement from <=2023.7.18 to <=2023.8.12
- **deps-dev**: update tifffile requirement
- **deps-dev**: bump coverage[toml] from 7.2.7 to 7.3.0 (#262)
- **deps-dev**: bump mypy from 1.4.1 to 1.5.0 (#261)
- **deps-dev**: bump ruff from 0.0.283 to 0.0.284 (#260)
- **deps-dev**: bump ruff from 0.0.282 to 0.0.283 (#259)
- **deps-dev**: update dask[distributed] requirement from <=2023.7.1 to <=2023.8.0 (#257)
- **deps-dev**: bump pygments from 2.15.1 to 2.16.1 (#256)
- prettier --prose-wrap=preserve
- **deps-dev**: bump sphinx from 7.1.1 to 7.1.2 (#255)
- **deps-dev**: bump ruff from 0.0.281 to 0.0.282 (#254)
- **deps-dev**: bump ruff from 0.0.280 to 0.0.281 (#252)
- **deps-dev**: update numpy requirement from <1.25.2 to <1.25.3 (#251)
- **deps-dev**: bump commitizen from 3.5.4 to 3.6.0 (#250)

## v0.7.1 (2023-07-31)

### Fix

- **docs**: Update .gitignore

### Docs

- Update tfor API inclusion

### Test

- Rename script to cli

### Build

- **deps-dev**: bump commitizen from 3.5.3 to 3.5.4 (#249)
- drop python 3.8 and 3.9
- **deps-dev**: bump typeguard from 2.13.3 to 4.1.0
- **deps-dev**: bump sphinx from 7.1.0 to 7.1.1 (#247)

### Refactor

- Follow also SIM code in ruff
- After adding ANN code to ruff
- Align code to ruff linter provisions

## v0.7.0 (2023-07-27)

### Feat

- Add support for python 3.11
- add generat

### Fix

- a darglint error after vmin vmax for kwargs
- bias return read noise instead of Shapiro-Wilk test
- imwrite photometric warnings

### Docs

- Update Readme

### Style

- Add prettier to pre-commit

### Build

- **fix**: update sphinx together with theme and myst-parser
- Update dependencies
  - bump numpy from 1.23.3 to 1.25.1
  - bump pandas from 1.5.0 to 2.0.3
  - bump scikit-image from 0.19.3 to 0.21.0
  - bump matplotlib from 3.6.0 to 3.7.2
  - bump tifffile from 2022.8.12 to 2023.7.18
  - bump scipy from 1.9.1 to 1.11.1
  - bump click from 8.1.3 to 8.1.6
  - bump dask from 2022.9.2 to 2023.7.1
  - bump zarr from 2.13.2 to 2.16.0
  - bump bokeh from 2.4.3 to 3.2.1
  - bump pre-commit from 2.20.0 to 3.3.3
  - bump sphinx-click from 4.3.0 to 4.4.0
  - bump pytest from 7.1.3 to 7.4.0
  - bump pandas-stubs from 1.5.0.221003 to 2.0.2.230605
  - bump mypy from 0.982 to 1.4.1
- **deps-dev**: bump sphinx from 5.2.3 to 7.1.0
- **deps-dev**: bump commitizen from 2.35.0 to 3.5.3 (#243)
- **deps-dev**: bump types-setuptools from 65.4.0.0 to 68.0.0.3 (#237)
- **deps-dev**: bump xdoctest from 1.1.0 to 1.1.1 (#235)
- **deps-dev**: bump coverage[toml] from 6.5.0 to 7.2.7 (#232)
- **deps**: bump actions/cache from 2 to 3 (#230)
- **deps**: bump actions/configure-pages from 2 to 3 (#229)
- **constraint.txt**: Update pip from 22.3.3 to 23.2.1
- **deps-dev**: bump pygments from 2.13.0 to 2.15.1 (#234)
- Move dependabot from deps to target main branch
- **pre-commit**: Add pygrep-hooks, bandit, shellcheck-py and blaken-docs
- from poetry to hatch and customized commitizen
- updates:
  - virtualenv in /.github/workflows
  - pydata-sphinx-theme in /docs
  - xdoctest from 1.0.2 to 1.1.0
  - poetry from 1.1.14 to 1.2.1 in /.github/workflows
  - codecov/codecov-action from 3.1.0 to 3.1.1
  - myst-parser from 0.18.0 to 0.18.1 in /docs
  - sphinx from 5.1.1 to 5.2.3 in /docs
  - dask from 2022.8.1 to 2022.9.2
- **deps**: bump pywavelets from 1.3.0 to 1.4.1 (#121)
- **deps**: bump pytz from 2022.2.1 to 2022.4 (#119)
- **deps**: bump certifi from 2022.6.15 to 2022.9.24 (#120)
- Updating distlib (0.3.4 -> 0.3.5)

### Refactor

- **ruff**: isort
- Add ruff (drop pyupgrade, isort, flake8) to pre-commit
- **codespell**: Add codespell to pre-commit

### chore

- dependabot with time

## v0.6.0 (2022-07-15)

### Feat

- bima bias|dark|flat|mflat removed scripts.py
- new dark with solid hot pixel identification
- tf8 cannot use Client() and persist()

### Refactor

- switch to pathlib.
- replace README.rst with docs/README.md containing Version for cz bump.
- import annotation from \_\_future\_\_.
- dropped lfs for .ipynb and large test .tif.

## v0.5.7 (2022-06-20)

### Feat

- **ci**: add cz

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

- [build] New linters: flake8-rst-docstrings, pep8-naming,
  flake8-comprehensions, flake8_eradicate and flake8-pytest-style.
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

- `poetry version …` and use importlib.metadata.version(\_\_name\_\_) when
  needed.

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

- Move out of flake8-based linting bandit; use system wide as:
  `bandit -r src tests`
- Move out safety; when updating packages dependencies consider:

```
poetry run safety check
poetry show --outdated
```

- Update all packages except matplotlib (I will fix its tests).
- Dropped python-3.7.
- Added python-3.10.
- Changed noxfile to use nox_poetry.

## Version 0.3.1

- Dropping Pyscaffold in favor of poetry.
- Works with python \< 3.7.

## Version 0.3

- Transition to Pyscaffold template for faster dev cycles.

## Version 0.2.3

- Works for clophensor data.
- Heavy on memory.
- Flat and Dark not tested.
