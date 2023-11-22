# Changelog

## v0.8.0 (2023-11-22)

### Feat

- New bg kind `inverse_local_yen`

### Fix

- Remove bokeh dep; generating win-py310 error
- warnings for not closing plots and entropy filter on 16-bit

### Style

- Add type: ignore to all skimage function calls

### Build

- **deps-dev**: bump sphinx-click from 5.0.1 to 5.1.0 (#376)
- **deps-dev**: bump pygments from 2.17.1 to 2.17.2 (#375)
- **deps-dev**: update scipy requirement from <1.11.4 to <1.11.5 (#373)
- **deps-dev**: bump pygments from 2.16.1 to 2.17.1 (#372)
- **deps-dev**: bump matplotlib from 3.8.1 to 3.8.2 (#371)
- **deps-dev**: bump ruff from 0.1.5 to 0.1.6 (#370)
- **deps-dev**: bump pandas from 2.1.2 to 2.1.3 (#367)
- **deps-dev**: bump mypy from 1.6.1 to 1.7.0 (#368)
- **deps-dev**: update dask[distributed] requirement (#366)
- **deps-dev**: bump sphinx-autodoc-typehints from 1.25.0 to 1.25.2 (#365)
- **deps-dev**: update numpy requirement from <1.26.2 to <1.26.3 (#364)
- **deps-dev**: bump sphinx-autodoc-typehints from 1.24.1 to 1.25.0 (#363)
- **deps-dev**: update bokeh requirement from <3.3.1 to <3.3.2 (#362)
- **deps-dev**: bump types-setuptools from 68.2.0.0 to 68.2.0.1 (#361)
- **deps-dev**: bump ruff from 0.1.4 to 0.1.5 (#360)
- **deps-dev**: bump ruff from 0.1.3 to 0.1.4 (#358)
- **deps-dev**: bump sphinx-autodoc-typehints from 1.24.0 to 1.24.1 (#357)
- **deps-dev**: bump matplotlib from 3.8.0 to 3.8.1 (#356)
- **deps-dev**: bump pydata-sphinx-theme from 0.14.2 to 0.14.3 (#355)
- **deps-dev**: update dask[distributed] requirement (#353)
- **deps-dev**: bump ruff from 0.1.2 to 0.1.3 (#352)
- **deps-dev**: bump pandas from 2.1.1 to 2.1.2 (#351)
- **deps-dev**: bump xdoctest from 1.1.1 to 1.1.2 (#350)
- **deps-dev**: bump pydata-sphinx-theme from 0.14.1 to 0.14.2 (#349)
- **deps-dev**: bump ruff from 0.1.1 to 0.1.2 (#348)
- **deps-dev**: bump pytest from 7.4.2 to 7.4.3 (#347)
- **deps**: bump pip from 23.3 to 23.3.1 in /.github/workflows (#345)
- **deps-dev**: bump commitizen from 3.10.1 to 3.12.0 (#342)
- **deps-dev**: bump ruff from 0.1.0 to 0.1.1 (#344)
- **deps-dev**: bump mypy from 1.6.0 to 1.6.1 (#343)
- **deps-dev**: bump ruff from 0.0.292 to 0.1.0 (#341)
- **deps-dev**: bump commitizen from 3.10.0 to 3.10.1 (#338)
- **deps**: bump pip from 23.2.1 to 23.3 in /.github/workflows (#339)
- **deps-dev**: update numpy requirement from <1.26.1 to <1.26.2 (#337)
- **deps-dev**: update dask[distributed] requirement (#336)
- **deps-dev**: bump pre-commit from 3.4.0 to 3.5.0 (#335)
- **deps-dev**: bump mypy from 1.5.1 to 1.6.0 (#334)
- **deps-dev**: update bokeh requirement from <3.2.3 to <3.3.1 (#333)
- **deps-dev**: bump scikit-image from 0.21.0 to 0.22.0
- **deps-dev**: bump coverage[toml] from 7.3.1 to 7.3.2 (#329)
- **deps-dev**: bump ruff from 0.0.291 to 0.0.292 (#328)
- **deps-dev**: update dask[distributed] requirement (#327)
- **deps-dev**: bump pandas-stubs from 2.0.3.230814 to 2.1.1.230928 (#326)
- **deps-dev**: update tifffile requirement (#325)
- **deps-dev**: update scipy requirement from <1.11.3 to <1.11.4 (#324)
- **deps-dev**: bump commitizen from 3.9.0 to 3.10.0 (#322)
- **deps-dev**: bump ruff from 0.0.290 to 0.0.291 (#321)
- **deps-dev**: bump pydata-sphinx-theme from 0.14.0 to 0.14.1 (#320)
- **deps-dev**: bump pandas from 2.1.0 to 2.1.1 (#319)
- **deps-dev**: update tifffile requirement (#318)

## v0.7.4 (2023-09-18)

### Fix

- type checking for matplotlib-3.8.0
- seaborn left over in hist profile plot

### Build

- **deps-dev**: bump matplotlib from 3.7.3 to 3.8.0
- **deps-dev**: update dask[distributed] requirement (#315)
- **deps-dev**: update numpy requirement from <1.25.3 to <1.26.1 (#314)
- **deps-dev**: bump ruff from 0.0.289 to 0.0.290 (#313)
- **deps-dev**: bump commitizen from 3.8.2 to 3.9.0 (#312)
- **deps-dev**: bump pydata-sphinx-theme from 0.13.3 to 0.14.0 (#311)
- **deps-dev**: bump sphinx from 7.2.5 to 7.2.6 (#310)
- **deps-dev**: bump ruff from 0.0.288 to 0.0.289 (#309)
- **deps-dev**: bump matplotlib from 3.7.2 to 3.7.3 (#308)
- **deps-dev**: bump typeguard from 4.1.4 to 4.1.5 (#306)
- **deps-dev**: bump ruff from 0.0.287 to 0.0.288 (#305)
- **deps-dev**: bump typeguard from 4.1.3 to 4.1.4 (#304)
- **deps-dev**: bump commitizen from 3.8.0 to 3.8.2 (#303)
- **deps-dev**: bump pytest from 7.4.1 to 7.4.2 (#302)
- **deps-dev**: bump types-setuptools from 68.1.0.1 to 68.2.0.0 (#301)
- **deps-dev**: update dask[distributed] requirement (#300)
- **deps-dev**: bump coverage[toml] from 7.3.0 to 7.3.1 (#299)
- **deps-dev**: bump commitizen from 3.7.1 to 3.8.0 (#298)
- **deps**: bump actions/checkout from 3 to 4 (#296)
- **deps-dev**: bump commitizen from 3.7.0 to 3.7.1 (#295)
- **deps-dev**: update dask[distributed] requirement (#294)
- **deps-dev**: bump ruff from 0.0.286 to 0.0.287 (#293)
- **deps-dev**: bump pytest from 7.4.0 to 7.4.1 (#292)
- **deps-dev**: bump pre-commit from 3.3.3 to 3.4.0 (#291)
- **deps-dev**: update tifffile requirement (#290)
- **deps-dev**: bump pandas from 2.0.3 to 2.1.0 (#289)
- **deps-dev**: bump sphinx from 7.2.4 to 7.2.5 (#288)
- **deps-dev**: bump types-setuptools from 68.1.0.0 to 68.1.0.1 (#287)
- **deps-dev**: bump sphinx from 7.2.3 to 7.2.4 (#285)
- **deps-dev**: bump sigfig from 1.3.2 to 1.3.3 (#284)
- **deps-dev**: bump typeguard from 4.1.2 to 4.1.3 (#283)
- **deps-dev**: bump ruff from 0.0.285 to 0.0.286 (#282)
- **deps-dev**: bump commitizen from 3.6.0 to 3.7.0 (#281)
- **deps-dev**: update tifffile requirement (#280)
- **deps-dev**: update dask[distributed] requirement (#277)

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
