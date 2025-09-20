# Changelog

## [0.11.8] - 2025-09-20

### 🐙 CI/CD

- Fix script and changelog

## [0.11.7] - 2025-09-20

### 🐙 CI/CD

- Fix release

## [0.11.6] - 2025-09-19

### 🐙 CI/CD

- Add a script to dedup (deps) lines in changelog

## [0.11.5] - 2025-09-19

### 🐛 Bug Fixes

- Renovate configuration file
- Yanked click 8.2.2

### 🦾 Build

- Update ruff codes
- *(deps)* Bump tifffile from 2025.5.21 to 2025.9.9
- *(deps)* Bump s3fs from 2025.5.0 to 2025.9.0
- *(deps)* Bump pandas from 2.2.3 to 2.3.2
- *(deps)* Bump numpy from 2.2.6 to 2.3.3
- *(deps)* Bump scipy from 1.15.3 to 1.16.2
- *(deps)* Bump bioio-tifffile from 1.1.0 to 1.3.0
- *(deps)* Bump dask from 2025.5.1 to 2025.9.1
- *(deps)* Bump pyarrow from 20.0.0 to 21.0.0
- *(deps)* Bump matplotlib from 3.10.3 to 3.10.6
- *(deps)* Bump click from 8.2.1 to 8.3.0
- *(deps)* Bump actions/download-artifact from v4 to v5
- *(deps)* Bump actions/checkout from v4 to v5
- *(deps)* Bump codecov/codecov-action from v5.4.3 to v5.5.1
- *(deps)* Bump actions/upload-pages-artifact from v3 to v4
- *(deps)* Bump xmltodict from 0.14.2 to 1.0.2
- *(deps)* Bump actions/setup-python from v5 to v6

### 🐙 CI/CD

- Drop hatch from ci

## [0.11.4] - 2025-05-22

### 🐛 Bug Fixes

- Changelog and release generation

## [0.11.3] - 2025-05-22

### 🐛 Bug Fixes

- Noisy changelog

## [0.11.2] - 2025-05-22

### 🐛 Bug Fixes

- Release into a separate workflow

## [0.11.1] - 2025-05-22

### 🐛 Bug Fixes

- Release

### 🦾 Build

- *(deps)* Bump tifffile from 2025.5.10 to 2025.5.21 (#1079)

## [0.11.0] - 2025-05-21

### 🚀 Features

- StagePosition as dict key in Metadata
- Add DimensionOrder metadata
- Add read_tiffmd also for multitiled tiff
- Add metadata from TiffReader

### 🐛 Bug Fixes

- Bump and release
- *(build)* Typos linting
- *(docs)* Incompatible (nb)sphinx versions
- Pre-commit commitizen pre-push
- Deps ipython-8.18 questionary-2.0.1 committizen
- *(ci)* Skip pytest for windows and py3.10
- Add pylint to dev deps
- When @Gain is absent in channel DetectorSettings
- Xdoc after pylint refactor
- Xdoc after rename to bg_refine_iteratively()
- Pyproject known first party

### 🚜 Refactor

- Bima mflat test often failing in py10 on windows
- Drop bg2 signal.find_peaks too fragile
- After completing pylint suggestions
- Add pylint
- NPY lint
- For additional linting

### 🦾 Build

- *(ci)* Stop dependabot and use uv add --optional
- *(hooks)* Bump pre-commit hook astral-sh/uv-pre-commit from 0.7.5 to 0.7.6 (#1068)
- *(hooks)* Bump pre-commit hook executablebooks/mdformat from 0.7.17 to 0.7.22
- *(deps)* Bump astral-sh/setup-uv action from v5 to v6 (#1073)
- *(deps)* Bump autodocsumm from 0.2.12 to 0.2.14
- *(deps)* Bump bioio-tifffile from 1.0.0 to 1.1.0
- *(deps)* Bump click from 8.1.7 to 8.2.1
- *(deps)* Bump codecov/codecov-action from 4.3.1 to 5.4.3
- *(deps)* Bump commitizen from 3.27.0 to 4.8.0
- *(deps)* Bump coverage[toml] from 7.5.4 to 7.8.1
- *(deps)* Bump dask to 2025.5.1
- *(deps)* Bump hatch from 1.12.0 to 1.14.1 in /.github/workflows
- *(deps)* Bump ipykernel from 6.29.4 to 6.29.5 (#644)
- *(deps)* Bump jsh9/pydoclint from 0.6.6 to 0.6.7 (#1063)
- *(deps)* Bump jupyter from 1.0.0 to 1.1.1
- *(deps)* Bump jupyterlab-server from 2.27.2 to 2.27.3 (#657)
- *(deps)* Bump matplotlib from 3.9.0 to 3.10.3
- *(deps)* Bump mypy from 1.10.1 to 1.15.0
- *(deps)* Bump nbsphinx from 0.9.4 to 0.9.7
- *(deps)* Bump node from 22.0.0 to 22.15.1 (#967)
- *(deps)* Bump numpy from 2.0.0 to 2.2.6
- *(deps)* Bump pandas from 2.2.2 to 2.2.3 (#731)
- *(deps)* Bump pandas-stubs
- *(deps)* Bump pascalgn/automerge-action from 0.16.3 to 0.16.4 (#730)
- *(deps)* Bump pims from 0.4.1 to 0.6.0 (#921)
- *(deps)* Bump pip from 24.1 to 25.1.1 in /.github/workflows
- *(deps)* Bump pyarrow from 16.1.0 to 20.0.0
- *(deps)* Bump pydata-sphinx-theme
- *(deps)* Bump pygments from 2.18.0 to 2.19.1
- *(deps)* Bump pypdf from 4.2.0 to 5.5.0
- *(deps)* Bump pytest from 8.2.2 to 8.3.5
- *(deps)* Bump ruff from 0.4.10 to 0.11.10
- *(deps)* Bump s3fs from 2023.6.0 to 2025.5.0
- *(deps)* Bump scikit-image from 0.24.0 to 0.25.2
- *(deps)* Bump scipy from 1.14.0 to 1.15.3
- *(deps)* Bump sigfig from 1.3.3 to 1.3.19
- *(deps)* Bump sphinx from 8.1.3 to 8.2.3 (#929)
- *(deps)* Bump sphinx from 7.3.7 to 8.2.3
- *(deps)* Bump sphinx-autodoc-typehints to 3.2.0
- *(deps)* Bump tifffile from 2024.6.18 to 2025.5.10
- *(deps)* Bump urllib3 from 1.26.18 to 2.3.0
- *(deps)* Bump xdoctest from 1.1.5 to 1.2.0
- *(deps)* Bump xmltodict from 0.13.0 to 0.14.2
- *(deps)* Bump zarr from 2.18.2 to 3.0.5
- *(deps)* bump pre-commit from 3.7.1 to 4.2.0
- *(docs)* Bump seaborn from 0.13.1 to 0.13.2
- *(docs)* Bump statsmodels from 0.14.3 to 0.14.4
- *(pre-commit)* Update hooks (#643)
- Add actionlint
- Add json.tool based pre-commit
- Add mdformat
- Add python 3.13
- Add taplo; Drop language-formatters-pre-commit-hooks
- Add uv as the installer
- Add yamlfmt
- Adopt the single ci.yml with automerge dependabot PR
- Adopt typos and drop codespell
- Drop bandit
- Drop prettier
- Drop pylint
- Drop python 3.10
- Fix (nb)sphinx issue
- Fix sphinx update
- Fix yamlfmt (ipv6 issue)
- Lint with npm prettier
- Mdformat config in a dedicated file
- No need to touch uv.lock during updating
- Separate lint and docs scripts
- Separate lint from dev
- Upgrade uv.lock manually

### 🐙 CI/CD

- Adopt uv (#1069)
- *(lint)* Drop pip in favor of uv
- Drop prettier lint
- Drop pre-commit ci
- Fix renovate updating pre-commit
- Renovate update pre-commit hooks
- Fix renovate optional groups update
- Renovate won’t update sphinx-autodoc-typehints as well
- Renovate will not update sphinx\<8.2
- Pin deps for dev and docs
- Avoid double run on push and pull request
- Renovate adjust
- Fix strict json for renovate config file
- Keep using pascalgn/automerge-action
- Config renovate
- Try restoring prettier
- Fix pre-commit cache key
- Temporarily disable prettier
- Fix pre-commit cache
- Fix pip cache

## v0.10.2 (2024-06-25)

### Fix

- bump script
- Add class attribute docstring
- Try to fix the hatch run bump script

### Build

- **deps**: update mypy requirement from \<=1.10.0 to \<=1.10.1 (#635)
- **deps**: update scipy requirement from \<=1.13.1 to \<=1.14.0 (#634)
- **pre-commit**: update hooks

## v0.10.1 (2024-06-24)

### Build

- **deps**: bump codecov/codecov-action from 4.3.1 to 4.5.0
- **deps**: bump hatch from 1.10.0 to 1.12.0 in /.github/workflows
- **deps**: bump jupyterlab-server from 2.27.1 to 2.27.2 (#592)
- **deps**: bump pip from 24.0 to 24.1 in /.github/workflows (#628)
- **deps**: bump pooch from 1.8.1 to 1.8.2 (#615)
- **deps**: bump commitizen from 3.25.0 to 3.27.0
- **deps**: bump coverage[toml] from 7.5.1 to 7.5.4
- **deps**: bump dask[distributed]
- **deps**: bump matplotlib from 3.8.4 to 3.9.0 (#582)
- **deps**: bump numpy from 1.26.4 to 2.0.0 (#619)
- **deps**: bump pandas-stubs
- **deps**: bump pre-commit from 3.7.0 to 3.7.1 (#575)
- **deps**: bump pyarrow from 16.0.0 to 16.1.0 (#579)
- **deps**: bump pydata-sphinx-theme requirement (#603)
- **deps**: bump pytest from 8.2.0 to 8.2.2
- **deps**: bump ruff from 0.4.3 to 0.4.10
- **deps**: bump scikit-image from 0.23.2 to 0.24.0 (#624)
- **deps**: bump scipy from 1.13.0 to 1.13.1 (#596)
- **deps**: bump sphinx-autodoc-typehints
- **deps**: bump sphinx-click from 5.1.0 to 6.0.0
- **deps**: bump tifffile from 2024.5.3 to 2024.6.18
- **deps**: bump xdoctest from 1.1.3 to 1.1.5
- **deps**: bump zarr from 2.18.0 to 2.18.2
- **pre-commit**: update hooks

### Refactor

- Drop aicsimageio in favor of bioio-tifffile

## v0.10.0 (2024-05-10)

### Feat

- Add geometri mean filter

### Fix

- gen_frame.clip(0) revealed fit_gaussian unstable with large sd
- iteratively_refine_background
- Improve BgParams with optional clip and erosion
- slow env create
- **docs**: Warning for main docstr with \*

### Docs

- Compare methods for bg estimation
- Fix an import in usage

### Style

- NDArray

### Test

- Test PDF output without reference files

### Build

- **deps**: bump actions/configure-pages from 4 to 5 (#513)
- **deps**: bump codecov/codecov-action from 4.1.1 to 4.3.1
- **deps**: bump hatch from 1.9.4 to 1.10.0 in /.github/workflows
- **deps**: bump jupyterlab-server from 2.27.0 to 2.27.1 (#554)
- **deps**: bump questionary from 2.0.0 to 2.0.1 (#550)
- **deps**: bump commitizen from 3.20.0 to 3.25.0
- **deps**: bump coverage[toml] from 7.4.4 to 7.5.1
- **deps**: bump dask[distributed] requirement
- **deps**: bump ipython from 8.18.0 to 8.23.0 (#546)
- **deps**: bump jupyterlab-server
- **deps**: bump mypy from 1.9.0 to 1.10.0
- **deps**: bump nbsphinx from 0.9.3 to 0.9.4
- **deps**: bump pyarrow from 15.0.2 to 16.0.0
- **deps**: bump pygments from 2.17.2 to 2.18.0
- **deps**: bump pytest from 8.1.1 to 8.2.0
- **deps**: bump ruff from 0.3.4 to 0.4.3
- **deps**: bump scikit-image to 0.23.2
- **deps**: bump sphinx from 7.2.6 to 7.3.7
- **deps**: bump sphinx-autodoc-typehints
- **deps**: bump tifffile from 2024.2.12 to 2024.5.3
- **deps**: bump types-setuptools
- **deps**: bump urllib3 from 1.26.18 to 2.2.1 (#547)
- **deps**: bump zarr from 2.17.1 to 2.18.0
- **deps-dev**: bump matplotlib from 3.8.3 to 3.8.4 (#519)
- **deps-dev**: bump pandas from 2.2.1 to 2.2.2 (#527)
- **deps-dev**: bump scipy from 1.12.0 to 1.13.0 (#518)
- Correct mypy configuration
- Drop typeguard test
- Drop types-setuptools
- Move aicsimageio into deps
- Pin few indirect deps to speed up pip
- Update to py3.12

### Refactor

- Add BgResult grouping dataclass
- Drop myhist()
- bg introducing BgParams
- generat for pylint
- To include pylint rule extract watershed function
- skimage import in nima; d_mask_label
- main cli
- read_tiff
- boolean traps
- exceptions
- Drop data-science-types
- **test**: Use consistently TESTS_PATH
- Add segmentation module
- **build**: From black to ruff format

### chore

- Modify pre-commit update message
- update pre-commit hooks (#545)

## v0.9.1 (2024-03-28)

### Fix

- **ci**: Add codecov token

### Build

- **deps-dev**: update ipykernel requirement from \<=6.29.3 to \<=6.29.4 (#512)

## v0.9.0 (2024-03-28)

### Feat

- Add utils and more temporary tutorials

### Fix

- **ci**: TestPYPI upload
- tests after refactoring
- trivial type error
- commitizen version v3.13.0 from 3.12.0

### Docs

- Cleaning tutorials
- Improve CLI doc strings
- Reorganize and update tutorials
- Fix tutorials
- Fix tutorials building
- Add ipynb tutorial
- Adopt github-pages-deploy-action

### Style

- pyproject.toml

### Build

- **deps**: bump actions/cache from 3 to 4 (#422)
- **deps**: bump actions/configure-pages from 3 to 4 (#385)
- **deps**: bump actions/deploy-pages from 2 to 3 (#384)
- **deps**: bump actions/download-artifact from 3 to 4 (#394)
- **deps**: bump actions/setup-python from 4 to 5 (#388)
- **deps**: bump codecov/codecov-action from 3.1.4 to 4.1.1
- **deps**: bump hatch from 1.7.0 to 1.9.4 in /.github/workflows
- **deps**: bump pip from 23.3.1 to 24.0 in /.github/workflows
- **deps-dev**: bump commitizen from 3.12.0 to 3.13.0
- **deps-dev**: bump mypy from 1.7.0 to 1.7.1 (#378)
- **deps-dev**: bump pydata-sphinx-theme from 0.14.3 to 0.14.4 (#379)
- **deps-dev**: bump types-setuptools from 68.2.0.1 to 69.0.0.0
- **deps-dev**: bump autodocsumm
- **deps-dev**: bump commitizen from 3.13.0 to 3.20.0
- **deps-dev**: bump coverage[toml]
- **deps-dev**: bump dask[distributed]
- **deps-dev**: bump ipykernel from 6.27.0 to 6.29.3
- **deps-dev**: bump matplotlib from 3.8.2 to 3.8.3
- **deps-dev**: bump mypy from 1.7.1 to 1.9.0
- **deps-dev**: bump numpy from 1.26.2 to 1.26.4
- **deps-dev**: bump pandas from 2.1.3 to 2.2.1
- **deps-dev**: bump pandas-stubs
- **deps-dev**: bump pre-commit from 3.5.0 to 3.7.0
- **deps-dev**: bump pyarrow from 15.0.0 to 15.0.2
- **deps-dev**: bump pydata-sphinx-theme
- **deps-dev**: bump pytest from 7.4.3 to 8.1.1
- **deps-dev**: bump ruff from 0.1.11 to 0.3.4
- **deps-dev**: bump scipy from 1.11.4 to 1.12.0
- **deps-dev**: bump sphinx-autodoc-typehints
- **deps-dev**: bump tifffile
- **deps-dev**: bump typeguard from 4.1.5 to 4.2.1
- **deps-dev**: bump types-setuptools
- **deps-dev**: bump xdoctest from 1.1.2 to 1.1.3
- **deps-dev**: bump zarr from 2.16.1 to \<=2.17.1
- **docs**: Drop darglint in favor of pydoclint; fix some docstring
- **pre-commit**: Run autoupdate
- **pre-commit**: prettier v4.0.0-alpha.8

### CI/CD

- Docs out of gh-pages

### Refactor

- utils
- Improve generat module

### chore

- Fix lint

## v0.8.0 (2023-11-22)

### Feat

- New bg kind `inverse_local_yen`

### Fix

- Remove bokeh dep; generating win-py310 error
- warnings for not closing plots and entropy filter on 16-bit

### Style

- Add type: ignore to all skimage function calls

### Build

- **deps**: bump pip from 23.2.1 to 23.3.1 in /.github/workflows
- **deps-dev**: bump commitizen from 3.9.0 to 3.12.0
- **deps-dev**: bump coverage[toml] from 7.3.1 to 7.3.2 (#329)
- **deps-dev**: bump matplotlib from 3.8.0 to 3.8.2
- **deps-dev**: bump mypy from 1.5.1 to 1.7.0
- **deps-dev**: bump pandas from 2.1.0 to 2.1.3
- **deps-dev**: bump pandas-stubs from 2.0.3.230814 to 2.1.1.230928 (#326)
- **deps-dev**: bump pre-commit from 3.4.0 to 3.5.0 (#335)
- **deps-dev**: bump pydata-sphinx-theme from 0.14.0 to 0.14.3
- **deps-dev**: bump pygments from 2.16.1 to 2.17.2
- **deps-dev**: bump pytest from 7.4.2 to 7.4.3 (#347)
- **deps-dev**: bump ruff from 0.0.290 to 0.1.6
- **deps-dev**: bump scikit-image from 0.21.0 to 0.22.0
- **deps-dev**: bump sphinx-autodoc-typehints from 1.24.0 to 1.25.2
- **deps-dev**: bump sphinx-click from 5.0.1 to 5.1.0 (#376)
- **deps-dev**: bump types-setuptools from 68.2.0.0 to 68.2.0.1 (#361)
- **deps-dev**: bump xdoctest from 1.1.1 to 1.1.2 (#350)
- **deps-dev**: bump bokeh from 3.2.3 to 3.3.2
- **deps-dev**: update dask[distributed] requirement
- **deps-dev**: update scipy requirement from \<1.11.3 to \<1.26.3
- **deps-dev**: update tifffile requirement (#325)

## v0.7.4 (2023-09-18)

### Fix

- type checking for matplotlib-3.8.0
- seaborn left over in hist profile plot

### Build

- **deps-dev**: bump matplotlib from 3.7.2 to 3.8.0
- **deps-dev**: bump dask[distributed]
- **deps-dev**: bump numpy from 1.25.3 to 1.26.1 (#314)
- **deps-dev**: bump ruff from 0.0.285 to 0.0.290
- **deps-dev**: bump commitizen from 3.6.0 to 3.9.0
- **deps-dev**: bump pydata-sphinx-theme from 0.13.3 to 0.14.0 (#311)
- **deps-dev**: bump sphinx from 7.2.3 to 7.2.6
- **deps-dev**: bump typeguard from 4.1.2 to 4.1.5
- **deps-dev**: bump pytest from 7.4.0 to 7.4.2
- **deps-dev**: bump types-setuptools from 68.1.0.0 to 68.2.0.0
- **deps-dev**: bump coverage[toml] from 7.3.0 to 7.3.1 (#299)
- **deps-dev**: bump pre-commit from 3.3.3 to 3.4.0 (#291)
- **deps-dev**: bump tifffile
- **deps-dev**: bump pandas from 2.0.3 to 2.1.0 (#289)
- **deps-dev**: bump sigfig from 1.3.2 to 1.3.3 (#284)
- **deps**: bump actions/checkout from 3 to 4 (#296)

## v0.7.3 (2023-08-26)

### Build

- **deps-dev**: bump sphinx from 7.1.2 to 7.2.3 (#274)
- **deps-dev**: bump zarr from 2.16.0 to 2.16.1 (#276)
- **deps-dev**: bump scipy from 1.11.2 to 1.11.3 (#273)
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
- **deps-dev**: bump bokeh from 3.2.2 to 3.2.3 (#265)
- **deps-dev**: bump pandas-stubs from 2.0.2.230605 to 2.0.3.230814 (#264)
- **deps-dev**: bump tifffile from 2023.7.18 to 2023.8.12
- **deps-dev**: bump coverage[toml] from 7.2.7 to 7.3.0 (#262)
- **deps-dev**: bump mypy from 1.4.1 to 1.5.0 (#261)
- **deps-dev**: bump ruff from 0.0.280 to 0.0.284
- **deps-dev**: bump dask[distributed] from 2023.7.1 to 2023.8.0 (#257)
- **deps-dev**: bump pygments from 2.15.1 to 2.16.1 (#256)
- **deps-dev**: bump sphinx from 7.1.1 to 7.1.2 (#255)
- **deps-dev**: bump numpy from 1.25.2 to 1.25.3 (#251)
- **deps-dev**: bump commitizen from 3.5.4 to 3.6.0 (#250)
- prettier --prose-wrap=preserve

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
- Works with python < 3.7.

## Version 0.3

- Transition to Pyscaffold template for faster dev cycles.

## Version 0.2.3

- Works for clophensor data.
- Heavy on memory.
- Flat and Dark not tested.
