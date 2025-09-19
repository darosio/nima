# Changelog

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

- *(tests)* Bump coverage from 7.8.0 to 7.8.1 (#1076)
- *(deps)* Bump dask from 2025.5.0 to 2025.5.1 (#1074)
- *(deps)* Bump click from 8.2.0 to 8.2.1 (#1075)
- Mdformat config in a dedicated file
- *(deps)* Bump astral-sh/setup-uv action from v5 to v6 (#1073)
- No need to touch uv.lock during updating
- *(deps)* Bump s3fs from 2025.3.2 to 2025.5.0 (#1072)
- *(hooks)* Bump pre-commit hook commitizen-tools/commitizen from v4.7.2 to v4.8.0 (#1071)
- *(dev)* Bump commitizen from 4.7.2 to 4.8.0 (#1070)
- *(hooks)* Bump pre-commit hook astral-sh/uv-pre-commit from 0.7.5 to 0.7.6 (#1068)
- Fix (nb)sphinx issue
- *(hooks)* Bump pre-commit hook executablebooks/mdformat from 0.7.17 to 0.7.22 (#1065)
- *(docs)* Pin dependencies (#1064)
- *(deps)* Bump jsh9/pydoclint from 0.6.6 to 0.6.7 (#1063)
- *(deps)* Bump astral-sh/ruff-pre-commit from v0.11.9 to v0.11.10 (#1060)
- *(deps)* Bump commitizen-tools/commitizen from v4.7.0 to v4.7.2 (#1061)
- Add actionlint
- Fix yamlfmt (ipv6 issue)
- Add taplo; Drop language-formatters-pre-commit-hooks
- Add mdformat
- Add yamlfmt
- Add json.tool based pre-commit
- Drop prettier
- *(dev)* Bump commitizen from 4.7.1 to 4.7.2 (#1058)
- Separate lint and docs scripts
- Separate lint from dev
- *(docs)* Bump statsmodels from 0.14.3 to 0.14.4
- *(docs)* Bump seaborn from 0.13.1 to 0.13.2 (#1055)
- *(deps)* Bump numpy from 2.2.5 to 2.2.6 (#1056)
- *(docs)* Bump seaborn from 0.13.1 to 0.13.2 (#992)
- *(dev)* Bump dev dev (#991)
- *(docs)* Bump seaborn from 0.13.1 to 0.13.2 (#990)
- *(docs)* Bump seaborn from 0.13.1 to 0.13.2 (#989)
- *(docs)* Bump seaborn from 0.13.1 to 0.13.2 (#988)
- *(dev)* Bump ruff from 0.11.7 to 0.11.10 (#986)
- *(dev)* Bump dev dev (#984)
- *(docs)* Bump docs docs (#979)
- *(tests)* Bump pypdf from 5.4.0 to 5.5.0 (#981)
- *(deps)* Bump dask from 2025.4.1 to 2025.5.0 (#966)
- *(deps)* Bump tifffile from 2025.3.30 to 2025.5.10 (#969)
- *(deps)* Bump node from 22.0.0 to 22.15.1 (#967)
- *(deps)* Bump click from 8.1.8 to 8.2.0 (#965)
- *(deps)* Bump matplotlib from 3.10.1 to 3.10.3 (#962)
- *(deps)* Bump scipy from 1.15.2 to 1.15.3 (#963)
- *(deps)* Bump codecov/codecov-action from v5.4.2 to v5.4.3 (#961)
- *(pre-commit)* Update hooks (#960)
- *(deps)* Bump commitizen from 4.4.1 to 4.6.3 (#959)
- *(ci)* Stop dependabot and use uv add --optional
- *(deps)* Bump commitizen from 4.4.1 to 4.6.3 (#957)
- *(deps)* Bump ruff from 0.11.2 to 0.11.8 (#958)
- *(deps)* Bump pyarrow from 19.0.1 to 20.0.0 (#952)
- *(deps)* Bump sphinx-autodoc-typehints from 3.1.0 to 3.2.0 (#951)
- *(deps)* Bump dask from 2025.3.0 to 2025.4.1 (#950)
- *(deps)* Bump ruff from 0.11.2 to 0.11.8 (#946)
- *(deps)* Bump numpy from 2.2.4 to 2.2.5 (#945)
- *(deps)* Bump commitizen from 4.4.1 to 4.6.3 (#949)
- *(deps)* Bump bioio-tifffile from 1.0.1 to 1.1.0 (#948)
- *(pre-commit)* Update hooks (#943)
- *(deps)* Bump pip from 25.1 to 25.1.1 in /.github/workflows (#942)
- *(deps)* Bump nbsphinx from 0.9.6 to 0.9.7 (#941)
- *(pre-commit)* Update hooks (#940)
- *(deps)* Bump pip from 25.0.1 to 25.1 in /.github/workflows (#939)
- *(pre-commit)* Update hooks (#938)
- *(deps)* Bump codecov/codecov-action from 5.4.0 to 5.4.2 (#937)
- *(pre-commit)* Update hooks (#936)
- *(deps)* Bump nbsphinx from 0.9.6 to 0.9.7 (#935)
- *(deps)* Bump nbsphinx from 0.9.6 to 0.9.7 (#934)
- *(deps)* Bump hatch from 1.14.0 to 1.14.1 in /.github/workflows (#933)
- *(pre-commit)* Update hooks (#932)
- *(deps)* Bump nbsphinx from 0.9.6 to 0.9.7 (#931)
- *(deps)* Bump nbsphinx from 0.9.6 to 0.9.7 (#930)
- Fix sphinx update
- *(deps)* Bump sphinx from 8.1.3 to 8.2.3 (#929)
- *(deps)* Bump sphinx-autodoc-typehints from 3.0.1 to 3.1.0 (#927)
- Drop python 3.10
- Upgrade uv.lock manually
- *(pre-commit)* Update hooks (#925)
- *(deps)* Bump s3fs from 2025.3.1 to 2025.3.2 (#926)
- *(deps)* Update tifffile requirement from \<=2025.3.13 to \<=2025.3.30 (#924)
- *(deps)* Bump s3fs from 2025.3.0 to 2025.3.1 (#923)
- *(deps)* Update coverage[toml] requirement from \<=7.7.1 to \<=7.8.0 (#922)
- *(deps)* Bump pims from 0.4.1 to 0.6.0 (#921)
- *(deps)* Update dask[distributed] requirement from \<=2025.2.0 to \<=2025.3.0 (#918)
- *(pre-commit)* Update hooks (#920)
- *(deps)* Update coverage[toml] requirement from \<=7.7.0 to \<=7.7.1 (#919)
- *(deps)* Update ruff requirement from \<=0.11.1 to \<=0.11.2 (#917)
- *(deps)* Update bioio-tifffile requirement from \<=1.0.0 to \<=1.0.1 (#915)
- *(deps)* Update ruff requirement from \<=0.11.0 to \<=0.11.1 (#916)
- *(deps)* Update pre-commit requirement from \<=4.1.0 to \<=4.2.0 (#914)
- *(pre-commit)* Update hooks (#913)
- *(deps)* Update ruff requirement from \<=0.10.0 to \<=0.11.0 (#909)
- *(deps)* Update pypdf requirement from \<=5.3.1 to \<=5.4.0 (#912)
- *(deps)* Update coverage[toml] requirement from \<=7.6.12 to \<=7.7.0 (#911)
- *(deps)* Update numpy requirement from \<=2.2.3 to \<=2.2.4 (#910)
- *(deps)* Update tifffile requirement from \<=2025.2.18 to \<=2025.3.13 (#908)
- *(deps)* Update ruff requirement from \<=0.9.10 to \<=0.10.0 (#907)
- *(deps)* Update sigfig requirement from \<=1.3.18 to \<=1.3.19 (#906)
- *(pre-commit)* Update hooks (#905)
- *(deps)* Update pandas-stubs requirement (#904)
- *(deps)* Update ruff requirement from \<=0.9.9 to \<=0.9.10 (#903)
- *(deps)* Bump s3fs from 2025.2.0 to 2025.3.0 (#902)
- *(deps)* Update zarr requirement from \<=3.0.4 to \<=3.0.5 (#901)
- *(deps)* Update pypdf requirement from \<=5.3.0 to \<=5.3.1 (#897)
- *(deps)* Update nbsphinx requirement from \<=0.9.6 to \<=0.9.7 (#900)
- *(pre-commit)* Update hooks (#899)
- *(deps)* Update ruff requirement from \<=0.9.8 to \<=0.9.9 (#896)
- *(deps)* Update commitizen requirement from \<=4.2.2 to \<=4.4.1 (#898)
- *(deps)* Update sphinx requirement from \<=8.1.3 to \<=8.2.3 (#895)
- *(deps)* Update pytest requirement from \<=8.3.4 to \<=8.3.5 (#894)
- *(deps)* Update ruff requirement from \<=0.9.7 to \<=0.9.8 (#893)
- *(deps)* Update matplotlib requirement from \<=3.10.0 to \<=3.10.1 (#892)
- *(deps)* Bump codecov/codecov-action from 5.3.1 to 5.4.0 (#891)
- *(pre-commit)* Update hooks (#890)
- *(deps)* Update zarr requirement from \<=3.0.3 to \<=3.0.4 (#889)
- Lint with npm prettier
- Add python 3.13
- *(deps)* Update ruff requirement from \<=0.9.6 to \<=0.9.7 (#887)
- Add uv as the installer
- Drop pylint
- Drop bandit
- *(deps)* Update tifffile requirement from \<=2025.1.10 to \<=2025.2.18 (#881)
- *(deps)* Update sphinx-autodoc-typehints requirement (#885)
- *(deps)* Update pyarrow requirement from \<=19.0.0 to \<=19.0.1 (#884)
- *(deps)* Update commitizen requirement from \<=4.2.1 to \<=4.2.2 (#886)
- *(deps)* Update scikit-image requirement from \<=0.25.1 to \<=0.25.2 (#882)
- *(pre-commit)* Update hooks (#880)
- *(deps)* Update scipy requirement from \<=1.15.1 to \<=1.15.2 (#879)
- *(deps)* Update zarr requirement from \<=3.0.2 to \<=3.0.3 (#878)
- *(deps)* Update dask[distributed] requirement (#877)
- *(deps)* Update numpy requirement from \<=2.2.2 to \<=2.2.3 (#876)
- *(deps)* Update coverage[toml] requirement from \<=7.6.11 to \<=7.6.12 (#875)
- *(deps)* Update ruff requirement from \<=0.9.5 to \<=0.9.6 (#874)
- *(pre-commit)* Update hooks (#873)
- *(deps)* Update coverage[toml] requirement from \<=7.6.10 to \<=7.6.11 (#872)
- *(deps)* Update commitizen requirement from \<=4.2.0 to \<=4.2.1 (#871)
- *(deps)* Update pypdf requirement from \<=5.2.0 to \<=5.3.0 (#870)
- *(deps)* Bump pip from 25.0 to 25.0.1 in /.github/workflows (#869)
- *(deps)* Update commitizen requirement from \<=4.1.1 to \<=4.2.0 (#868)
- *(deps)* Update ruff requirement from \<=0.9.4 to \<=0.9.5 (#867)
- *(deps)* Update mypy requirement from \<=1.14.1 to \<=1.15.0 (#866)
- *(pre-commit)* Update hooks (#865)
- *(deps)* Update zarr requirement from \<=3.0.1 to \<=3.0.2 (#864)
- *(deps)* Bump s3fs from 2024.12.0 to 2025.2.0 (#863)
- *(deps)* Update ruff requirement from \<=0.9.3 to \<=0.9.4 (#862)
- *(pre-commit)* Update hooks (#861)
- Adopt typos and drop codespell
- *(deps)* Bump codecov/codecov-action from 5.3.0 to 5.3.1 (#860)
- *(deps)* Bump pip from 24.3.1 to 25.0 in /.github/workflows (#859)
- *(deps)* Update scikit-image requirement from \<=0.25.0 to \<=0.25.1 (#858)
- *(deps)* Update commitizen requirement from \<=4.1.0 to \<=4.1.1 (#857)
- *(deps)* Update pypdf requirement from \<=5.1.0 to \<=5.2.0 (#856)
- *(deps)* Bump codecov/codecov-action from 5.2.0 to 5.3.0 (#855)
- *(deps)* Update ruff requirement from \<=0.9.2 to \<=0.9.3 (#854)
- *(deps)* Bump codecov/codecov-action from 5.1.2 to 5.2.0 (#853)
- *(deps)* Update pre-commit requirement from \<=4.0.1 to \<=4.1.0 (#852)
- *(pre-commit)* Update hooks (#851)
- *(deps)* Update dask[distributed] requirement (#850)
- *(deps)* Update numpy requirement from \<=2.2.1 to \<=2.2.2 (#849)
- *(deps)* Update zarr requirement from \<=3.0.0 to \<=3.0.1 (#848)
- *(deps)* Update pyarrow requirement from \<=18.1.0 to \<=19.0.0 (#847)
- *(deps)* Update ruff requirement from \<=0.9.1 to \<=0.9.2 (#846)
- *(deps)* Update sphinx-autodoc-typehints requirement (#845)
- *(pre-commit)* Update hooks (#844)
- *(deps)* Update scipy requirement from \<=1.15.0 to \<=1.15.1 (#843)
- *(deps)* Update tifffile requirement (#842)
- *(deps)* Update ruff requirement from \<=0.9.0 to \<=0.9.1 (#841)
- *(deps)* Update zarr requirement from \<=2.18.4 to \<=3.0.0 (#839)
- *(deps)* Update ruff requirement from \<=0.8.6 to \<=0.9.0 (#840)
- *(deps)* Update pygments requirement from \<=2.18.0 to \<=2.19.1 (#838)
- *(pre-commit)* Update hooks (#837)
- *(deps)* Update pygments requirement from \<=2.18.0 to \<=2.19.0 (#836)
- *(deps)* Update ruff requirement from \<=0.8.5 to \<=0.8.6 (#835)
- *(deps)* Update scipy requirement from \<=1.14.1 to \<=1.15.0 (#834)
- *(deps)* Update sphinx-autodoc-typehints requirement (#833)
- *(deps)* Update ruff requirement from \<=0.8.4 to \<=0.8.5 (#832)
- *(deps)* Update mypy requirement from \<=1.14.0 to \<=1.14.1 (#831)
- *(pre-commit)* Update hooks (#830)
- *(deps)* Update coverage[toml] requirement from \<=7.6.9 to \<=7.6.10 (#829)
- *(deps)* Update nbsphinx requirement from \<=0.9.5 to \<=0.9.6 (#828)
- *(pre-commit)* Update hooks (#827)
- *(deps)* Update click requirement from \<=8.1.7 to \<=8.1.8 (#824)
- *(deps)* Update mypy requirement from \<=1.13.0 to \<=1.14.0 (#826)
- *(deps)* Bump urllib3 from 2.2.3 to 2.3.0 (#825)
- *(deps)* Update numpy requirement from \<=2.2.0 to \<=2.2.1 (#823)
- *(deps)* Bump s3fs from 2024.10.0 to 2024.12.0 (#822)
- *(deps)* Update ruff requirement from \<=0.8.3 to \<=0.8.4 (#821)
- *(deps)* Bump codecov/codecov-action from 5.1.1 to 5.1.2 (#820)
- *(deps)* Update pydata-sphinx-theme requirement (#819)
- *(deps)* Update dask[distributed] requirement (#818)
- *(pre-commit)* Update hooks (#817)
- *(deps)* Update scikit-image requirement from \<=0.24.0 to \<=0.25.0 (#816)
- *(deps)* Update matplotlib requirement from \<=3.9.3 to \<=3.10.0 (#815)
- *(deps)* Bump hatch from 1.13.0 to 1.14.0 in /.github/workflows (#814)
- *(deps)* Update tifffile requirement (#813)
- *(deps)* Update zarr requirement from \<=2.18.3 to \<=2.18.4 (#812)
- *(deps)* Update ruff requirement from \<=0.8.2 to \<=0.8.3 (#811)
- *(pre-commit)* Update hooks (#810)
- *(deps)* Update numpy requirement from \<=2.1.3 to \<=2.2.0 (#807)
- *(deps)* Update commitizen requirement from \<=4.0.0 to \<=4.1.0 (#809)
- *(deps)* Update coverage[toml] requirement from \<=7.6.8 to \<=7.6.9 (#808)
- *(deps)* Bump codecov/codecov-action from 5.0.7 to 5.1.1 (#806)
- *(deps)* Update ruff requirement from \<=0.8.1 to \<=0.8.2 (#805)
- *(deps)* Update dask[distributed] requirement (#804)
- *(pre-commit)* Update hooks (#803)
- *(deps)* Update ruff requirement from \<=0.8.0 to \<=0.8.1 (#802)
- *(deps)* Update matplotlib requirement from \<=3.9.2 to \<=3.9.3 (#801)
- *(deps)* Update pytest requirement from \<=8.3.3 to \<=8.3.4 (#800)
- *(deps)* Update pandas-stubs requirement (#799)
- *(pre-commit)* Update hooks (#796)
- *(deps)* Update commitizen requirement from \<=3.31.0 to \<=4.0.0 (#798)
- *(deps)* Update pyarrow requirement from \<=18.0.0 to \<=18.1.0 (#797)
- *(deps)* Update ruff requirement from \<=0.7.4 to \<=0.8.0 (#795)
- *(deps)* Update coverage[toml] requirement from \<=7.6.7 to \<=7.6.8 (#794)
- *(deps)* Bump codecov/codecov-action from 5.0.4 to 5.0.7 (#793)
- *(deps)* Bump codecov/codecov-action from 5.0.2 to 5.0.4 (#792)
- *(pre-commit)* Update hooks (#791)
- *(deps)* Update commitizen requirement from \<=3.30.1 to \<=3.31.0 (#790)
- *(deps)* Update ruff requirement from \<=0.7.3 to \<=0.7.4 (#789)
- *(deps)* Update coverage[toml] requirement from \<=7.6.5 to \<=7.6.7 (#788)
- *(deps)* Bump codecov/codecov-action from 5.0.0 to 5.0.2 (#787)
- *(deps)* Bump codecov/codecov-action from 4.6.0 to 5.0.0 (#786)
- *(deps)* Update coverage[toml] requirement from \<=7.6.4 to \<=7.6.5 (#785)
- *(deps)* Update dask[distributed] requirement (#784)
- *(deps)* Update dask[distributed] requirement (#783)
- *(pre-commit)* Update hooks (#782)
- *(deps)* Update ruff requirement from \<=0.7.2 to \<=0.7.3 (#781)
- *(deps)* Update commitizen requirement from \<=3.30.0 to \<=3.30.1 (#780)
- *(deps)* Update dask[distributed] requirement (#779)
- *(pre-commit)* Update hooks (#778)
- *(deps)* Update sigfig requirement from \<=1.3.17 to \<=1.3.18 (#777)
- *(deps)* Update numpy requirement from \<=2.1.2 to \<=2.1.3 (#776)
- *(deps)* Update ruff requirement from \<=0.7.1 to \<=0.7.2 (#775)
- *(deps)* Update pyarrow requirement from \<=17.0.0 to \<=18.0.0 (#774)
- *(pre-commit)* Update hooks (#773)
- *(deps)* Update pypdf requirement from \<=5.0.1 to \<=5.1.0 (#772)
- *(deps)* Update sigfig requirement from \<=1.3.14 to \<=1.3.17 (#771)
- *(deps)* Bump pip from 24.2 to 24.3.1 in /.github/workflows (#770)
- *(deps)* Update ruff requirement from \<=0.7.0 to \<=0.7.1 (#769)
- *(deps)* Update sigfig requirement from \<=1.3.3 to \<=1.3.14 (#768)
- *(deps)* Update autodocsumm requirement from \<=0.2.13 to \<=0.2.14 (#767)
- *(deps)* Update commitizen requirement from \<=3.29.1 to \<=3.30.0 (#766)
- *(deps)* Update mypy requirement from \<=1.12.1 to \<=1.13.0 (#765)
- *(deps)* Update pydata-sphinx-theme requirement (#764)
- *(pre-commit)* Update hooks (#763)
- *(deps)* Update mypy requirement from \<=1.12.0 to \<=1.12.1 (#761)
- *(deps)* Bump s3fs from 2024.9.0 to 2024.10.0 (#762)
- *(deps)* Update coverage[toml] requirement from \<=7.6.3 to \<=7.6.4 (#760)
- *(deps)* Update ruff requirement from \<=0.6.9 to \<=0.7.0 (#759)
- *(deps)* Update dask[distributed] requirement (#758)
- *(deps)* Update xmltodict requirement from \<=0.14.1 to \<=0.14.2 (#757)
- *(deps)* Update mypy requirement from \<=1.11.2 to \<=1.12.0 (#756)
- *(pre-commit)* Update hooks (#755)
- *(deps)* Update coverage[toml] requirement from \<=7.6.2 to \<=7.6.3 (#754)
- *(deps)* Update sphinx requirement from \<=8.1.0 to \<=8.1.3 (#753)
- *(deps)* Bump hatch from 1.12.0 to 1.13.0 in /.github/workflows (#752)
- *(deps)* Update sphinx requirement from \<=8.0.2 to \<=8.1.0 (#751)
- *(deps)* Update coverage[toml] requirement from \<=7.6.1 to \<=7.6.2 (#748)
- *(deps)* Update xmltodict requirement from \<=0.14.0 to \<=0.14.1 (#750)
- *(deps)* Update pandas-stubs requirement (#749)
- *(deps)* Update sphinx-autodoc-typehints requirement (#747)
- *(deps)* Update xmltodict requirement from \<=0.13.0 to \<=0.14.0 (#746)
- *(deps)* Update pre-commit requirement from \<=4.0.0 to \<=4.0.1 (#745)
- *(deps)* Update numpy requirement from \<=2.1.1 to \<=2.1.2 (#743)
- *(deps)* Update pre-commit requirement from \<=3.8.0 to \<=4.0.0 (#742)
- *(deps)* Update ruff requirement from \<=0.6.8 to \<=0.6.9 (#741)
- *(deps)* Bump codecov/codecov-action from 4.5.0 to 4.6.0 (#740)
- *(pre-commit)* Update hooks (#739)
- *(deps)* Update pypdf requirement from \<=5.0.0 to \<=5.0.1 (#738)
- *(deps)* Update dask[distributed] requirement (#737)
- *(deps)* Update ruff requirement from \<=0.6.7 to \<=0.6.8 (#736)
- *(deps)* Update commitizen requirement from \<=3.29.0 to \<=3.29.1 (#735)
- *(pre-commit)* Update hooks (#734)
- *(deps)* Update tifffile requirement from \<=2024.8.30 to \<=2024.9.20 (#733)
- *(deps)* Update ruff requirement from \<=0.6.5 to \<=0.6.7 (#732)
- *(deps)* Update pandas requirement from \<=2.2.2 to \<=2.2.3 (#731)
- *(deps)* Bump pascalgn/automerge-action from 0.16.3 to 0.16.4 (#730)
- *(deps)* Update sphinx-autodoc-typehints requirement (#729)
- *(deps)* Update pypdf requirement from \<=4.3.1 to \<=5.0.0 (#728)
- *(deps)* Update sphinx-autodoc-typehints requirement (#727)
- *(deps)* Update ruff requirement from \<=0.6.4 to \<=0.6.5 (#725)
- *(pre-commit)* Update hooks (#720)
- *(deps)* Update dask[distributed] requirement (#726)
- *(deps)* Bump urllib3 from 1.26.18 to 2.2.3 (#724)
- *(deps)* Update sphinx-autodoc-typehints requirement (#723)
- *(deps)* Update pytest requirement from \<=8.3.2 to \<=8.3.3 (#722)
- *(deps)* Update pandas-stubs requirement (#721)
- *(deps)* Update sphinx-autodoc-typehints requirement (#719)
- *(deps)* Update ruff requirement from \<=0.6.3 to \<=0.6.4 (#718)
- *(deps)* Update zarr requirement from \<=2.18.2 to \<=2.18.3 (#717)
- *(deps)* Bump s3fs from 2024.6.1 to 2024.9.0 (#716)
- *(deps)* Update numpy requirement from \<=2.1.0 to \<=2.1.1 (#715)
- *(pre-commit)* Update hooks (#714)
- *(deps)* Update tifffile requirement from \<=2024.8.28 to \<=2024.8.30 (#713)
- *(deps)* Update jupyter requirement from \<=1.1.0 to \<=1.1.1 (#712)
- *(deps)* Update dask[distributed] requirement (#710)
- *(deps)* Update ruff requirement from \<=0.6.2 to \<=0.6.3 (#709)
- *(deps)* Update sphinx-autodoc-typehints requirement (#708)
- *(deps)* Update jupyter requirement from \<=1.0.0 to \<=1.1.0 (#707)
- *(deps)* Update tifffile requirement from \<=2024.8.24 to \<=2024.8.28 (#706)
- *(pre-commit)* Update hooks (#705)
- *(deps)* Update tifffile requirement from \<=2024.8.10 to \<=2024.8.24 (#704)
- *(deps)* Update mypy requirement from \<=1.11.1 to \<=1.11.2 (#703)
- *(deps)* Update ruff requirement from \<=0.6.1 to \<=0.6.2 (#702)
- *(deps)* Update scipy requirement from \<=1.14.0 to \<=1.14.1 (#701)
- *(deps)* Update xdoctest requirement from \<=1.1.6 to \<=1.2.0 (#700)
- *(pre-commit)* Update hooks (#699)
- *(deps)* Update ruff requirement from \<=0.6.0 to \<=0.6.1 (#698)
- *(deps)* Update dask[distributed] requirement (#697)
- *(deps)* Update numpy requirement from \<=2.0.1 to \<=2.1.0 (#696)
- *(deps)* Update ruff requirement from \<=0.5.7 to \<=0.6.0 (#695)
- *(deps)* Update nbsphinx requirement from \<=0.9.4 to \<=0.9.5 (#694)
- *(deps)* Update matplotlib requirement from \<=3.9.1.post1 to \<=3.9.2 (#693)
- *(pre-commit)* Update hooks (#692)
- *(deps)* Update commitizen requirement from \<=3.28.0 to \<=3.29.0 (#691)
- *(deps)* Update tifffile requirement from \<=2024.7.24 to \<=2024.8.10 (#690)
- *(deps)* Update ruff requirement from \<=0.5.6 to \<=0.5.7 (#689)
- *(deps)* Update pandas-stubs requirement (#688)
- *(deps)* Update dask[distributed] requirement (#687)
- *(deps)* Update matplotlib requirement from \<=3.9.1 to \<=3.9.1.post1 (#686)
- *(deps)* Update pandas-stubs requirement from \<=2.2.2.240603 to \<=2.2.2.240805 (#685)
- *(deps)* Update autodocsumm requirement from \<=0.2.12 to \<=0.2.13 (#684)
- *(pre-commit)* Update hooks (#683)
- *(deps)* Update ruff requirement from \<=0.5.5 to \<=0.5.6 (#682)
- *(deps)* Update coverage[toml] requirement from \<=7.6.0 to \<=7.6.1 (#681)
- *(deps)* Update xdoctest requirement from \<=1.1.5 to \<=1.1.6 (#680)
- *(deps)* Update mypy requirement from \<=1.11.0 to \<=1.11.1 (#679)
- *(deps)* Update sphinx requirement from \<=7.4.7 to \<=8.0.2 (#678)
- *(pre-commit)* Update hooks (#677)
- *(deps)* Update pre-commit requirement from \<=3.7.1 to \<=3.8.0 (#676)
- *(deps)* Bump pip from 24.1.2 to 24.2 in /.github/workflows (#675)
- *(deps)* Update ruff requirement from \<=0.5.4 to \<=0.5.5 (#673)
- *(deps)* Update pytest requirement from \<=8.3.1 to \<=8.3.2 (#674)
- *(deps)* Update tifffile requirement from \<=2024.7.21 to \<=2024.7.24 (#672)
- *(pre-commit)* Update hooks (#671)
- *(deps)* Update pypdf requirement from \<=4.3.0 to \<=4.3.1 (#670)
- *(deps)* Update sphinx requirement from \<=7.4.6 to \<=7.4.7 (#669)
- *(deps)* Update dask[distributed] requirement (#668)
- *(deps)* Update mypy requirement from \<=1.10.1 to \<=1.11.0 (#664)
- *(deps)* Update tifffile requirement from \<=2024.7.2 to \<=2024.7.21 (#667)
- *(deps)* Update ruff requirement from \<=0.5.3 to \<=0.5.4 (#666)
- *(deps)* Update numpy requirement from \<=2.0.0 to \<=2.0.1 (#665)
- *(deps)* Update pytest requirement from \<=8.2.2 to \<=8.3.1 (#663)
- *(deps)* Update ruff requirement from \<=0.5.2 to \<=0.5.3 (#661)
- *(deps)* Update sphinx requirement from \<=7.4.5 to \<=7.4.6 (#662)
- *(deps)* Update sphinx-autodoc-typehints requirement (#660)
- *(deps)* Update commitizen requirement from \<=3.27.0 to \<=3.28.0 (#659)
- *(deps)* Update pyarrow requirement from \<=16.1.0 to \<=17.0.0 (#656)
- *(deps)* Update sphinx requirement from \<=7.4.4 to \<=7.4.5 (#658)
- *(deps)* Bump jupyterlab-server from 2.27.2 to 2.27.3 (#657)
- *(pre-commit)* Update hooks (#654)
- *(deps)* Update sphinx requirement from \<=7.3.7 to \<=7.4.4 (#655)
- *(deps)* Update ruff requirement from \<=0.5.1 to \<=0.5.2 (#653)
- *(deps)* Update pypdf requirement from \<=4.2.0 to \<=4.3.0 (#652)
- *(deps)* Update coverage[toml] requirement from \<=7.5.4 to \<=7.6.0 (#651)
- *(pre-commit)* Update hooks (#650)
- *(deps)* Update ruff requirement from \<=0.5.0 to \<=0.5.1 (#648)
- *(deps)* Bump pip from 24.1.1 to 24.1.2 in /.github/workflows (#649)
- *(deps)* Update dask[distributed] requirement (#647)
- *(deps)* Update matplotlib requirement from \<=3.9.0 to \<=3.9.1 (#646)
- *(deps)* Update tifffile requirement from \<=2024.6.18 to \<=2024.7.2 (#645)
- *(deps)* Update ipykernel requirement from \<=6.29.4 to \<=6.29.5 (#644)
- *(pre-commit)* Update hooks (#643)
- *(deps)* Update ruff requirement from \<=0.4.10 to \<=0.5.0 (#640)
- *(deps)* Bump s3fs from 2023.6.0 to 2024.6.1 (#639)
- *(deps)* Bump pip from 24.1 to 24.1.1 in /.github/workflows (#638)
- *(deps)* Bump codecov/codecov-action from 4.3.1 to 4.5.0 (#637)
- *(deps)* Update pydata-sphinx-theme requirement (#636)
- Adopt the single ci.yml with automerge dependabot PR

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

- **deps**: update sphinx-autodoc-typehints requirement (#632)
- **deps**: update coverage[toml] requirement from \<=7.5.3 to \<=7.5.4 (#631)
- **deps**: update ruff requirement from \<=0.4.9 to \<=0.4.10 (#630)
- **deps**: update dask[distributed] requirement (#629)
- **deps**: bump pip from 24.0 to 24.1 in /.github/workflows (#628)
- **deps**: update dask[distributed] requirement (#627)
- **deps**: update sphinx-autodoc-typehints requirement (#626)
- **deps**: update tifffile requirement from \<=2024.5.22 to \<=2024.6.18 (#625)
- **deps**: update scikit-image requirement from \<=0.23.2 to \<=0.24.0 (#624)
- **pre-commit**: update hooks (#622)
- **deps**: update dask[distributed] requirement (#621)
- **deps**: update ruff requirement from \<=0.4.8 to \<=0.4.9 (#620)
- **deps**: update numpy requirement from \<=1.26.4 to \<=2.0.0 (#619)
- **deps**: bump codecov/codecov-action from 4.4.1 to 4.5.0 (#618)
- **pre-commit**: update hooks (#617)
- **deps**: update xdoctest requirement from \<=1.1.4 to \<=1.1.5 (#616)
- **deps**: bump pooch from 1.8.1 to 1.8.2 (#615)
- **deps**: update ruff requirement from \<=0.4.7 to \<=0.4.8 (#614)
- **deps**: update pytest requirement from \<=8.2.1 to \<=8.2.2 (#613)
- **deps**: update sphinx-autodoc-typehints requirement (#612)
- **deps**: update pandas-stubs requirement (#610)
- **pre-commit**: update hooks (#609)
- **deps**: update ruff requirement from \<=0.4.6 to \<=0.4.7 (#607)
- **deps**: update dask[distributed] requirement (#606)
- **deps**: update xdoctest requirement from \<=1.1.3 to \<=1.1.4 (#605)
- **deps**: bump hatch from 1.11.1 to 1.12.0 in /.github/workflows (#604)
- **deps**: update pydata-sphinx-theme requirement (#603)
- **deps**: update ruff requirement from \<=0.4.5 to \<=0.4.6 (#602)
- **deps**: update coverage[toml] requirement from \<=7.5.2 to \<=7.5.3 (#601)
- **pre-commit**: update hooks (#600)
- **deps**: update zarr requirement from \<=2.18.1 to \<=2.18.2 (#599)
- **deps**: update coverage[toml] requirement from \<=7.5.1 to \<=7.5.2 (#598)
- **deps**: update tifffile requirement from \<=2024.5.10 to \<=2024.5.22 (#597)
- **deps**: update scipy requirement from \<=1.13.0 to \<=1.13.1 (#596)
- **deps**: bump hatch from 1.11.0 to 1.11.1 in /.github/workflows (#595)
- **deps**: update commitizen requirement from \<=3.26.1 to \<=3.27.0 (#594)
- **deps**: update ruff requirement from \<=0.4.4 to \<=0.4.5 (#593)
- **deps**: bump jupyterlab-server from 2.27.1 to 2.27.2 (#592)
- **pre-commit**: update hooks (#589)
- **deps**: update commitizen requirement from \<=3.25.0 to \<=3.26.0 (#588)
- **deps**: update dask[distributed] requirement (#587)
- **deps**: update pytest requirement from \<=8.2.0 to \<=8.2.1 (#586)
- **deps**: update zarr requirement from \<=2.18.0 to \<=2.18.1 (#585)
- **deps**: update sphinx-click requirement from \<=5.2.2 to \<=6.0.0 (#583)
- **deps**: update matplotlib requirement from \<=3.8.4 to \<=3.9.0 (#582)
- **deps**: bump hatch from 1.10.0 to 1.11.0 in /.github/workflows (#581)
- **deps**: bump codecov/codecov-action from 4.3.1 to 4.4.0 (#580)
- **deps**: update pyarrow requirement from \<=16.0.0 to \<=16.1.0 (#579)
- **deps**: update sphinx-click requirement from \<=5.1.0 to \<=5.2.2 (#578)
- **deps**: update pandas-stubs requirement (#577)
- **pre-commit**: update hooks (#576)
- **deps**: update pre-commit requirement from \<=3.7.0 to \<=3.7.1 (#575)
- **deps**: update tifffile requirement from \<=2024.5.3 to \<=2024.5.10 (#574)
- **deps**: update ruff requirement from \<=0.4.3 to \<=0.4.4 (#573)

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

- **deps**: update zarr requirement from \<=2.17.2 to \<=2.18.0 (#572)
- **deps**: update nbsphinx requirement from \<=0.9.3 to \<=0.9.4 (#571)
- **pre-commit**: update hooks (#570)
- **deps**: update dask[distributed] requirement (#569)
- **deps**: update ruff requirement from \<=0.4.2 to \<=0.4.3 (#568)
- **deps**: update coverage[toml] requirement from \<=7.5.0 to \<=7.5.1 (#567)
- **deps**: update tifffile requirement from \<=2024.4.24 to \<=2024.5.3 (#566)
- **deps**: update pygments requirement from \<=2.17.2 to \<=2.18.0 (#565)
- **deps**: bump hatch from 1.9.7 to 1.10.0 in /.github/workflows (#564)
- **deps**: bump codecov/codecov-action from 4.3.0 to 4.3.1 (#563)
- **deps**: update commitizen requirement from \<=3.24.0 to \<=3.25.0 (#562)
- **pre-commit**: update hooks (#561)
- **deps**: update pytest requirement from \<=8.1.1 to \<=8.2.0 (#559)
- **deps**: update ruff requirement from \<=0.4.1 to \<=0.4.2 (#558)
- **deps**: update tifffile requirement from \<=2024.4.18 to \<=2024.4.24 (#557)
- **deps**: update mypy requirement from \<=1.9.0 to \<=1.10.0 (#556)
- **deps**: bump hatch from 1.9.4 to 1.9.7 in /.github/workflows (#555)
- **deps**: bump jupyterlab-server from 2.27.0 to 2.27.1 (#554)
- **deps**: update coverage[toml] requirement from \<=7.4.4 to \<=7.5.0 (#552)
- **deps**: bump questionary from 2.0.0 to 2.0.1 (#550)
- Pin few indirect deps to speed up pip
- **deps**: update jupyterlab-server requirement (#548)
- **deps**: update urllib3 requirement from \<=1.26.18 to \<=2.2.1 (#547)
- **deps**: update ipython requirement from \<=8.18.0 to \<=8.23.0 (#546)
- **deps**: update scikit-image requirement from \<=0.23.1 to \<=0.23.2 (#544)
- **deps**: update pyarrow requirement from \<=15.0.2 to \<=16.0.0 (#543)
- **deps**: update ruff requirement from \<=0.4.0 to \<=0.4.1 (#542)
- **deps**: update sphinx requirement from \<=7.3.6 to \<=7.3.7 (#541)
- **deps**: update dask[distributed] requirement (#540)
- Move aicsimageio into deps
- Drop types-setuptools
- Drop typeguard test
- Update to py3.12
- Correct mypy configuration
- **deps**: update ruff requirement from \<=0.3.7 to \<=0.4.0 (#539)
- **deps**: update tifffile requirement from \<=2024.2.12 to \<=2024.4.18 (#538)
- **deps**: update commitizen requirement from \<=3.22.0 to \<=3.24.0 (#537)
- **deps**: update sphinx requirement from \<=7.3.4 to \<=7.3.6 (#536)
- **deps**: update sphinx-autodoc-typehints requirement (#535)
- **deps**: update sphinx requirement from \<=7.2.6 to \<=7.3.4 (#533)
- **deps**: bump codecov/codecov-action from 4.2.0 to 4.3.0 (#524)
- **deps**: update types-setuptools requirement (#532)
- **deps**: update ruff requirement from \<=0.3.6 to \<=0.3.7 (#530)
- **deps**: update commitizen requirement from \<=3.21.3 to \<=3.22.0 (#529)
- **deps**: update ruff requirement from \<=0.3.5 to \<=0.3.6 (#528)
- **deps-dev**: update pandas requirement from \<=2.2.1 to \<=2.2.2 (#527)
- **deps-dev**: update sphinx-autodoc-typehints requirement (#526)
- **deps-dev**: update scikit-image requirement (#525)
- **deps-dev**: update zarr requirement from \<=2.17.1 to \<=2.17.2 (#522)
- **deps-dev**: update dask[distributed] requirement (#521)
- **deps**: bump codecov/codecov-action from 4.1.1 to 4.2.0 (#520)
- **deps-dev**: update matplotlib requirement from \<=3.8.3 to \<=3.8.4 (#519)
- **deps-dev**: update scipy requirement from \<=1.12.0 to \<=1.13.0 (#518)
- **deps-dev**: update ruff requirement from \<=0.3.4 to \<=0.3.5 (#517)
- **deps-dev**: update dask[distributed] requirement (#516)
- **deps-dev**: update commitizen requirement from \<=3.20.0 to \<=3.21.3 (#514)
- **deps**: bump actions/configure-pages from 4 to 5 (#513)

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

- **deps**: bump codecov/codecov-action from 4.1.0 to 4.1.1 (#511)
- **deps-dev**: update typeguard requirement from \<=4.1.5 to \<=4.2.1 (#509)
- **deps-dev**: update pre-commit requirement from \<=3.6.2 to \<=3.7.0 (#508)
- **deps-dev**: update ruff requirement from \<=0.3.3 to \<=0.3.4 (#507)
- **deps-dev**: update commitizen requirement from \<=3.18.4 to \<=3.20.0 (#506)
- **deps-dev**: update pyarrow requirement from \<=15.0.1 to \<=15.0.2 (#505)
- **deps-dev**: update pandas-stubs requirement (#503)
- **deps-dev**: update types-setuptools requirement (#502)
- **deps-dev**: update ruff requirement from \<=0.3.2 to \<=0.3.3 (#501)
- **deps-dev**: update dask[distributed] requirement (#500)
- **deps-dev**: update coverage[toml] requirement (#499)
- **deps-dev**: update commitizen requirement from \<=3.18.3 to \<=3.18.4 (#498)
- **deps**: bump hatch from 1.9.3 to 1.9.4 in /.github/workflows (#497)
- **deps-dev**: update dask[distributed] requirement (#496)
- **deps-dev**: update commitizen requirement from \<=3.18.0 to \<=3.18.3 (#494)
- **deps-dev**: update mypy requirement from \<=1.8.0 to \<=1.9.0 (#493)
- **deps-dev**: update pytest requirement from \<=8.1.0 to \<=8.1.1 (#492)
- **deps-dev**: update ruff requirement from \<=0.3.1 to \<=0.3.2 (#491)
- **deps-dev**: update types-setuptools requirement (#490)
- **deps-dev**: update types-setuptools requirement (#489)
- **deps-dev**: update commitizen requirement from \<=3.17.0 to \<=3.18.0 (#488)
- **deps-dev**: update pyarrow requirement from \<=15.0.0 to \<=15.0.1 (#487)
- **deps-dev**: update ruff requirement from \<=0.3.0 to \<=0.3.1 (#486)
- **deps-dev**: update zarr requirement from \<=2.17.0 to \<=2.17.1 (#485)
- **deps-dev**: update commitizen requirement from \<=3.16.0 to \<=3.17.0 (#484)
- **deps-dev**: update types-setuptools requirement (#483)
- **deps-dev**: update pytest requirement from \<=8.0.2 to \<=8.1.0 (#482)
- **deps-dev**: update types-setuptools requirement (#481)
- **deps-dev**: update ruff requirement from \<=0.2.2 to \<=0.3.0 (#480)
- **deps-dev**: update types-setuptools requirement (#479)
- **deps**: bump codecov/codecov-action from 4.0.2 to 4.1.0 (#478)
- **deps-dev**: update commitizen requirement from \<=3.15.0 to \<=3.16.0 (#477)
- **deps-dev**: update ipykernel requirement from \<=6.29.2 to \<=6.29.3 (#476)
- **deps-dev**: update pytest requirement from \<=8.0.1 to \<=8.0.2 (#473)
- **deps**: bump codecov/codecov-action from 4.0.1 to 4.0.2 (#475)
- **deps-dev**: update dask[distributed] requirement (#474)
- **deps-dev**: update pandas requirement from \<=2.2.0 to \<=2.2.1 (#472)
- **deps-dev**: update coverage[toml] requirement (#471)
- **deps-dev**: update types-setuptools requirement (#470)
- **deps-dev**: update coverage[toml] requirement (#469)
- **deps-dev**: update commitizen requirement from \<=3.14.1 to \<=3.15.0 (#464)
- **deps-dev**: update pytest requirement from \<=8.0.0 to \<=8.0.1 (#466)
- **deps-dev**: update ruff requirement from \<=0.2.1 to \<=0.2.2 (#465)
- **deps-dev**: update types-setuptools requirement (#463)
- **deps-dev**: update pre-commit requirement from \<=3.6.1 to \<=3.6.2 (#462)
- **deps-dev**: update pandas-stubs requirement (#461)
- **deps-dev**: update matplotlib requirement from \<=3.8.2 to \<=3.8.3 (#460)
- **deps-dev**: update zarr requirement from \<=2.16.1 to \<=2.17.0 (#459)
- **deps-dev**: update types-setuptools requirement (#458)
- **deps-dev**: update tifffile requirement (#457)
- **deps-dev**: update pre-commit requirement from \<=3.6.0 to \<=3.6.1 (#455)
- **deps-dev**: update dask[distributed] requirement (#454)
- **deps-dev**: update ipykernel requirement from \<=6.29.1 to \<=6.29.2 (#453)
- **deps-dev**: update sphinx-autodoc-typehints requirement (#452)
- **deps-dev**: update ipykernel requirement from \<=6.29.0 to \<=6.29.1 (#450)
- **deps-dev**: update numpy requirement from \<=1.26.3 to \<=1.26.4 (#451)
- **deps-dev**: update ruff requirement from \<=0.2.0 to \<=0.2.1 (#449)
- **deps**: bump pip from 23.3.2 to 24.0 in /.github/workflows (#445)
- **deps-dev**: update commitizen requirement from \<=3.14.0 to \<=3.14.1 (#444)
- **pre-commit**: Run autoupdate
- **pre-commit**: prettier v4.0.0-alpha.8
- **docs**: Drop darglint in favor of pydoclint; fix some docstring
- **deps-dev**: update tifffile requirement
- **deps**: bump codecov/codecov-action from 4.0.0 to 4.0.1 (#442)
- **deps-dev**: update commitizen requirement from \<=3.13.0 to \<=3.14.0 (#441)
- **deps-dev**: update ruff requirement from \<=0.1.15 to \<=0.2.0 (#440)
- **deps**: bump codecov/codecov-action from 3.1.5 to 4.0.0 (#439)
- **deps-dev**: update xdoctest requirement from \<=1.1.2 to \<=1.1.3 (#438)
- **deps-dev**: update ruff requirement from \<=0.1.14 to \<=0.1.15 (#435)
- **deps-dev**: update coverage[toml] requirement (#434)
- **deps-dev**: update dask[distributed] requirement (#433)
- **deps-dev**: update pytest requirement from \<=7.4.4 to \<=8.0.0 (#432)
- **deps-dev**: update sphinx-autodoc-typehints requirement (#431)
- **deps**: bump codecov/codecov-action from 3.1.4 to 3.1.5 (#430)
- **deps**: bump hatch from 1.9.2 to 1.9.3 in /.github/workflows (#429)
- **deps-dev**: update types-setuptools requirement (#428)
- **deps**: bump hatch from 1.9.1 to 1.9.2 in /.github/workflows (#427)
- **deps-dev**: update ruff requirement from \<=0.1.13 to \<=0.1.14 (#426)
- **deps-dev**: update pandas requirement from \<=2.1.4 to \<=2.2.0 (#425)
- **deps-dev**: update scipy requirement from \<=1.11.4 to \<=1.12.0 (#424)
- **deps-dev**: update pydata-sphinx-theme requirement (#423)
- **deps**: bump actions/cache from 3 to 4 (#422)
- **deps-dev**: update ipykernel requirement from \<=6.28.0 to \<=6.29.0 (#421)
- **deps-dev**: update ruff requirement from \<=0.1.12 to \<=0.1.13 (#420)
- **deps-dev**: update types-setuptools requirement (#419)
- **deps-dev**: update dask[distributed] requirement (#418)
- **deps-dev**: update ruff requirement from \<=0.1.11 to \<=0.1.12 (#417)
- **deps-dev**: update autodocsumm requirement (#416)
- **deps-dev**: update types-setuptools requirement (#415)
- **deps-dev**: update pydata-sphinx-theme requirement (#414)
- **deps-dev**: update ruff requirement from \<=0.1.9 to \<=0.1.11 (#413)
- **deps-dev**: update numpy requirement from \<=1.26.2 to \<=1.26.3 (#412)
- **deps-dev**: update pytest requirement from \<=7.4.3 to \<=7.4.4 (#411)
- **deps-dev**: update coverage[toml] requirement (#410)
- **deps-dev**: update pandas-stubs requirement (#409)
- **deps-dev**: update ipykernel requirement from \<=6.27.1 to \<=6.28.0 (#408)
- **deps**: bump hatch from 1.9.0 to 1.9.1 in /.github/workflows (#407)
- **deps-dev**: update ruff requirement from \<=0.1.8 to \<=0.1.9 (#406)
- **deps-dev**: update mypy requirement from \<=1.7.1 to \<=1.8.0 (#405)
- **deps-dev**: update coverage[toml] requirement (#404)
- **deps**: bump hatch from 1.8.1 to 1.9.0 in /.github/workflows (#400)
- **deps-dev**: update pandas-stubs requirement (#399)
- **deps**: bump pip from 23.3.1 to 23.3.2 in /.github/workflows (#398)
- **deps-dev**: update dask[distributed] requirement from \<=2023.12.0 to \<=2023.12.1 (#397)
- **deps-dev**: update coverage[toml] requirement (#396)
- **deps**: bump hatch from 1.8.0 to 1.8.1 in /.github/workflows (#395)
- **deps**: bump actions/download-artifact from 3 to 4 (#394)
- **deps-dev**: update ruff requirement from \<=0.1.7 to \<=0.1.8 (#393)
- **deps**: bump hatch from 1.7.0 to 1.8.0 in /.github/workflows (#392)
- **deps-dev**: update pandas requirement from \<=2.1.3 to \<=2.1.4 (#391)
- **deps-dev**: update pre-commit requirement from \<=3.5.0 to \<=3.6.0 (#390)
- **deps-dev**: update tifffile requirement (#389)
- **deps**: bump actions/setup-python from 4 to 5 (#388)
- **deps-dev**: update ruff requirement from \<=0.1.6 to \<=0.1.7 (#387)
- **deps-dev**: update ipykernel requirement from \<=6.27.0 to \<=6.27.1 (#386)
- **deps**: bump actions/configure-pages from 3 to 4 (#385)
- **deps**: bump actions/deploy-pages from 2 to 3 (#384)
- **deps-dev**: update dask[distributed] requirement (#382)
- **deps-dev**: bump commitizen from 3.12.0 to 3.13.0
- **deps-dev**: bump types-setuptools from 68.2.0.2 to 69.0.0.0 (#380)
- **deps-dev**: bump pydata-sphinx-theme from 0.14.3 to 0.14.4 (#379)
- **deps-dev**: bump mypy from 1.7.0 to 1.7.1 (#378)
- **deps-dev**: bump types-setuptools from 68.2.0.1 to 68.2.0.2 (#377)

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

- **deps-dev**: bump sphinx-click from 5.0.1 to 5.1.0 (#376)
- **deps-dev**: bump pygments from 2.17.1 to 2.17.2 (#375)
- **deps-dev**: update scipy requirement from \<1.11.4 to \<1.11.5 (#373)
- **deps-dev**: bump pygments from 2.16.1 to 2.17.1 (#372)
- **deps-dev**: bump matplotlib from 3.8.1 to 3.8.2 (#371)
- **deps-dev**: bump ruff from 0.1.5 to 0.1.6 (#370)
- **deps-dev**: bump pandas from 2.1.2 to 2.1.3 (#367)
- **deps-dev**: bump mypy from 1.6.1 to 1.7.0 (#368)
- **deps-dev**: update dask[distributed] requirement (#366)
- **deps-dev**: bump sphinx-autodoc-typehints from 1.25.0 to 1.25.2 (#365)
- **deps-dev**: update numpy requirement from \<1.26.2 to \<1.26.3 (#364)
- **deps-dev**: bump sphinx-autodoc-typehints from 1.24.1 to 1.25.0 (#363)
- **deps-dev**: update bokeh requirement from \<3.3.1 to \<3.3.2 (#362)
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
- **deps-dev**: update numpy requirement from \<1.26.1 to \<1.26.2 (#337)
- **deps-dev**: update dask[distributed] requirement (#336)
- **deps-dev**: bump pre-commit from 3.4.0 to 3.5.0 (#335)
- **deps-dev**: bump mypy from 1.5.1 to 1.6.0 (#334)
- **deps-dev**: update bokeh requirement from \<3.2.3 to \<3.3.1 (#333)
- **deps-dev**: bump scikit-image from 0.21.0 to 0.22.0
- **deps-dev**: bump coverage[toml] from 7.3.1 to 7.3.2 (#329)
- **deps-dev**: bump ruff from 0.0.291 to 0.0.292 (#328)
- **deps-dev**: update dask[distributed] requirement (#327)
- **deps-dev**: bump pandas-stubs from 2.0.3.230814 to 2.1.1.230928 (#326)
- **deps-dev**: update tifffile requirement (#325)
- **deps-dev**: update scipy requirement from \<1.11.3 to \<1.11.4 (#324)
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
- **deps-dev**: update numpy requirement from \<1.25.3 to \<1.26.1 (#314)
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
- **deps-dev**: update scipy requirement from \<1.11.2 to \<1.11.3 (#273)
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
- **deps-dev**: update bokeh requirement from \<3.2.2 to \<3.2.3 (#265)
- **deps-dev**: bump pandas-stubs from 2.0.2.230605 to 2.0.3.230814 (#264)
- **deps-dev**: update tifffile requirement from \<=2023.7.18 to \<=2023.8.12
- **deps-dev**: update tifffile requirement
- **deps-dev**: bump coverage[toml] from 7.2.7 to 7.3.0 (#262)
- **deps-dev**: bump mypy from 1.4.1 to 1.5.0 (#261)
- **deps-dev**: bump ruff from 0.0.283 to 0.0.284 (#260)
- **deps-dev**: bump ruff from 0.0.282 to 0.0.283 (#259)
- **deps-dev**: update dask[distributed] requirement from \<=2023.7.1 to \<=2023.8.0 (#257)
- **deps-dev**: bump pygments from 2.15.1 to 2.16.1 (#256)
- prettier --prose-wrap=preserve
- **deps-dev**: bump sphinx from 7.1.1 to 7.1.2 (#255)
- **deps-dev**: bump ruff from 0.0.281 to 0.0.282 (#254)
- **deps-dev**: bump ruff from 0.0.280 to 0.0.281 (#252)
- **deps-dev**: update numpy requirement from \<1.25.2 to \<1.25.3 (#251)
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
- Works with python < 3.7.

## Version 0.3

- Transition to Pyscaffold template for faster dev cycles.

## Version 0.2.3

- Works for clophensor data.
- Heavy on memory.
- Flat and Dark not tested.
