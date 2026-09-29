# Modernization Assessment — `nima` v0.11.9

*Generated 2026-09-27 (metrics refreshed 2026-09-28) by `/modernize-assess` against the repository root
(`/home/dan/workspace/nima`, commit `6ac51dc`). No `legacy/` directory
existed, so the current repo was assessed in place.*

## Executive Summary

`nima` is a small, actively maintained Python library and Click CLI for
ratiometric fluorescence imaging. It covers bias, dark and flat calibration,
background estimation, cell segmentation and pH/Cl⁻ ratio measurement, built
on xarray/dask, scikit-image and scipy. It has about 1.8 KSLOC of product code
(3.0 KSLOC of Python including tests and tooling; `scc`).

The stack is already modern: Python 3.12–3.14, uv, ruff, strict mypy (clean)
and a 3×3 OS×Python CI matrix. Line coverage is about 85 %, from a stale local
`coverage.xml`. **The risk is not technological obsolescence.** It is
correctness debt in the CLI layer (several untested crash paths and one
data-destroying default) and an unfinished migration from dict-of-channels to
channel-as-dimension.

**Recommendation: Refactor in place on the same stack.** Fix the confirmed
bugs first, then finish the channel-as-dimension migration and retire the
legacy NumPy helpers. A rewrite or re-architecture is not warranted.

## System Inventory

**Tooling.** Figures come from `scc` 4.0.0, `cloc` (`--vcs=git`) and `lizard`
1.24.0, updated 2026-09-28. They were measured on the working tree, which
includes the two bug fixes of that date (about +10 SLOC in `__main__.py`); the
untracked `analysis/` directory is excluded. All three tools count docstrings
as comments, so product SLOC is about 1.8 K rather than the 2.4 K from the
earlier `wc`-based fallback, which counted docstrings as code.

**Complexity counters disagree.** For `measure`, lizard gives CCN 18, the
earlier stdlib-`ast` counter gave 17, and ruff's McCabe check (the one
enforced in CI, limit 12) gives 11. lizard also counts `and`/`or` and
comprehensions, which ruff's McCabe count ignores. Use lizard's numbers for
ranking and ruff's for the CI limit.

### Lines of code (`scc .`, tracked and untracked-but-not-ignored files)

| Language                                   |  Files |      Lines |    Blanks |  Comments |      Code | scc complexity |
| ------------------------------------------ | -----: | ---------: | --------: | --------: | --------: | -------------: |
| Python                                     |     20 |      4,972 |       660 |     1,336 |     2,976 |            348 |
| Jupyter (docs)                             |      4 |      4,191 |         0 |         0 |     4,191 |              0 |
| Markdown                                   |      3 |      1,099 |       315 |         0 |       784 |              0 |
| YAML                                       |      7 |        509 |        48 |        16 |       445 |              0 |
| TOML                                       |      4 |        336 |        26 |        16 |       294 |              8 |
| reST                                       |      9 |        249 |        73 |         0 |       176 |              0 |
| Other (JSON, Makefile, CSV, bat, css, txt) |     11 |        325 |        40 |        20 |       265 |              8 |
| **Total**                                  | **58** | **11,681** | **1,162** | **1,388** | **9,131** |        **364** |

`cloc --vcs=git` agrees for Python (2,969 code lines). It reports the
notebooks as 1,115 code lines plus 3,076 comment lines, because it treats
JSON/markdown cells differently from `scc`.

| Python scope                                    | Lines |       Code (SLOC) |
| ----------------------------------------------- | ----: | ----------------: |
| `src/nima/` (8 files; `scc` / `cloc`)           | 3,101 | **1,816 / 1,815** |
| All Python (src + tests + benchmarks + scripts) | 4,972 |             2,976 |

Per file (`scc --by-file -s complexity src`):

| File                       | Lines | Code | scc complexity |
| -------------------------- | ----: | ---: | -------------: |
| `src/nima/nima.py`         | 1,179 |  647 |             97 |
| `src/nima/__main__.py`     |   666 |  445 |             70 |
| `src/nima/segmentation.py` |   690 |  401 |             55 |
| `src/nima/generat.py`      |   306 |  199 |             26 |
| `src/nima/utils.py`        |   177 |   96 |             16 |
| `src/nima/io.py`           |    53 |   15 |              1 |
| `src/nima/nima_types.py`   |    29 |   13 |              0 |

### Complexity (`lizard src/nima`)

69 functions, 1,815 NLOC. Mean CCN is 4.0 and the maximum is 18. With its
default threshold (CCN > 15), lizard warns on 3 functions: `measure`, `main`
and `run_simulation`. `run_simulation` is flagged for length, not CCN.

| CCN | NLOC | Length | Params | Function                                      |
| --: | ---: | -----: | -----: | --------------------------------------------- |
|  18 |   83 |    157 |      3 | `src/nima/nima.py:681` `measure`              |
|  15 |   98 |    129 |     24 | `src/nima/__main__.py:157` `main`             |
|  13 |   95 |    139 |      6 | `src/nima/generat.py:168` `run_simulation`    |
|  12 |   20 |     46 |      4 | `src/nima/nima.py:179` `shading`              |
|  12 |   43 |     83 |      2 | `src/nima/segmentation.py:359` `calculate_bg` |
|   9 |   36 |     75 |      2 | `src/nima/__main__.py:373` `bias`             |
|   9 |   46 |     69 |      3 | `src/nima/nima.py:840` `plot_meas`            |
|   8 |   36 |     67 |      4 | `src/nima/nima.py:42` `plot_img`              |
|   8 |   38 |     74 |      4 | `src/nima/nima.py:227` `bg`                   |

`main` takes 24 parameters, one per Click option. That is the main driver of
its size.

The largest module is `nima.py` (1,179 lines, scc complexity 97). It is close
to a god module: it holds I/O helpers, preprocessing, segmentation,
measurement and four plotting functions.

### Technology fingerprint

| Aspect             | Evidence                                                                                                                                                 |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Language/runtime   | Python `>=3.12` (`pyproject.toml`); dev pin `3.14` (`.python-version`); CI matrix 3.12/3.13/3.14 × Linux/macOS/Windows (`.github/workflows/ci.yml`)      |
| Build              | hatchling; uv with `uv.lock`; `Makefile` targets (lint/test/type/xdoc/docs/bump)                                                                         |
| Core libs          | xarray 2026.7 (transitive), dask/dask-image, scikit-image ≥0.26, scipy ≥1.18, pandas ≥3.0.6, matplotlib ≥3.11, click ≥8.3, sigfig, tifffile (transitive) |
| I/O                | External `nima-io<=0.4.2` → bioio (+ bioio-bioformats → JVM, Maven-fetched JARs for non-TIFF) (`src/nima/io.py`)                                         |
| Data stores        | None (file-based: TIFF in, TIFF/CSV/PDF/PNG out)                                                                                                         |
| Integration points | CLIs `nima` and `bima` (`src/nima/__main__.py`); Python API; Read the Docs + GitHub Pages docs with executed notebooks                                   |
| Quality gates      | ruff (preview, near-all rule families), mypy strict (**clean: 0 errors**), pre-commit, xdoctest, codecov                                                 |
| Tests              | 6 test modules, ~1.35 k lines, golden-output CLI tests (`tests/data/output/`); coverage ≈ 84.5 % lines (stale local `coverage.xml`, Feb 2026)            |
| Automation         | Renovate, weekly lockfile update, Cruft template sync, tag-triggered PyPI release                                                                        |
| History            | First commit 2015-10-28; 1,616 commits                                                                                                                   |

**Dependency hygiene:**

- `s3fs` is a declared runtime dependency but is never imported.
- `pyarrow` is never imported. It is plausibly used implicitly by pandas 3 as
  the string backend, so keep it and add a comment saying why.
- `xarray`, `dask`, `numpy` and `tifffile` are imported directly but not
  declared. They arrive only transitively.
- `nima-io<=0.4.2` has an upper cap only.

## Architecture-at-a-Glance

The full diagram is in [`ARCHITECTURE.mmd`](ARCHITECTURE.mmd).

| #   | Domain                                | Files                                        | Key entry points                                                              |
| --- | ------------------------------------- | -------------------------------------------- | ----------------------------------------------------------------------------- |
| D1  | Image I/O & types                     | `io.py`, `nima_types.py`, ext. `nima_io`     | `read_image` io.py:19 → lazy `DataArray` TCZYX                                |
| D2  | Calibration CLI (`bima`) & hot pixels | `__main__.py`, `nima.py`                     | `bias` :374, `dark` :449, `mflat` :491, `flat` :544; `hotpixels` nima.py:1078 |
| D3  | Preprocessing & shading               | `nima.py`                                    | `median` :111, `shading` :179                                                 |
| D4  | Background estimation (xarray/dask)   | `segmentation.py`, `nima.py`                 | `BgParams` seg:84, `calculate_bg` seg:359, `nima.bg` :227                     |
| D5  | Legacy NumPy bg & ratio utils         | `segmentation.py`, `utils.py`                | `calculate_bg_iteratively` seg:560, `utils.ratio_df` :106                     |
| D6  | Segmentation & labelling              | `nima.py`                                    | `segment` :340, `process_watershed` :493                                      |
| D7  | Measurement & ratio imaging           | `nima.py`                                    | `ratio` :588, `measure` :681                                                  |
| D8  | Visualisation                         | `nima.py`, `__main__.py`, `segmentation.py`  | `plot_img` :42, `plot_meas` :840, `plt_img_profile` :911                      |
| D9  | Ratio pipeline CLI (`nima`)           | `__main__.py`                                | `main` :157 → `output_results` :289                                           |
| D10 | Synthetic data / bg benchmarking      | `generat.py`, `simulate_bg.py`               | `run_simulation` :168                                                         |
| D11 | Dev/QA umbrella                       | `tests/`, `benchmarks/`, `docs/`, `scripts/` | benchmark not wired into CI                                                   |

**The channel migration is incomplete.** README.md claims "Channels are
dimensions (C), not keys in a dictionary". That holds in the xarray core (D3,
D4 strategies, D6). It does not hold in these places:

- `nima.bg` returns a channel-keyed DataFrame plus a channel-keyed dict. It
  builds a `(T,C)` DataArray at nima.py:291 but discards it.
- `measure` returns `dict[str, …]`.
- `output_results` hard-codes `["C","G","R"]` (`__main__.py:315`, marked `FIXME`).
- `plot_meas` derives colours from the channel letters.
- All of `utils.py` is positional or dict-based.
- The `bima` path uses `tifffile` directly and positional channels.

**Dangling code:**

- Never used: `Kwargs` (nima.py:35), `AXES_LENGTH_3D` (nima.py:38),
  `_VerbosityLevel.MEDIUM/HIGH`. Extra `-v` flags therefore have no effect.
- `io.read_image(stitch_tiles=True)` has no caller.
- `utils.ave`, `utils.channel_mean`, `utils.ratio_df` and
  `utils.mask_all_channels` are used only by tests.

## Production Runtime Profile

**No telemetry is available.** `nima` is a locally run research tool with no
APM, job logs or runtime exports. The closest thing is
`benchmarks/benchmark_dask.py` and `docs/references/dask_performance.md`. The
benchmark is not run in CI, and its documented pixel count does not match the
script (see Documentation Gaps). There is no p50/p95/p99 data, so no domain
can be flagged on runtime variance.

The static-analysis concern that stands in for it is D4/D7:

- `calculate_bg` calls `.compute()` on the whole image and makes figures for
  every frame (segmentation.py:427-439). This defeats dask laziness.
- `measure` looks up labels with a scan of labels × timepoints × regions
  (nima.py:757-772).

## Technical Debt

These are the top 10 findings, ranked by remediation value. **Bold** marks
confirmed defects. Findings 1 and 5 were reproduced by running code. The type
mismatch behind finding 3 was confirmed by running click's path conversion;
the resulting crash follows from reading the code. The rest are from reading.

1. **`nima --bg-downscale` always crashes.**
   - Cause: `__main__.py:224-232` puts `"downscale"` into `BgParams(**kwargs)`,
     which fails with `TypeError: unexpected keyword argument 'downscale'`.
     No test passes this option.
   - Also at :231: the `if value` filter silently drops legitimate `0`s, so
     `--bg-percentile 0` becomes 10. That is a silent scientific error.
   - Also at :233: a debug `print`.
   - Also: `inverse_yen` is missing from the `--bg-method` choices.
   - Also: the `--min-size` help says the default is 2000; it is actually 640.
1. **`utils.ratio_df` raises `KeyError` on real data.**
   - `channel_mean` writes `str` column keys (utils.py:100), but `ratio_df`
     reads `int` keys (:126-127).
   - The test monkeypatches `channel_mean` (tests/test_utils.py:116-119), so
     the bug is hidden.
1. **`bima dark`, `bima flat` and `bima plot` crash without `-o`.**
   - These commands use `click.Path()` without `path_type=Path`, so click
     passes a `str` (__main__.py:444,448,488,541,543,619).
   - The code then calls `.with_suffix()` on it (:468,565,632).
   - Every test passes `-o`.
   - **Fix finding 5 in the security list (SEC-007) before this one.**
     Otherwise `flat` will start overwriting its input.
1. **Channel names C/G/R are hard-coded and inconsistent.**
   - `output_results` hard-codes the list (__main__.py:315,331-342).
   - Default order is `G,R,C` in the CLI (:206) but `C,G,R` in the library
     (nima.py:343,684).
   - Any other channel names crash output.
1. **`BgParams` changes value on copy and is mutated in place.**
   - `__post_init__` divides `perc` by 100 (segmentation.py:127). As a result,
     `dataclasses.replace(BgParams(perc=10)).perc == 0.001` (confirmed).
   - `calculate_bg` writes `adaptive_radius` back into the caller's object
     (seg:384-387). Because `nima.bg` reuses the object, the first frame's
     radius is fixed for every later channel and timepoint.
1. **NumPy and dask code paths are duplicated.**
   - Each background strategy has a NumPy and a dask twin
     (segmentation.py:146-355). There is also dead code at :185-189.
   - `utils.bg` (utils.py:51-74) duplicates `segmentation.fit_gaussian`
     (seg:531-556).
   - `plt_img_profile` and `plt_img_profile_2` duplicate each other
     (nima.py:911-1075).
1. **Redundant and non-scaling computation.**
   - Ratio images are computed in `measure` (nima.py:785-798) and again in
     `main` (__main__.py:268-273).
   - `measure` has the quadratic label lookup mentioned above.
   - `calculate_bg` computes eagerly and keeps every figure.
1. **`print` in library code, and blind exception handling.**
   - `print` calls: segmentation.py:631,650,685; generat.py:164,304;
     __main__.py:233,483.
   - `T20` (the ruff print rule) is in both `select` and `ignore`, so it is off.
   - `generat.safe_call` swallows every exception (generat.py:161-165).
   - `geometric_mean_filter` mutates its input (seg:679).
1. **Stale or contradictory tool configuration.**
   - In `pyproject.toml`:
     - ruff `target-version = "py311"`, but `requires-python >=3.12`.
     - A leftover `[tool.isort]` (black profile).
     - Both `TC` and its deprecated alias `TCH` are selected, plus `AIR`.
     - Coverage `omit` names a non-existent `types.py`.
     - `fix = true` together with `unsafe-fixes = true`.
     - No pytest `filterwarnings = ["error"]`.
   - The pre-commit ruff (v0.15.13) lags the lock (0.16.6). This produces 43
     RUF105 findings locally that pre-commit/CI do not report.
   - `types-click` (click 7 stubs) overrides click 8's own types. That forces
     3 `type: ignore[type-var]`.
1. **Dependency declarations and magic numbers.**
   - Unused `s3fs` (and `selenium` in `docs`); undeclared direct imports.
   - The flat path uses an unexplained `+ 20` and `sigma=100`, marked `FIXME`
     (__main__.py:605-609).
   - `dark_thr = 4.5`.
   - `ave` clamps (20/10) and ignores its `bgmax` argument (utils.py:79-88).
   - Percentiles 18.4/81.6 are commented as "66.6 %" but span 63.2 %
     (nima.py:985-987).

**Other defects:**

- `img_hist` uses `if vmin is None: … elif vmax is None:` (nima.py:984-987).
  When `vmin` is None, `vmax` is never set. Verified by reading.
- `correct_hotpixel` handles borders wrongly: index −1 wraps around, and the
  last row or column raises `IndexError` (nima.py:1168-1178).
- The `plt_img_profiles` title grows on each loop iteration (__main__.py:652).
- Leftover refactor narration comments ("Wait, the original code…",
  nima.py:626-641).
- `nima_types.py` still describes itself as the "clophfit package".

**Suppression density.** 43 `# noqa` (30 in src) and 72 `# type: ignore[...]`
(68 in src; 43 of them `no-untyped-call`, mostly skimage/dask). Every
`type: ignore` carries an error code.

## Security Findings

Credential inventory in SECRETS.local.md (gitignored; not for sharing). It
contains a single public Codecov badge token (`README.md:6`, `OU6F****`). No
real secrets were found in tracked files or git history.

The threat model is a locally run research tool, so the realistic risks are
the supply chain (CI/release), crafted or oversized input files, and loss of
raw data. No scanner was available (`pip-audit`/`osv-scanner`/`bandit` are not
installed). Dependency status was reasoned from `uv.lock`, and many 2026-dated
versions could not be checked for CVEs.

| ID      | CWE          | Sev        | Location                                                                                      | Issue                                                                                                                                                                                                                                             | Fix                                                                                              |
| ------- | ------------ | ---------- | --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| SEC-007 | CWE-73       | **Medium** | `__main__.py:421,433-437` (`bias`); `:565,603,612` (`flat`)                                   | **`bima bias stack.tiff` without `-o` overwrites the raw input stack with its median frame** (default output `stack.png` → `.with_suffix(".tiff")`). `flat` has the same flaw but is currently masked by the SEC-011 crash. Confirmed by reading. | Default to `f"{stem}_bias.tiff"`; refuse when output resolves to input; add `--overwrite`        |
| SEC-001 | CWE-829      | Medium     | `.github/workflows/release.yml:16,42,51`; all `uses:`                                         | Actions pinned by tag (and `@git-cliff`), not SHA. The release job has `contents: write` and checkout persists credentials.                                                                                                                       | Pin to SHAs; `persist-credentials: false`; re-enable Renovate `github-actions` with `pinDigests` |
| SEC-002 | CWE-522      | Medium     | `release.yml:36-38`                                                                           | Long-lived `PYPI_TOKEN`, no protected environment                                                                                                                                                                                                 | PyPI Trusted Publishing (OIDC) + `environment: pypi` with reviewers                              |
| SEC-003 | CWE-494      | Medium     | `ci.yml:141-170`; `cruft-update.yml`; `lockfile-update.yml`; `.cruft.json` (`checkout: null`) | PRs labelled `dependencies` or `update-template` are auto-merged once CI is green, with no human review. Cruft output (which can change workflows via `PAT_WORKFLOWS`) is included.                                                               | Require review; exclude template and workflow changes from auto-merge; pin the template ref      |
| SEC-004 | CWE-250      | Low        | `ci.yml` (no top-level `permissions:`)                                                        | Default token scope                                                                                                                                                                                                                               | `permissions: contents: read`                                                                    |
| SEC-005 | CWE-522      | Low        | `cruft-update.yml:52,56`; `lockfile-update.yml:41,45`                                         | Long-lived PAT with `workflow` scope embedded in the remote URL                                                                                                                                                                                   | GitHub App token; `gh auth setup-git`                                                            |
| SEC-006 | CWE-494      | Low        | `io.py:53` → nima_io → bioio-bioformats (`ome:formats-gpl:RELEASE`)                           | Bio-Formats JARs fetched from Maven at a floating version at runtime; bypasses lock hashes and hurts reproducibility                                                                                                                              | Pin `BIOFORMATS_VERSION`; make bioformats optional                                               |
| SEC-008 | CWE-409      | Low        | `__main__.py:389-391`                                                                         | Zip member fully read into memory; an empty zip raises `IndexError`                                                                                                                                                                               | Size check; stream with `zf.open`; `BadParameter`                                                |
| SEC-009 | CWE-789/400  | Low        | `__main__.py:510-514,520,564`; `--ratio-median-radii`                                         | Eager `tifffile.imread` of every file; unbounded filter radii                                                                                                                                                                                     | Lazy `aszarr`; size guard; `click.IntRange`                                                      |
| SEC-010 | CWE-20 / 369 | Low        | `__main__.py:124-150,231,247,255-257`; `nima.shading`                                         | No range checks; truthiness drops `0`; bad radii string raises an unhandled error; zeros in the flat give `inf` in ratios; a single `-f`/`-d` is silently ignored                                                                                 | `FloatRange`/`IntRange`; `is not None`; mask `flat<=0`; `UsageError`                             |
| SEC-011 | CWE-704      | Low        | `__main__.py:444,448,488,543,619`                                                             | `str` paths crash without `-o` (= debt #3)                                                                                                                                                                                                        | `PATH_IN` everywhere, **after** SEC-007                                                          |
| SEC-012 | CWE-1104     | Low        | `pyproject.toml` (`s3fs`, `selenium`, `nima-io<=0.4.2`)                                       | Unused network stack (aiohttp/botocore) in every install; the cap blocks patch releases                                                                                                                                                           | Remove or move to an extra; `nima-io>=0.4.2,<0.5`                                                |
| SEC-013 | CWE-295      | Low        | `.envrc:2-5`                                                                                  | `GOINSECURE=…` disables Go TLS checks in the developer shell (irrelevant to this project)                                                                                                                                                         | Delete the lines                                                                                 |
| SEC-014 | CWE-200      | Info       | `docs/tutorials/*.ipynb`                                                                      | Internal absolute data paths (`/home/dati/...`) published                                                                                                                                                                                         | Relative paths or sample-data fetchers                                                           |
| SEC-015 | CWE-1395     | Info       | `.readthedocs.yml:17-21`                                                                      | RTD installs `.[docs]` without the lock                                                                                                                                                                                                           | `uv export --frozen` in `build.jobs`                                                             |

**Checked and clean:** no `subprocess`, `eval`/`exec`, `pickle`,
`yaml.load` or `allow_pickle`; no zip-slip; no `pull_request_target`; no
expression injection in the workflows; notebooks stripped of outputs.

## Documentation Gaps

These are the top 5 behaviours a new engineer would need explained.

1. **What each background-estimation method does and when to use it.** There
   are six strategies: `arcsinh`, `entropy`, `adaptive`, `li_adaptive`,
   `li_li` and `inverse_yen` (segmentation.py:389-396). They have many coupled
   parameters (`radius`, `adaptive_radius`, `perc` stored as a fraction,
   `arcsinh_perc`, `erosion_disk`, `clip`). No narrative doc compares them, and
   `inverse_yen`, `erosion_disk` and `clip` are not reachable from the CLI.
   `generat.run_simulation` is the de facto benchmark but is undocumented as
   such.

1. **The calibration model and its magic constants.** Nothing explains the
   physical meaning of these numbers:

   - the `+ 20` pedestal and `sigma=100` in `_output_flat` (`FIXME`);
   - `dark_thr = 4.5`;
   - the hot-pixel `n_sd = 20`;
   - the `ave` clamps.

   `bima mflat` and `bima plot` are absent from the README, and the README's
   `bima dark` deprecation note is still unresolved.

1. **The output contract of the `nima` CLI.** Nothing documents:

   - the directory layout (`nima-<ver>/<basename>/`);
   - the CSV schemas (`bg.csv`, `label*.csv` columns);
   - the ratio TIFFs;
   - the fact that channel names must be exactly {C, G, R}.

   The README describes `nima <TIFFSTK> CHANNELS` with default `["G","R","C"]`,
   which disagrees with the library defaults.

1. **The data model and channel convention.** It is undocumented which APIs
   return a DataArray and which return channel-keyed dicts or DataFrames
   (`bg`, `measure`), and that `utils.py` is a legacy positional API. The
   README TODO still shows the removed `DIm = dict[str, ...]`, and
   `nima_types.py` documents a non-existent `ImArray`.

1. **Stale or incorrect reference docs.**

   - `docs/references/nima.rst` promises a UML diagram that is not there.
   - `dask_performance.md` says the median filter uses
     `scipy.ndimage.median_filter`; the dask path uses `dask_image`.
   - It also claims about 1.2 billion pixels, but the benchmark uses about
     126 M.
   - The README contributing links point to ClopHfit.
   - `segment` says it returns a mask and labels; it returns labels only.

## Relative Scale

| Scope                            | KSLOC (`scc` code) | COCOMO-II index = 2.94 × KSLOC^1.10 |
| -------------------------------- | -----------------: | ----------------------------------: |
| Product (`src/nima`)             |              1.816 |                            **5.66** |
| All Python (incl. tests/tooling) |              2.976 |                                9.76 |

The inputs are `scc` code lines (docstrings counted as comments) with nominal
scale factors. The earlier `wc`-based figures (2.449 → 7.87) are superseded.
`scc`'s own COCOMO output uses the basic-COCOMO organic model (cost and
schedule). It is not reported here, as instructed.

**This is a relative size and complexity measure for ranking this system
against others. It is not an estimate of how long modernization will take or
what it will cost.** COCOMO assumes traditional human-team productivity, which
agentic transformation does not follow. By this index `nima` is a very small
system.

## Recommended Modernization Pattern

**Refactor, in place, on the same stack.**

`nima` already runs on the current Python and scientific-Python toolchain with
strict typing and multi-platform CI. There is nothing to rehost, replatform,
re-architect or rebuild, and the domain logic is specialised and valuable.
The work needed is corrective and consolidating:

1. Fix the confirmed defects with regression tests, in this order:
   - SEC-007 (data loss);
   - the `--bg-downscale` crash;
   - the `-o`/`str` path crashes;
   - `ratio_df` keys;
   - `BgParams` mutation and `perc`;
   - `img_hist`.
1. Harden the CI and release chain: SHA pins, Trusted Publishing, and no
   unreviewed auto-merge.
1. Finish the channel-as-dimension migration:
   - `bg` returns the `(T,C)` DataArray;
   - `measure` and `output_results` derive columns from `channels`;
   - retire `utils.py` or re-express it in xarray.
1. Collapse the NumPy/dask duplication behind single kernels with
   `apply_ufunc`/`map_overlap`.
1. Split `nima.py` along the domain lines above (preprocessing, segmentation,
   measurement, plotting).

**Routing.** The template maps Refactor to `/modernize-uplift`. That command
is built for version-to-version deltas, and `nima` has no pending runtime or
framework version bump, so it fits poorly. A better route:

- `/modernize-harden` for the SEC items, which produces a reviewable patch;
- then ordinary TDD-driven refactoring for items 1 and 3–5.

`/modernize-transform` and `/modernize-reimagine` are not recommended.

**Open questions for the maintainer:**

- What do the `+20`/`sigma=100` in the flat path represent physically?
- Should `bima dark` be removed?
- Are `utils.ave`, `channel_mean` and `ratio_df` still used by external
  workflows?
- Is `s3fs` needed by anyone, or can it go?
- Is `inverse_yen` meant to be excluded from the CLI?
