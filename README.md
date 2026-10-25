# NImA

[![PyPI](https://img.shields.io/pypi/v/nima.svg)](https://pypi.org/project/nima/)
[![CI](https://github.com/darosio/nima/actions/workflows/ci.yml/badge.svg)](https://github.com/darosio/nima/actions/workflows/ci.yml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/darosio/nima/main.svg)](https://results.pre-commit.ci/latest/github/darosio/nima/main)
[![codecov](https://codecov.io/gh/darosio/nima/branch/main/graph/badge.svg?token=OU6F9VFUQ6)](https://codecov.io/gh/darosio/nima)
[![RtD](https://readthedocs.org/projects/nima/badge/)](https://nima.readthedocs.io/)
[![](https://img.shields.io/badge/Pages-blue?logo=github)](https://darosio.github.io/nima/)

<!-- [![](https://img.shields.io/badge/docs-GitHub%20Pages-blue?logo=github)](https://darosio.github.io/nima/) -->

A library and a command-line interface (CLI) designed to assist with image
analysis tasks using scipy.ndimage and scikit-image.

## Features

- Modern IO:
  - Replaced legacy `tifffile` parsing with `nima_io` for standardized, plugin-based support.
  - Returns `xarray.DataArray` with standardized `TCZYX` dimensions and lazy Dask loading.
  - Channels are dimensions (`C`), not keys in a dictionary.
  - Pure Python implementation (no JVM required for basic TIFF).
- Bias and Flat Correction
- Automatic Cell Segmentation
- Background Estimation Simulation Tools
- Multi-Ratio Ratiometric Imaging, enabling users to analyze multiple ratios
  with ease.

## Installation

From PyPI with pip:

```
pip install nima
```

Or isolate with pipx:

```
pipx install nima
```

Shell completion (Click/Typer):

- Bash:

  ```
  _NIMA_COMPLETE=bash_source nima > ~/.local/bin/nima-complete.bash
  source ~/.local/bin/nima-complete.bash
  ```

- Fish:

```bash
  _NIMA_COMPLETE=fish_source nima | source
```

## Usage

Docs: https://nima.readthedocs.io/

To use nima in your python code, import it as follows:

```
from nima import nima, generat, utils
```

### Command-Line Interface (CLI)

The CLI for this project provides two main commands: `nima` and `bima`. You can
find detailed usage information and examples in the
[documentation](https://nima.readthedocs.io/en/latest/click.html). Here are some
examples of how to use each command:

#### nima

The `nima` command is used to perform multi-ratio ratiometric imaging analyses
on multi-channel TIFF time-lapse stacks.

To perform multi-ratio ratiometric imaging analyses on a multichannel TIFF
time-lapse stack, use the following command:

```
nima <TIFFSTK> CHANNELS
```

Replace \<TIFFSTK> with the path to the TIFF time-lapse stack file, and `CHANNELS`
with the channel names. By default, the channels are set to ["G", "R", "C"].

#### bima

The `bima` command is used to compute bias, dark, and flat corrections.

To estimate the detector bias frame:

```
bima bias <FPATH>
```

Replace \<FPATH> with the paths to the bias stack (Light Off - 0 acquisition time).
Without `-o`, outputs are written next to the input as `<stem>_bias.tiff`,
`<stem>_bias.png` and `<stem>_bias.csv` (hot pixels); an output path that
would overwrite the input stack is rejected.

To estimate the system dark (multi-channel) frame:

```
bima dark <FPATH>
```

Replace \<FPATH> with the paths to the dark stack (Light Off - Long acquisition time).
Without `-o`, outputs are `<stem>_dark.tiff` and `<stem>_dark.png`.

Note: The estimation of the system dark may be removed in future versions
because it risks being redundant with the flat estimation. It is likely to be
removed soon.

To estimate the system flat (multi-channel) frame:

```
bima flat --bias <BIAS_PATH> <FPATH>
```

Replace \<FPATH> with the path to the tf8 stack and \<BIAS_PATH> with the path to
the bias image.
Without `-o`, outputs are `<stem>_flat.tiff`, `<stem>_flat-raw.tiff` and
`<stem>_flat.png`; `bima mflat "<glob>"` writes `<glob stem>_flat.*` in the
current directory. Any output path that would overwrite an input is rejected.

## TODO

- jaxtyping

```
ImFrame: TypeAlias = Float32[Array, "height width"]  # noqa: F722
ImSequence: TypeAlias = Float32[Array, "time height width"]  # noqa: F722
DIm: TypeAlias = dict[str, ImSequence]
```

## Dependency updates (Renovate)

We use Renovate to keep dependencies current.

Enable Renovate:

1. Install the GitHub App: https://github.com/apps/renovate (Settings → Integrations → GitHub Apps → Configure → select this repo/org).
1. This repo includes a `renovate.json` policy. Renovate will open a “Dependency Dashboard” issue and PRs accordingly.

Notes:

- Commit style: `build(deps): bump <dep> from <old> to <new>`
- Pre-commit hooks are grouped and labeled.

Python upgrades are **not** managed by Renovate. Bumping `.python-version`
on its own would make local dev and the release workflow use an interpreter
that CI never tests. Upgrade Python in a single manual PR that updates
`.python-version`, the `requires-python`/classifier/ruff settings in
`pyproject.toml`, the CI matrix in `.github/workflows/ci.yml` and the version
in `.readthedocs.yml`, then refresh `uv.lock`.

Migrating from Dependabot:

- You may keep “Dependabot alerts” ON for vulnerability visibility, but disable Dependabot security PRs.

## Template updates (Cruft)

This project is linked to its Cookiecutter template with Cruft.

- Check for updates: `cruft check`
- Apply updates: `cruft update -y` (resolve conflicts, then commit)

CI runs a weekly job to open a PR when template updates are available.

First-time setup if you didn’t generate with Cruft:

```bash
pipx install cruft  # or: pip install --user cruft
cruft link --checkout main https://github.com/darosio/cookiecutter-python.git
```

Notes:

- The CI workflow skips if `.cruft.json` is absent.
- If you maintain a stable template branch (e.g., `v1`), link with `--checkout v1`. You can also update within that line using `cruft update -y --checkout v1`.

### Enabling the CI push (GitHub App, one-time per repo)

The `cruft-update` and `lockfile-update` workflows need write access to push
their update branches and open PRs, and `github.token` alone cannot push
changes under `.github/workflows/*`. Rather than a personal access token
(which expires and must be rotated), this template uses a GitHub App
installation token via
[`actions/create-github-app-token`](https://github.com/actions/create-github-app-token).
Both workflows fall back to `github.token` automatically if the App isn't
configured; the `cruft-update` fallback fails whenever Cruft touches workflow
files.

One-time setup (shared across all your repos once the App exists):

1. Create the App (skip if you already have one for this purpose):
   [github.com/settings/apps/new](https://github.com/settings/apps/new) —
   Repository permissions: `Contents: Read & write`,
   `Pull requests: Read & write`, `Workflows: Read & write`. Webhook: inactive.

1. Generate a private key on the App's settings page and download the `.pem`.

1. Install the App on this repository (or "All repositories" on your account
   to cover future repos too).

1. Set two **repo-level** secrets (GitHub Apps on personal accounts don't
   support org-wide shared secrets, so this step is repeated per repo):

   ```bash
   gh secret set APP_ID --repo <owner>/<repo> --body "<app-id>"
   gh secret set APP_PRIVATE_KEY --repo <owner>/<repo> < /path/to/key.pem
   ```

## License

We use a shared copyright model that enables all contributors to maintain the
copyright on their contributions.

All code is licensed under the terms of the [revised BSD license](LICENSE.txt).

## Contributing

Contributions to the project are welcome!

If you are interested in contributing to the project, please read our
[contributing](https://darosio.github.io/ClopHfit/references/contributing.html)
and [development
environment](https://darosio.github.io/ClopHfit/references/development.html)
guides, which outline the guidelines and conventions that we follow for
contributing code, documentation, and other resources.
