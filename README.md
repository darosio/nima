# NImA

[![PyPI](https://img.shields.io/pypi/v/nima.svg)](https://pypi.org/project/nima/)
[![CI](https://github.com/darosio/nima/actions/workflows/ci.yml/badge.svg)](https://github.com/darosio/nima/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/darosio/nima/branch/main/graph/badge.svg?token=OU6F9VFUQ6)](https://codecov.io/gh/darosio/nima)
[![RtD](https://readthedocs.org/projects/nima/badge/)](https://nima.readthedocs.io/)

A library and a command-line interface (CLI) designed to assist with image
analysis tasks using scipy.ndimage and scikit-image.

- Version: “0.10.2”

## Features

- Bias and Flat Correction
- Automatic Cell Segmentation
- Multi-Ratio Ratiometric Imaging, enabling users to analyze multiple ratios
  with ease.

## Installation

You can get the library directly from [PyPI](https://pypi.org/project/nima/)
using `pip`:

```
pip install nima
```

Alternatively, you can use [pipx](https://pypa.github.io/pipx/) to install it in
an isolated environment:

```
pipx install nima
```

To enable auto completion for the `nima` command, follow these steps:

1. Generate the completion script by running the following command:

   ```
   _CLOP_COMPLETE=bash_source nima > ~/.local/bin/nima-complete.bash
   ```

1. Source the generated completion script to enable auto completion:

   ```
   source ~/.local/bin/nima-complete.bash
   ```

## Usage

### Library

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

To estimate the system dark (multi-channel) frame:

```
bima dark <FPATH>
```

Replace \<FPATH> with the paths to the dark stack (Light Off - Long acquisition time).

Note: The estimation of the system dark may be removed in future versions
because it risks being redundant with the flat estimation. It is likely to be
removed soon.

To estimate the system flat (multi-channel) frame:

```
bima flat --bias <BIAS_PATH> <FPATH>
```

Replace \<FPATH> with the path to the tf8 stack and \<BIAS_PATH> with the path to
the bias image.

## TODO

- jaxtyping

```
ImFrame: TypeAlias = Float32[Array, "height width"]  # noqa: F722
ImSequence: TypeAlias = Float32[Array, "time height width"]  # noqa: F722
DIm: TypeAlias = dict[str, ImSequence]
```

## Contributing

Contributions to the project are welcome!

If you are interested in contributing to the project, please read our
[contributing](https://darosio.github.io/ClopHfit/references/contributing.html)
and [development
environment](https://darosio.github.io/ClopHfit/references/development.html)
guides, which outline the guidelines and conventions that we follow for
contributing code, documentation, and other resources.

## License

We use a shared copyright model that enables all contributors to maintain the
copyright on their contributions - see the [revised BSD license](LICENSE.txt)
for details.

## Acknowledgments

Special thanks to the developers of scipy.ndimage and scikit-image for their
invaluable contributions to image processing in Python.
