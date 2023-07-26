# NImA

[![PyPI](https://img.shields.io/pypi/v/nima.svg)](https://pypi.org/project/nima/)
[![CI](https://github.com/darosio/nima/actions/workflows/ci.yml/badge.svg)](https://github.com/darosio/nima/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/darosio/nima/branch/main/graph/badge.svg?token=OU6F9VFUQ6)](https://codecov.io/gh/darosio/nima)
[![RtD](https://readthedocs.org/projects/nima/badge/)](https://nima.readthedocs.io/)

A library and a cli to help image analyses based on scipy.ndimage and
scikit-image.

- Version: “0.6.0”

## Features

- easy dark and flat correction
- automatic cell segmentation
- easy ratio analyses

## Installation

You can get the library directly from [PyPI](https://pypi.org/project/nima/)
using `pip`:

    pip install nima

Alternatively, you can use [pipx](https://pypa.github.io/pipx/) to install it in
an isolated environment:

    pipx install nima

To enable auto completion for the `nima` command, follow these steps:

1.  Generate the completion script by running the following command:

        _CLOP_COMPLETE=bash_source nima > ~/.local/bin/nima-complete.bash

2.  Source the generated completion script to enable auto completion:

        source ~/.local/bin/nima-complete.bash

## Usage

To use nima in a project:

    from nima import nima

See documentation for the `nima` command line.

## Description

A longer description of your project goes here\...

## Note

    pyenv activate nima-...
    poetry install pre-commit

install before next first commit: pre-commit run --all-files

    nox --session=pre-commit -- install
    and activate poetry


    pyenv activate nima-0.2
    poetry install
    pip install .

so it is not installed in development mode and this version will persist to
updates.

## todo

- restore sane complexity value (< 21).
