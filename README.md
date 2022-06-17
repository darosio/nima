[![PyPI](https://img.shields.io/pypi/v/nima.svg)][pypi status]
[![Python Version](https://img.shields.io/pypi/pyversions/nima)][pypi status]
[![image](https://github.com/darosio/nima/actions/workflows/tests.yml/badge.svg)](https://github.com/darosio/nima/actions/workflows/tests.yml)
[![image](https://codecov.io/gh/darosio/nima/branch/main/graph/badge.svg?token=OR0LUZUJUR)](https://codecov.io/gh/darosio/nima)

[pypi status]: https://pypi.org/project/nima/

A library and a cli to help image analyses based on scipy.ndimage and
scikit-image.

# Features

-   easy dark and flat correction
-   automatic cell segmentation
-   easy ratio analyses

# Installation

:

	pyenv virtualenv 3.10.4 nima-0.5.4
	pyenv activate
	poetry install
	pip install .

Optionally:

	python -m ipykernel install --user --name="nima0.3.1"


# Usage

To use nima in a project:

    from nima import nima

See documentation for the `nima` command line.

## Description

A longer description of your project goes here\...

## Note

pyenv activate nima-...
poetry install pre-commit
install before next first commit: pre-commit run --all-files

pyenv activate nima-0.2
poetry install
pip install .

so it is not installed in development mode and this version will persist to updates.

## todo

- restore sane complexity value (< 21).
