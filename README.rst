.. image:: https://github.com/darosio/nimg/actions/workflows/tests.yml/badge.svg
   :target: https://github.com/darosio/nimg/actions/workflows/tests.yml

.. image:: https://codecov.io/gh/darosio/nimg/branch/main/graph/badge.svg?token=OR0LUZUJUR
   :target: https://codecov.io/gh/darosio/nimg

.. image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
   :target: https://github.com/pre-commit/pre-commit
   :alt: pre-commit
A library and cli for image analysis based on scipy.ndimage and scikit-image.


Features
--------
- easy dark and flat correction
- automatic cell segmentation
- easy ratio analyses


Installation
------------

    $ pyenv virtualenv 3.6.13 nimg-0.3.1-py36
    $ poetry install
    $ pip install .

Optionally:
    $ python -m ipykernel install --user --name="nimg0.3.1"

    # Jedi not working
    %config Completer.use_jedi = False
    for python >= 3.7 should not be needed because ipython >= 7.20 will be used.


Usage
-----

To use nimg in a project::

    from nimg import nimg



.. click:: nimg.__main__:main
    :prog: nimg
    :nested: full

.. click:: nimg.__main__:bias
    :prog: bias
    :nested: full



Description
===========

A longer description of your project goes here...


Note
====

poetry rocks?
development
my idea is to use global flake8 and black and no need to track linting and safety in poetry. KISS.

pyenv activate nimg-…
poetry install
pre-commit install
before next first commit:
pre-commit run --all-files

pyenv activare nimg-0.2
poetry install
pip install .
so it is not installed in development mode and this version will persist to updates.

todo
====
- restore sane complexity value
- CI and static typing
