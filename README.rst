..
   .. image:: https://img.shields.io/pypi/v/clophfit.svg
           :target: https://pypi.python.org/pypi/clophfit


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

todo
====
pre-commit
CI and static typing
