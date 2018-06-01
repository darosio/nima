===============================
nimg
===============================

..
   .. image:: https://img.shields.io/travis/darosio/nimg.svg
           :target: https://travis-ci.org/darosio/nimg

   .. image:: https://img.shields.io/pypi/v/nimg.svg
           :target: https://pypi.python.org/pypi/nimg


A library and cli for image analysis based on scipy.ndimage and scikit-image.


Features
--------
- easy dark and flat correction
- automatic cell segmentation
- easy ratio analyses


Installation
------------

    $ mkvirtualenv nimg
    $ pip install nimg

Optionally:

    $ pip install -r dev-requirements.txt


Usage
-----

To use nimg in a project::

    from nimg import nimg




Description
===========

A longer description of your project goes here...


Note
====

putup nimg/ --force --no-skeleton -p nimg -l new-bsd

This project has been set up using PyScaffold 3.0.3. For details and usage
information on PyScaffold see http://pyscaffold.org/.
