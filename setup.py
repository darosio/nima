#!/usr/bin/env python
# -*- coding: utf-8 -*-


try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read().replace('.. :changelog:', '')

requirements = [
    # TODO: put package requirements here
    'docopt',
    'numpy',
    'scipy',
    'matplotlib',
    'pandas',
    'scikit-image',
    'tifffile'
]

test_requirements = [
    # TODO: put package test requirements here
    'pytest-cov'
]

setup(
    name='nimg',
    version='0.2.3',
    description="Image analysis scripts based on scipy.ndimage and skimage.",
    long_description=readme + '\n\n' + history,
    author="Daniele Arosio",
    author_email='danielepietroarosio@gmail.com',
    url='https://github.com/darosio/nimg',
    packages=[
        'nimg',
    ],
    package_dir={'nimg':
                 'nimg'},
    include_package_data=True,
    install_requires=requirements,
    # TODO: remove or add any other script
    entry_points={ 'console_scripts': [
                    'nimg = nimg.scripts:main',
                   ],
    },
    license="new-bsd",
    zip_safe=False,
    keywords='nimg',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD 3-Clause (BSD New)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    extras_require={
        'dev': ['check-manifest', 'watchdog', 'bumpversion'],
        'test': ['coverage', 'pytest-cov', 'tox', 'flake8'],
        'doc': ['sphinxcontrib-plantuml', 'numpydoc']
    }
)
