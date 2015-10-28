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
    'numpy', 'scipy', 'pandas', 'matplotlib', 'scikit-image'
]

test_requirements = [
    # TODO: put package test requirements here
]

setup(
    name='nimg',
    version='0.1.0',
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
    entry_points={ 'console_scripts': [
                    'dark = nimg.dark:main',
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
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
    ],
    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    extras_require={
        'dev': ['check-manifest'],
        'test': ['coverage', 'pytest-cov'],
    }
)
