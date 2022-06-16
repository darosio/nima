"""Nox sessions."""
import sys
from pathlib import Path

import nox
import nox_poetry
from nox_poetry.sessions import Session


package = "nima"
locations = "src", "tests", "./noxfile.py", "docs/conf.py"
python_versions = ["3.8", "3.9", "3.10"]
nox.options.sessions = "pre-commit", "safety", "mypy", "tests", "typeguard", "docs"


@nox_poetry.session(name="pre-commit", python=python_versions[-1])
def precommit(session: Session) -> None:
    """Lint using pre-commit."""
    args = session.posargs or [
        "run",
        "--all-files",
        "--hook-stage=manual",
        "--show-diff-on-failure",
    ]
    session.install(
        "black",
        "darglint",
        "flake8",
        "flake8-bandit",
        "flake8-bugbear",
        "flake8-docstrings",
        "flake8-eradicate",
        "flake8-rst-docstrings",
        "flake8-pytest-style",
        "flake8-comprehensions",
        "isort",
        "pep8-naming",
        "pre-commit",
        "pre-commit-hooks",
        "pyupgrade",
    )
    # TODO: other linters session.run("rst-lint", "README.rst")  # for PyPI readme.rst
    session.run("pre-commit", *args)


@nox_poetry.session(python=python_versions[-1])
def safety(session: Session) -> None:
    """Scan dependencies for insecure packages."""
    requirements = session.poetry.export_requirements()
    session.install("safety")
    session.run("safety", "check", "--full-report", f"--file={requirements}")


@nox_poetry.session(python=python_versions)
def mypy(session: Session) -> None:
    """Type-check using mypy."""
    args = session.posargs or ["src", "tests", "docs/conf.py"]
    session.run(
        "rm", "-rf", ".mypy_cache/", external=True
    )  # for types-jinja2 from pyparser
    session.install(".")
    session.install("mypy", "pytest", "data-science-types", "types-setuptools")
    session.run("mypy", *args)
    if not session.posargs:
        session.run("mypy", f"--python-executable={sys.executable}", "./noxfile.py")


@nox_poetry.session(python=python_versions)
def tests(session: Session) -> None:
    """Run the test suite and xdoctest."""
    session.install("coverage[toml]", "pytest", "pygments", "xdoctest", ".")
    try:
        session.run("coverage", "run", "--parallel", "-m", "pytest", *session.posargs)
    finally:
        if session.interactive:
            session.notify("coverage", posargs=[])
    session.run("python", "-m", "xdoctest", package)


@nox_poetry.session(python=python_versions[-1])
def coverage(session: Session) -> None:
    """Produce the coverage report."""
    args = session.posargs or ["report"]

    session.install("coverage[toml]")

    if not session.posargs and any(Path().glob(".coverage.*")):
        session.run("coverage", "combine")

    session.run("coverage", *args)


@nox_poetry.session(python=python_versions[-1])
def typeguard(session: Session) -> None:
    """Runtime type checking using Typeguard."""
    session.install(".")
    session.install("pytest", "typeguard", "pygments")
    session.run("pytest", f"--typeguard-packages={package}", *session.posargs)


@nox_poetry.session(python=python_versions[-1])
def docs(session: Session) -> None:
    """Build the documentation."""
    session.install(
        "sphinx",
        "sphinx-click",
        "pydata_sphinx_theme",
        "myst-parser",
        ".",
    )
    session.run("sphinx-build", "docs", "docs/_build")


@nox_poetry.session(python=python_versions[-1])
def clean(session: Session) -> None:
    """Clean local repository."""
    session.run(
        "rm",
        "-rf",
        ".coverage" "./README.tmp.html",
        "./__pycache__",
        "./.nox",
        "./.mypy_cache",
        "./.pytest_cache",
        "./docs/_build",
        "./src/" + package + "/__pycache__",
        "./tests/__pycache__",
        "./dist",
        external=True,
    )
