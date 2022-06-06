"""Nox sessions."""
import nox
import nox_poetry
from nox_poetry.sessions import Session

package = "nimg"
locations = "src", "tests", "./noxfile.py", "docs/conf.py"
nox.options.sessions = "lint", "tests", "docs"


@nox_poetry.session(python="3.10")
def lint(session: Session) -> None:
    """Lint using flake8."""
    args = session.posargs or locations
    session.install(
        "flake8",
        "flake8-black",
        "flake8-bugbear",
        "flake8-bandit",
        "flake8-docstrings",
        "darglint",
        "flake8-import-order",
    )
    session.run("flake8", *args)
    # TODO: other linters session.run("rst-lint", "README.rst")  # for PyPI readme.rst


@nox_poetry.session(python=["3.8", "3.9", "3.10"])
def tests(session: Session) -> None:
    """Run the test suite."""
    args = session.posargs or ["--cov", "-v"]
    # session.install("coverage[toml]", "pytest", "pytest-cov", ".")
    session.install(
        "coverage[toml]", "pytest", "pytest-cov", "xdoctest", "pygments", "."
    )
    session.run("pytest", *args)
    session.run("python", "-m", "xdoctest", package)


# @nox_poetry.session(python=["3.8", "3.9", "3.10"])
# def xdoctest(session: Session) -> None:
#     """Run examples with xdoctest."""
#     args = session.posargs or ["all"]
#     session.install("xdoctest", "pygments", ".")
#     session.run("python", "-m", "xdoctest", package, *args)


@nox_poetry.session(python="3.10")
def docs(session: Session) -> None:
    """Build the documentation."""
    session.install(
        "sphinx",
        "myst-parser",
        ".",
    )
    session.run("sphinx-build", "docs", "docs/_build")
