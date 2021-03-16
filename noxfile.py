"""Nox sessions."""

import tempfile

import nox

package = "nimg"
locations = "src", "tests", "noxfile.py", "docs/conf.py"


@nox.session(python="3.9")
def lint(session):
    """Lint using flake8."""
    args = session.posargs or locations
    session.install(
        "flake8",
        "flake8-bandit",
        "flake8-black",
        "flake8-bugbear",
        "flake8-docstrings",
        "darglint",
        "flake8-import-order",
    )
    session.run("flake8", *args)


@nox.session(python="3.9")
def safety(session):
    """Scan dependencies for insecure packages."""
    with tempfile.NamedTemporaryFile() as requirements:
        session.run(
            "poetry",
            "export",
            "--dev",
            "--format=requirements.txt",
            "--without-hashes",
            f"--output={requirements.name}",
            external=True,
        )
        session.install("safety")
        session.run("safety", "check", f"--file={requirements.name}", "--full-report")


@nox.session(python=["3.6", "3.7", "3.8", "3.9"])
def tests(session):
    """Run the test suite."""
    args = session.posargs or ["--cov"]
    session.run("poetry", "install", external=True)
    session.run("pytest", *args)


@nox.session(python=["3.6", "3.7", "3.8", "3.9"])
def xdoctest(session):
    """Run examples with xdoctest."""
    args = session.posargs or ["all"]
    session.run("poetry", "install", external=True)
    session.run("python", "-m", "xdoctest", package, *args)


@nox.session(python="3.9")
def docs(session):
    """Build the documentation."""
    # session.install("sphinx")
    session.run("poetry", "install", external=True)
    session.run("sphinx-build", "docs", "docs/_build")
