"""
Perform test automation with nox.

For further details, see https://nox.thea.codes/en/stable/#

"""

import os

import nox


#: Default to reusing any pre-existing nox environments.
nox.options.reuse_existing_virtualenvs = True

#: Name of the package to test.
PACKAGE = "esmf_regrid"

#: Cirrus-CI environment variable hook.
PY_VER = os.environ.get("PY_VER", "3.8")

#: Cirrus-CI environment variable hook.
COVERAGE = os.environ.get("COVERAGE", False)


@nox.session
def lint(session):
    """
    Perform linting of the code-base.

    Parameters
    ----------
    session: object
        A `nox.sessions.Session` object.

    """
    # Pip install the session requirements.
    session.install("flake8", "flake8-docstrings", "flake8-import-order")
    # Execute the flake8 linter on the package.
    session.run("flake8", PACKAGE)
    # Execute the flake8 linter on this file.
    session.run("flake8", __file__)


@nox.session
def style(session):
    """
    Perform format checking of the code-base.

    Parameters
    ----------
    session: object
        A `nox.sessions.Session` object.

    """
    # Pip install the session requirements.
    session.install("black==20.8b1")
    # Execute the black format checker on the package.
    session.run("black", "--check", PACKAGE)
    # Execute the black format checker on this file.
    session.run("black", "--check", __file__)


@nox.session(python=[PY_VER], venv_backend="conda")
def tests(session):
    """
    Support for conda in nox is relatively new and maturing.

    Parameters
    ----------
    session: object
        A `nox.sessions.Session` object.

    Notes
    -----
    See
      - https://github.com/theacodes/nox/issues/346
      - https://github.com/theacodes/nox/issues/260

    """
    # Determine the conda requirements yaml file.
    fname = f"requirements/py{PY_VER.replace('.', '')}.yml"
    # Back-door approach to force nox to use "conda env update".
    command = (
        "conda",
        "env",
        "update",
        f"--prefix={session.virtualenv.location}",
        f"--file={fname}",
        "--prune",
    )
    session._run(*command, silent=True, external="error")
    if COVERAGE:
        # Execute the tests with code coverage.
        session.conda_install("--channel=conda-forge", *COVERAGE.split())
        session.run("pytest", "--cov-report=xml", "--cov")
        session.run("codecov")
    else:
        # Execute the tests.
        session.run("pytest")
