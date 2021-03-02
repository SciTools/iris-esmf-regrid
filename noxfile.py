"""
Perform test automation with nox.

For further details, see https://nox.thea.codes/en/stable/#

"""

import hashlib
import os
from pathlib import Path

import nox


#: Default to reusing any pre-existing nox environments.
nox.options.reuse_existing_virtualenvs = True

#: Name of the package to test.
PACKAGE = "esmf_regrid"

#: Cirrus-CI environment variable hook.
PY_VER = os.environ.get("PY_VER", ["3.6", "3.7", "3.8"])

#: Cirrus-CI environment variable hook.
COVERAGE_PACKAGES = os.environ.get("COVERAGE_PACKAGES", False)


def venv_cached(session):
    """
    Determine whether the nox session environment has been cached.

    Parameters
    ----------
    session: object
        A `nox.sessions.Session` object.

    Returns
    -------
    bool
        Whether the session has been cached.

    """
    result = False
    yml = Path(f"requirements/py{session.python.replace('.', '')}.yml")
    tmp_dir = Path(session.create_tmp())
    cache = tmp_dir / yml.name
    if cache.is_file():
        with open(yml, "rb") as fi:
            expected = hashlib.sha256(fi.read()).hexdigest()
        with open(cache, "r") as fi:
            actual = fi.read()
        result = actual == expected
    return result


def cache_venv(session):
    """
    Cache the nox session environment.

    This consists of saving a hexdigest (sha256) of the associated
    conda requirements YAML file.

    Parameters
    ----------
    session: object
        A `nox.sessions.Session` object.

    """
    yml = Path(f"requirements/py{session.python.replace('.', '')}.yml")
    with open(yml, "rb") as fi:
        hexdigest = hashlib.sha256(fi.read()).hexdigest()
    tmp_dir = Path(session.create_tmp())
    cache = tmp_dir / yml.name
    with open(cache, "w") as fo:
        fo.write(hexdigest)


@nox.session
def flake8(session):
    """
    Perform flake8 linting of the code-base.

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
def black(session):
    """
    Perform black format checking of the code-base.

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


@nox.session(python=PY_VER, venv_backend="conda")
def tests(session):
    """
    Perform esmf-regrid integration and unit tests.

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
    if not venv_cached(session):
        # Determine the conda requirements yaml file.
        fname = f"requirements/py{session.python.replace('.', '')}.yml"
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
        cache_venv(session)

    if COVERAGE_PACKAGES:
        # Execute the tests with code coverage.
        session.conda_install("--channel=conda-forge", *COVERAGE_PACKAGES.split())
        session.run("pytest", "--cov-report=xml", "--cov")
        session.run("codecov")
    else:
        # Execute the tests.
        session.run("pytest")
