"""
Perform test automation with nox.

For further details, see https://nox.thea.codes/en/stable/#

"""

import hashlib
import os
from pathlib import Path

import nox
import yaml


#: Default to reusing any pre-existing nox environments.
nox.options.reuse_existing_virtualenvs = True

#: Name of the package to test.
PACKAGE = "esmf_regrid"

#: Cirrus-CI environment variable hook.
PY_VER = os.environ.get("PY_VER", ["3.6", "3.7", "3.8"])

#: Cirrus-CI environment variable hook.
COVERAGE = os.environ.get("COVERAGE", False)

#: Cirrus-CI environment variable hook.
IRIS_SOURCE = os.environ.get("IRIS_SOURCE", None)

COVERAGE_PACKAGES = ["pytest-cov", "codecov"]
IRIS_GITHUB = "https://github.com/scitools/iris.git"


def _cache_venv(session, fname):
    """
    Cache the nox session environment.

    This consists of saving a hexdigest (sha256) of the associated
    conda requirements YAML file.

    Parameters
    ----------
    session: object
        A `nox.sessions.Session` object.
    fname: str
        Requirements filename that defines this environment.

    """
    python_version = session.python.replace(".", "")
    with open(fname, "rb") as fi:
        hexdigest = hashlib.sha256(fi.read()).hexdigest()
    tmp_dir = Path(session.create_tmp())
    cache = tmp_dir / f"py{python_version}.sha"
    with open(cache, "w") as fo:
        fo.write(hexdigest)


def _combine_requirements(primary, secondary, ignore=None):
    """
    Combine the conda environment YAML files together into one.

    Parameters
    ----------
    primary: str
        The filename of the primary YAML conda environment.
    secondary: str
        The filename of the subordinate YAML conda environment.
    ignore: str, optional
        The prefix of any package name to be ignored from the
        combined dependency requirements.

    Returns
    -------
    dict
        A dictionary of the combined YAML conda environments.

    """
    with open(primary, "r") as fi:
        result = yaml.load(fi, Loader=yaml.FullLoader)
    with open(secondary, "r") as fi:
        secondary = yaml.load(fi, Loader=yaml.FullLoader)
    # Combine the channels and dependencies only.
    for key in ["channels", "dependencies"]:
        result[key] = sorted(set(result[key]).union(secondary[key]))
    if ignore:
        # Filter out any specific prefixed package dependencies.
        result["dependencies"] = [
            spec for spec in result["dependencies"] if not spec.startswith(ignore)
        ]
    return result


def _get_iris_github_artifact(session):
    """
    Determine whether an Iris source artifact from GitHub is required.

    This can be an Iris branch name, commit sha or tag name.

    Parameters
    ----------
    session: object
        A `nox.sessions.Session` object.

    Returns
    -------
    str
        The Iris GitHub artifact.

    """
    result = IRIS_SOURCE
    # The CLI overrides the environment variable.
    for arg in session.posargs:
        if arg.startswith("--iris="):
            parts = arg.split("=")
            if len(parts) == 2:
                result = parts[1].strip()
                break
    if result:
        parts = result.strip().split(":")
        result = None
        if len(parts) == 2:
            repo, artifact = parts
            if repo.startswith("'") or repo.startswith('"'):
                repo = repo[1:]
            if repo.lower() == "github":
                result = artifact
                if result.endswith("'") or result.endswith('"'):
                    result = result[:-1]
    return result


def _venv_cached(session, fname):
    """
    Determine whether the nox session environment has been cached.

    Parameters
    ----------
    session: object
        A `nox.sessions.Session` object.
    fname: str
        Requirements filename that defines this environment.

    Returns
    -------
    bool
        Whether the session has been cached.

    """
    result = False
    tmp_dir = Path(session.create_tmp())
    python_version = session.python.replace(".", "")
    cache = tmp_dir / f"py{python_version}.sha"
    if cache.is_file():
        with open(fname, "rb") as fi:
            hexdigest = hashlib.sha256(fi.read()).hexdigest()
        with open(cache, "r") as fi:
            cached = fi.read()
        result = cached == hexdigest
    return result


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
    artifact = _get_iris_github_artifact(session)
    python_version = session.python.replace(".", "")
    requirements_fname = f"requirements/py{python_version}.yml"

    if artifact:
        tmp_dir = Path(session.create_tmp())
        artifact_dir = tmp_dir / "iris"
        cwd = Path.cwd()
        if not artifact_dir.is_dir():
            session.run("git", "clone", IRIS_GITHUB, str(artifact_dir), external=True)
        session.cd(artifact_dir)
        session.run("git", "fetch", "origin", external=True)
        session.run("git", "checkout", artifact, external=True)
        session.cd(str(cwd))
        iris_requirements_fname = (
            f"{artifact_dir}/requirements/ci/py{python_version}.yml"
        )
        requirements_yaml = _combine_requirements(
            requirements_fname, iris_requirements_fname, ignore="iris"
        )
        requirements_fname = tmp_dir / "requirements.yml"
        with open(requirements_fname, "w") as fo:
            yaml.dump(requirements_yaml, fo)

    # Install the package requirements.
    if not _venv_cached(session, requirements_fname):
        # Back-door approach to force nox to use "conda env update".
        command = (
            "conda",
            "env",
            "update",
            f"--prefix={session.virtualenv.location}",
            f"--file={requirements_fname}",
            "--prune",
        )
        session._run(*command, silent=True, external="error")
        _cache_venv(session, requirements_fname)

    if artifact:
        # Install the iris source in develop mode.
        session.install("--no-deps", "--editable", str(artifact_dir))

    # Install the esmf-regrid source in develop mode.
    session.install("--no-deps", "--editable", ".")

    # Determine whether verbose diagnostics have been requested from the command line.
    verbose = "-v" in session.posargs or "--verbose" in session.posargs

    if verbose:
        session.run("conda", "info")
        session.run("conda", "list", f"--prefix={session.virtualenv.location}")
        session.run(
            "conda",
            "list",
            f"--prefix={session.virtualenv.location}",
            "--explicit",
        )

    if COVERAGE:
        # Execute the tests with code coverage.
        session.conda_install("--channel=conda-forge", *COVERAGE_PACKAGES)
        session.run("pytest", "--cov-report=xml", "--cov")
        session.run("codecov")
    else:
        # Execute the tests.
        session.run("pytest")
