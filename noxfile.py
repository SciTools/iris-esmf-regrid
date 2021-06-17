"""
Perform test automation with nox.

For further details, see https://nox.thea.codes/en/stable/#

"""

import os
from pathlib import Path
import shutil
from urllib.request import urlopen

import nox
from nox.logger import logger
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
LOCKFILE_PLATFORM = "linux-64"


def _lockfile_path(py_string: str, platform_placeholder: bool = False) -> Path:
    """
    Return a constructed lockfile path for the relevant python string e.g ``py38``.

    Optionally retain the ``{platform}`` placeholder to support conda-lock's
    ``--filename-template``.

    """
    dir = Path() / "requirements" / "nox.lock"
    name_template = "{py_string}-{platform}.lock"
    if platform_placeholder:
        platform = "{platform}"
    else:
        platform = LOCKFILE_PLATFORM
    lockfile_name = name_template.format(py_string=py_string, platform=platform)
    return dir / lockfile_name


def _session_lockfile(session: nox.sessions.Session) -> Path:
    """Return the path of the session lockfile."""
    return _lockfile_path(py_string=f"py{session.python.replace('.', '')}")


def _file_content(file_path: Path) -> str:
    with file_path.open("r") as file:
        return file.read()


def _session_cachefile(session: nox.sessions.Session) -> Path:
    """Return the path of the session lockfile cache."""
    tmp_dir = Path(session.create_tmp())
    cache = tmp_dir / _session_lockfile(session).name
    return cache


def _venv_populated(session: nox.sessions.Session) -> bool:
    """Return True if the Conda venv has been created and the list of packages in the lockfile installed."""
    return _session_cachefile(session).is_file()


def _venv_changed(session: nox.sessions.Session) -> bool:
    """Return True if the installed session is different to that specified in the lockfile."""
    result = False
    if _venv_populated(session):
        expected = _file_content(_session_lockfile(session))
        actual = _file_content(_session_cachefile(session))
        result = actual != expected
    return result


def _install_and_cache_venv(session: nox.sessions.Session) -> None:
    """
    Cache the nox session environment.

    This consists of saving a hexdigest (sha256) of the associated
    Conda lock file.

    Parameters
    ----------
    session: object
        A `nox.sessions.Session` object.

    """
    lockfile = _session_lockfile(session)
    session.conda_install(f"--file={lockfile}")
    with _session_cachefile(session).open("w") as cachefile:
        cachefile.write(_file_content(lockfile))


def _get_iris_github_artifact(session: nox.sessions.Session) -> str:
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
    if not result:
        # .cirrus.yml sets IRIS_SOURCE. Need to fetch the value (if any) when
        # called outside Cirrus (e.g. user, ASV).
        # .cirrus.yml = single-source-of-truth.
        with Path(".cirrus.yml").open("r") as file:
            cirrus_config = yaml.load(file, Loader=yaml.FullLoader)
        result = cirrus_config["env"].get("IRIS_SOURCE", None)

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


def _prepare_env(session: nox.sessions.Session) -> None:
    venv_dir = session.virtualenv.location_name

    if not _venv_populated(session):
        # Environment has been created but packages not yet installed.
        # Populate the environment from the lockfile.
        logger.debug(f"Populating conda env: {venv_dir}")
        _install_and_cache_venv(session)

    elif _venv_changed(session):
        # Destroy the environment and rebuild it.
        logger.debug(f"Lockfile changed. Recreating conda env: {venv_dir}")
        _reuse_original = session.virtualenv.reuse_existing
        session.virtualenv.reuse_existing = False
        session.virtualenv.create()
        _install_and_cache_venv(session)
        session.virtualenv.reuse_existing = _reuse_original

    logger.debug(f"Environment up to date: {venv_dir}")

    iris_artifact = _get_iris_github_artifact(session)
    if iris_artifact:
        # Install the iris source in develop mode.
        tmp_dir = Path(session.create_tmp())
        iris_dir = tmp_dir / "iris"
        cwd = Path.cwd()
        if not iris_dir.is_dir():
            session.run_always(
                "git", "clone", IRIS_GITHUB, str(iris_dir), external=True
            )
        session.cd(str(iris_dir))
        session.run_always("git", "fetch", "origin", external=True)
        session.run_always("git", "checkout", iris_artifact, external=True)
        session.cd(str(cwd))
        session.install("--no-deps", "--editable", str(iris_dir))

    # Determine whether verbose diagnostics have been requested
    # from the command line.
    verbose = "-v" in session.posargs or "--verbose" in session.posargs

    if verbose:
        session.run_always("conda", "info")
        session.run_always("conda", "list", f"--prefix={venv_dir}")
        session.run_always(
            "conda",
            "list",
            f"--prefix={venv_dir}",
            "--explicit",
        )


@nox.session
def update_lockfiles(session: nox.sessions.Session):
    """
    Re-resolve env specs and store as lockfiles (``requirements/nox.lock/``).

    Original Conda environment specifications are at:
    ``requirements/py**.yml``. The output lock files denote the dependencies
    that iris-esmf-regrid is tested against, and therefore officially supports.

    Parameters
    ----------
    session: object
        A `nox.sessions.Session` object.

    """
    session.install("conda-lock")

    for req_file in Path("requirements").glob(r"py[0-9]*.yml"):
        python_string = req_file.stem

        # Generate the appropriate conda-lock template name, keeping the {platform}
        # placeholder to support conda-lock's internals.
        filename_template = _lockfile_path(python_string, platform_placeholder=True)
        lockfile_path = _lockfile_path(python_string, platform_placeholder=False)
        # Create the parent directory if it doesn't already exist.
        try:
            filename_template.parent.mkdir()
        except FileExistsError:
            pass

        # Use a copy of the requirements file in a tmp dir - the file will
        # be modified if installing a custom Iris checkout.
        tmp_dir = Path(session.create_tmp())
        req_file_local = tmp_dir / req_file.name
        shutil.copy(req_file, req_file_local)

        conda_lock_cmd = [
            "conda-lock",
            "lock",
            f"--filename-template={filename_template}",
            f"--file={req_file_local}",
            f"--platform={LOCKFILE_PLATFORM}",
        ]

        # Get the requirements for Iris too, if an Iris checkout is specified.
        iris_artifact = _get_iris_github_artifact(session)
        if iris_artifact:
            # Remove ``iris`` from dependencies, if present.
            with req_file_local.open("r+") as file:
                reqs = yaml.load(file, Loader=yaml.FullLoader)
                reqs["dependencies"] = [
                    spec for spec in reqs["dependencies"] if not spec.startswith("iris")
                ]
                yaml.dump(reqs, file)

            iris_req_name = f"{python_string}.yml"
            iris_req_url = (
                f"https://raw.githubusercontent.com/SciTools/iris/"
                f"{iris_artifact}/requirements/ci/{iris_req_name}"
            )
            iris_req_file = (tmp_dir / iris_req_name).with_stem(f"{python_string}-iris")
            iris_req = urlopen(iris_req_url).read()
            with iris_req_file.open("wb") as file:
                file.write(iris_req)
            # Conda-lock can resolve multiple requirements files together.
            conda_lock_cmd.append(f"--file={iris_req_file}")

        session.run(*conda_lock_cmd, silent=True)
        print(f"Conda lock file created: {lockfile_path}")


@nox.session
def flake8(session: nox.sessions.Session):
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
def black(session: nox.sessions.Session):
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
def tests(session: nox.sessions.Session):
    """
    Perform esmf-regrid integration and unit tests.

    Parameters
    ----------
    session: object
        A `nox.sessions.Session` object.

    """
    _prepare_env(session)
    # Install the esmf-regrid source in develop mode.
    session.install("--no-deps", "--editable", ".")

    if COVERAGE:
        # Execute the tests with code coverage.
        session.conda_install("--channel=conda-forge", *COVERAGE_PACKAGES)
        session.run("pytest", "--cov-report=xml", "--cov")
        session.run("codecov")
    else:
        # Execute the tests.
        session.run("pytest")


@nox.session
@nox.parametrize(
    ["ci_mode", "gh_pages"],
    [(True, False), (False, False), (False, True)],
    ids=["ci compare", "full", "full then publish"],
)
def benchmarks(session: nox.sessions.Session, ci_mode: bool, gh_pages: bool):
    """
    Perform esmf-regrid performance benchmarks (using Airspeed Velocity).

    Parameters
    ----------
    session: object
        A `nox.sessions.Session` object.
    ci_mode: bool
        Run a cut-down selection of benchmarks, comparing the current commit to
        the last commit for performance regressions.
    gh_pages: bool
        Run ``asv gh-pages --rewrite`` once finished.

    Notes
    -----
    ASV is set up to use ``nox --session=tests --install-only`` to prepare
    the benchmarking environment.

    """
    session.install("asv", "nox", "pyyaml")
    session.cd("benchmarks")
    # Skip over setup questions for a new machine.
    session.run("asv", "machine", "--yes")

    def asv_exec(*sub_args: str) -> None:
        run_args = ["asv", *sub_args]
        help_output = session.run(*run_args, "--help", silent=True)
        if "--python" in help_output:
            # Not all asv commands accept the --python kwarg.
            run_args.append(f"--python={PY_VER[-1]}")
        session.run(*run_args)

    if ci_mode:
        # If on a PR: compare to the base (target) branch.
        #  Else: compare to previous commit.
        previous_commit = os.environ.get("CIRRUS_BASE_SHA", "HEAD^1")
        try:
            asv_exec("continuous", previous_commit, "HEAD", "--bench=ci")
        finally:
            asv_exec("compare", previous_commit, "HEAD")
    else:
        # f32f23a5 = first supporting commit for nox_asv_plugin.py .
        asv_exec("run", "HEAD^1..HEAD")

    if gh_pages:
        asv_exec("gh-pages", "--rewrite")
