"""Perform test automation with nox.

For further details, see https://nox.thea.codes/en/stable/#

"""

import os
from pathlib import Path
import shutil
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import urlopen

import nox
from nox.logger import logger
import yaml

#: Default to reusing any pre-existing nox environments.
nox.options.reuse_existing_virtualenvs = False

#: Name of the package to test.
PACKAGE = "esmf_regrid"

#: GHA-CI environment variable hook.
PY_VER = os.environ.get("PY_VER", ["3.10", "3.11", "3.12"])

#: GHA-CI environment variable hook.
COVERAGE = os.environ.get("COVERAGE", False)

#: GHA-CI environment variable hook.
#: If you change the IRIS_SOURCE here you will also need to change it in
#: the tests, wheel and benchmark workflows.
IRIS_SOURCE = os.environ.get("IRIS_SOURCE", "github:main")

IRIS_GITHUB = "https://github.com/scitools/iris.git"
LOCKFILE_PLATFORM = "linux-64"

ESMFMKFILE = "ESMFMKFILE"


def _lockfile_path(py_string: str, platform_placeholder: bool = False) -> Path:
    """Return a constructed lockfile path for the relevant python string e.g ``py38``.

    Optionally retain the ``{platform}`` placeholder to support conda-lock's
    ``--filename-template``.

    """
    lockfile_dir = Path() / "requirements" / "locks"
    name_template = "{py_string}-{platform}.lock"
    if platform_placeholder:
        platform = "{platform}"
    else:
        platform = LOCKFILE_PLATFORM
    lockfile_name = name_template.format(py_string=py_string, platform=platform)
    return lockfile_dir / lockfile_name


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
    """Cache the nox session environment.

    This consists of saving a hexdigest (sha256) of the associated
    Conda lock file.

    Parameters
    ----------
    session : object
        A `nox.sessions.Session` object.

    """
    lockfile = _session_lockfile(session)
    session.conda_install(f"--file={lockfile}")
    with _session_cachefile(session).open("w") as cachefile:
        cachefile.write(_file_content(lockfile))


def _get_iris_github_artifact(session: nox.sessions.Session) -> str:
    """Determine whether an Iris source artifact from GitHub is required.

    This can be an Iris branch name, commit sha or tag name.

    Parameters
    ----------
    session : object
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
            if repo.startswith("'", '"'):
                repo = repo[1:]
            if repo.lower() == "github":
                result = artifact
                if result.endswith("'", '"'):
                    result = result[:-1]
    return result


def _prepare_env(session: nox.sessions.Session) -> None:
    venv_dir = session.virtualenv.location_name

    esmf_mk_file = Path(venv_dir) / "lib" / "esmf.mk"
    session.env[ESMFMKFILE] = esmf_mk_file.absolute()

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
    """Re-resolve env specs and store as lockfiles (``requirements/locks/``).

    Original Conda environment specifications are at:
    ``requirements/py**.yml``. The output lock files denote the dependencies
    that iris-esmf-regrid is tested against, and therefore officially supports.

    Parameters
    ----------
    session : object
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
            "--kind=explicit",
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

            try:
                # first attempt with legacy requirements structure
                connection = urlopen(iris_req_url)
            except HTTPError as error:
                if error.code == 404:
                    # retry with new requirements structure i.e., no "ci" directory
                    url = urlparse(iris_req_url)
                    parts = url.path.split("/")
                    parts.remove("ci")
                    url = url._replace(path="/".join(parts))
                    connection = urlopen(url.geturl())

            iris_req = connection.read()
            with iris_req_file.open("wb") as fout:
                fout.write(iris_req)
            # Conda-lock can resolve multiple requirements files together.
            conda_lock_cmd.append(f"--file={iris_req_file}")

        session.run(*conda_lock_cmd, silent=True)
        print(f"Conda lock file created: {lockfile_path}")


@nox.session(python=PY_VER, venv_backend="conda")
def tests(session: nox.sessions.Session):
    """Perform esmf-regrid integration and unit tests.

    Parameters
    ----------
    session : object
        A `nox.sessions.Session` object.

    """
    _prepare_env(session)
    # Install the esmf-regrid source in develop mode.
    session.install("--no-deps", "--editable", ".")

    if COVERAGE:
        # Execute the tests with code coverage.
        session.run("pytest", "--cov-report=xml", "--cov")
    else:
        # Execute the tests.
        session.run("pytest")


@nox.session(python=PY_VER, venv_backend="conda")
def wheel(session: nox.sessions.Session):
    """Perform iris-esmf-regrid local wheel install and import test.

    Parameters
    ----------
    session : object
        A `nox.sessions.Session` object.

    """
    _prepare_env(session)
    session.cd("dist")
    fname = list(Path().glob("esmf_regrid-*.whl"))
    if len(fname) == 0:
        e_msg = "Cannot find wheel to install."
        raise ValueError(e_msg)
    if len(fname) > 1:
        emsg = f"Expected to find 1 wheel to install, found {len(fname)} instead."
        raise ValueError(emsg)
    session.install(fname[0].name)
    session.run(
        "python",
        "-c",
        "import esmf_regrid; print(f'{esmf_regrid.__version__=}')",
        external=True,
    )
