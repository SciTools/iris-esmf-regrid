"""
Perform test automation with nox.

For further details, see https://nox.thea.codes/en/stable/#

"""

from datetime import datetime
import os
from pathlib import Path
import shutil
from typing import Literal
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import urlopen

import nox
from nox.logger import logger
import yaml


#: Default to reusing any pre-existing nox environments.
nox.options.reuse_existing_virtualenvs = True

#: Name of the package to test.
PACKAGE = "esmf_regrid"

#: Cirrus-CI environment variable hook.
PY_VER = os.environ.get("PY_VER", ["3.8", "3.9", "3.10"])

#: Cirrus-CI environment variable hook.
COVERAGE = os.environ.get("COVERAGE", False)

#: Cirrus-CI environment variable hook.
IRIS_SOURCE = os.environ.get("IRIS_SOURCE", None)

IRIS_GITHUB = "https://github.com/scitools/iris.git"
LOCKFILE_PLATFORM = "linux-64"

ESMFMKFILE = "ESMFMKFILE"


def _lockfile_path(py_string: str, platform_placeholder: bool = False) -> Path:
    """
    Return a constructed lockfile path for the relevant python string e.g ``py38``.

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
    result = True
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
    Re-resolve env specs and store as lockfiles (``requirements/locks/``).

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
    """
    Perform esmf-regrid integration and unit tests.

    Parameters
    ----------
    session: object
        A `nox.sessions.Session` object.

    """
    esmf_mk_file = Path(session.virtualenv.location_name) / "lib" / "esmf.mk"
    session.env[ESMFMKFILE] = esmf_mk_file
    _prepare_env(session)
    # Install the esmf-regrid source in develop mode.
    session.install("--no-deps", "--editable", ".")

    if COVERAGE:
        # Execute the tests with code coverage.
        session.run("pytest", "--cov-report=xml", "--cov")
        session.run("codecov")
    else:
        # Execute the tests.
        session.run("pytest")


@nox.session
@nox.parametrize(
    "run_type",
    ["branch", "sperf", "custom"],
    ids=["branch", "sperf", "custom"],
)
def benchmarks(
    session: nox.sessions.Session,
    run_type: Literal["overnight", "branch", "sperf", "custom"],
):
    """
    Perform iris-esmf-regrid performance benchmarks (using Airspeed Velocity).

    All run types require a single Nox positional argument (e.g.
    ``nox --session="foo" -- my_pos_arg``) - detailed in the parameters
    section - and can optionally accept a series of further arguments that will
    be added to session's ASV command.

    Parameters
    ----------
    session: object
        A `nox.sessions.Session` object.
    run_type: {"branch", "sperf", "custom"}
        * ``branch``: compares ``HEAD`` and ``HEAD``'s merge-base with the
          input **base branch**. Fails if a performance regression is detected.
          This is the session used by IER's CI.
        * ``sperf``: Run the on-demand SPerf suite of benchmarks (part of the
          UK Met Office NG-VAT project) for the ``HEAD`` of ``upstream/main``
          only, and publish the results to the input **publish directory**,
          within a unique subdirectory for this run.
        * ``custom``: run ASV with the input **ASV sub-command**, without any
          preset arguments - must all be supplied by the user. So just like
          running ASV manually, with the convenience of re-using the session's
          scripted setup steps.

    Examples
    --------
    * ``nox --session="benchmarks(branch)" -- upstream/main``
    * ``nox --session="benchmarks(branch)" -- upstream/mesh-data-model``
    * ``nox --session="benchmarks(branch)" -- upstream/main --bench=ci``
    * ``nox --session="benchmarks(sperf)" -- my_publish_dir
    * ``nox --session="benchmarks(custom)" -- continuous a1b23d4 HEAD --quick``

    """
    # Make sure we're not working with a list of Python versions.
    if not isinstance(PY_VER, str):
        message = (
            "benchmarks session requires PY_VER to be a string - representing "
            f"a single Python version - instead got: {type(PY_VER)} ."
        )
        raise ValueError(message)

    # The threshold beyond which shifts are 'notable'. See `asv compare`` docs
    #  for more.
    COMPARE_FACTOR = 2.0

    session.install("asv", "nox", "pyyaml")

    data_gen_var = "DATA_GEN_PYTHON"
    if data_gen_var in os.environ:
        print("Using existing data generation environment.")
        data_gen_python = Path(os.environ[data_gen_var])
    else:
        print("Setting up the data generation environment...")
        # Get Nox to build an environment for the `tests` session, but don't
        #  run the session. Will re-use a cached environment if appropriate.
        session.run_always(
            "nox",
            "--session=tests",
            "--install-only",
            f"--python={PY_VER}",
        )
        # Find the environment built above, set it to be the data generation
        #  environment.
        data_gen_python = next(
            Path(".nox").rglob(f"tests*/bin/python{PY_VER}")
        ).resolve()
        session.env[data_gen_var] = data_gen_python
    esmf_mk_file = data_gen_python.parents[1] / "lib" / "esmf.mk"
    session.env[ESMFMKFILE] = esmf_mk_file

    print("Running ASV...")
    session.cd("benchmarks")
    # Skip over setup questions for a new machine.
    session.run("asv", "machine", "--yes")

    # All run types require one Nox posarg.
    run_type_arg = {
        "branch": "base branch",
        "sperf": "publish directory",
        "custom": "ASV sub-command",
    }
    if run_type not in run_type_arg.keys():
        message = f"Unsupported run-type: {run_type}"
        raise NotImplementedError(message)
    if not session.posargs:
        message = (
            f"Missing mandatory first Nox session posarg: " f"{run_type_arg[run_type]}"
        )
        raise ValueError(message)
    first_arg = session.posargs[0]
    # Optional extra arguments to be passed down to ASV.
    asv_args = session.posargs[1:]

    if run_type == "branch":
        base_branch = first_arg
        git_command = f"git merge-base HEAD {base_branch}"
        merge_base = session.run(*git_command.split(" "), silent=True, external=True)[
            :8
        ]

        try:
            asv_command = [
                "asv",
                "continuous",
                merge_base,
                "HEAD",
                f"--factor={COMPARE_FACTOR}",
                "--strict",
            ]
            session.run(*asv_command, *asv_args)
        finally:
            asv_command = [
                "asv",
                "compare",
                merge_base,
                "HEAD",
                f"--factor={COMPARE_FACTOR}",
                "--split",
            ]
            session.run(*asv_command)

    elif run_type == "sperf":
        publish_dir = Path(first_arg)
        if not publish_dir.is_dir():
            message = f"Input 'publish directory' is not a directory: {publish_dir}"
            raise NotADirectoryError(message)
        publish_subdir = (
            publish_dir / f"{run_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        publish_subdir.mkdir()

        # Activate on demand benchmarks (C/SPerf are deactivated for 'standard' runs).
        session.env["ON_DEMAND_BENCHMARKS"] = "True"
        commit_range = "upstream/main^!"

        asv_command = [
            "asv",
            "run",
            commit_range,
            "--bench=.*Scalability.*",
            "--attribute",
            "rounds=1",
        ]
        session.run(*asv_command, *asv_args)

        asv_command = ["asv", "publish", commit_range, f"--html-dir={publish_subdir}"]
        session.run(*asv_command)

        # Print completion message.
        location = Path().cwd() / ".asv"
        print(
            f'New ASV results for "{run_type}".\n'
            f'See "{publish_subdir}",'
            f'\n  or JSON files under "{location / "results"}".'
        )

    else:
        asv_subcommand = first_arg
        assert run_type == "custom"
        session.run("asv", asv_subcommand, *asv_args)
