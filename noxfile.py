import os

import nox


#: default to reusing any pre-existing nox environments.
nox.options.reuse_existing_virtualenvs = True

#: cirrus-ci environment variable hook.
PY_VER = os.environ.get("PY_VER", "3.8")
#: name of the package to test.
PACKAGE = "esmf_regrid"


@nox.session
def lint(session):
    # pip install the session requirements.
    session.install("flake8", "flake8-docstrings", "flake8-import-order")
    # execute the flake8 linter.
    session.run("flake8", PACKAGE)


@nox.session
def style(session):
    # pip install the session requirements.
    session.install("black==20.8b1")
    # execute the black format checker.
    session.run("black", "--check", PACKAGE)


@nox.session(python=[PY_VER], venv_backend="conda")
def tests(session):
    """
    nox conda support is relatively new and maturing.
      - https://github.com/theacodes/nox/issues/346
      - https://github.com/theacodes/nox/issues/260
    """
    # determine the conda requirements yaml file.
    fname = f"requirements/py{PY_VER.replace('.', '')}.yml"
    # back-door approach to force nox to use "conda env update".
    command = (
        "conda",
        "env",
        "update",
        f"--prefix={session.virtualenv.location}",
        f"--file={fname}",
        "--prune",
    )
    session._run(*command, silent=True, external="error")
    # TBD: replace with an actual test.
    session.run("python", "--version")
