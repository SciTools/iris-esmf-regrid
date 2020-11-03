import os

import nox


nox.options.reuse_existing_virtualenvs = True

PY_VER = os.environ.get("PY_VER", "3.8")
PACKAGE = "esmf_regrid"


@nox.session
def lint(session):
    session.install("flake8", "flake8-docstrings", "flake8-import-order")
    session.run("flake8", PACKAGE)


@nox.session
def style(session):
    session.install("black==20.8b1")
    session.run("black", "--check", PACKAGE)


@nox.session(python=[PY_VER], venv_backend="conda")
def tests(session):
    """
    Reference
      - https://github.com/theacodes/nox/issues/346
      - https://github.com/theacodes/nox/issues/260
    """
    fname = f"requirements/py{PY_VER.replace('.', '')}.yml"
    command = (
        "conda",
        "env",
        "update",
        f"--prefix={session.virtualenv.location}",
        f"--file={fname}",
        "--prune",
    )
    session._run(*command, silent=True, external="error")
    session.run("python", "--version")
