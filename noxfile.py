import nox


PACKAGE = "esmf_regrid"


nox.options.reuse_existing_virtualenvs = True


@nox.session
def lint(session):
    session.install("flake8", "flake8-docstrings", "flake8-import-order")
    session.run("flake8", PACKAGE)


@nox.session
def style(session):
    session.install("black==20.8b1")
    session.run("black", "--check", PACKAGE)
