"""Configuration file for the Sphinx documentation builder.

Created originally using sphinx-quickstart on 2022-02-21.
"""

from datetime import datetime
import locale
import os
from pathlib import Path
import re
import sys

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
#  add these directories to sys.path here - **using absolute paths**.

source_code_root = (Path(__file__).parents[2]).absolute()
sys.path.append(str(source_code_root))

on_rtd = os.environ.get("READTHEDOCS", None) == "True"
if on_rtd:
    rtd_version = os.environ.get("READTHEDOCS_VERSION", "latest")
    rtd_project = os.environ.get("READTHEDOCS_PROJECT")
    rtd_conda_prefix = f"/home/docs/checkouts/readthedocs.org/user_builds/{rtd_project}/conda/{rtd_version}"
    os.environ["ESMFMKFILE"] = f"{rtd_conda_prefix}/lib/esmf.mk"
    os.environ["PROJ_DATA"] = f"{rtd_conda_prefix}/share/proj"
    os.environ["PROJ_NETWORK"] = "OFF"
    locale.setlocale(locale.LC_NUMERIC, "C")


# -- Project information -----------------------------------------------------

from esmf_regrid import __version__ as esmf_r_version

copyright_years = f"2020 - {datetime.now().year}"

project = "iris-esmf-regrid"
copyright = f"{copyright_years}, SciTools"
author = "iris-esmf-regrid Contributors"

# The full version, including alpha/beta/rc tags
release = esmf_r_version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx_copybutton",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinxcontrib.apidoc",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


def _dotv(version):
    result = version
    match = re.match(r"^py(\d+)$", version)
    if match:
        digits = match.group(1)
        if len(digits) > 1:
            result = f"{digits[0]}.{digits[1:]}"
    return result


# Automate the discovery of the python versions tested with CI.
python_support_list = sorted(
    [fname.stem for fname in Path().glob("../../requirements/py*.yml")]
)


if not python_support_list:
    python_support = "unknown Python versions"
elif len(python_support_list) == 1:
    python_support = f"Python {_dotv(python_support_list[0])}"
else:
    rest = ", ".join([_dotv(v) for v in python_support_list[:-1]])
    last = _dotv(python_support_list[-1])
    python_support = f"Python {rest} and {last}"

rst_epilog = f"""
.. |python_support| replace:: {python_support}
"""


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "github_url": "https://github.com/SciTools/iris-esmf-regrid",
    "show_prev_next": False,
    "icon_links": [
        {
            "name": "Support",
            "url": "https://github.com/SciTools/iris-esmf-regrid/discussions",
            "icon": "fa fa-comments fa-fw",
        }
    ],
}


# # Add any paths that contain custom static files (such as style sheets) here,
# # relative to this directory. They are copied after the builtin static files,
# # so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static"]


# -- api generation configuration ---------------------------------------------
autoclass_content = "both"
autodoc_typehints = "none"
modindex_common_prefix = ["esmf_regrid"]


# -- copybutton extension -----------------------------------------------------
# See https://sphinx-copybutton.readthedocs.io/en/latest/
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True
copybutton_line_continuation_character = "\\"


# -- extlinks extension -------------------------------------------------------
# See https://www.sphinx-doc.org/en/master/usage/extensions/extlinks.html

extlinks = {
    "issue": (
        "https://github.com/SciTools/iris-esmf-regrid/issues/%s",
        "Issue #",
    ),
    "pull": ("https://github.com/SciTools/iris-esmf-regrid/pull/%s", "PR #"),
}


# -- intersphinx extension ----------------------------------------------------
# See https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html
intersphinx_mapping = {
    "iris": ("https://scitools-iris.readthedocs.io/en/latest/", None),
    "cartopy": ("https://scitools.org.uk/cartopy/docs/latest/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "esmpy": ("https://earthsystemmodeling.org/esmpy_doc/release/latest/html/", None),
}


# -- todo_ extension ----------------------------------------------------------
# See https://www.sphinx-doc.org/en/master/usage/extensions/todo.html
todo_include_todos = True


# -- apidoc extension ---------------------------------------------------------
# See https://github.com/sphinx-contrib/apidoc
module_dir = source_code_root / "esmf_regrid"

apidoc_module_dir = str(module_dir)
apidoc_output_dir = str(Path(__file__).parent / "_api_generated")
apidoc_excluded_paths = [str(module_dir / "tests")]
apidoc_separate_modules = True
apidoc_extra_args = ["-H", "API"]
