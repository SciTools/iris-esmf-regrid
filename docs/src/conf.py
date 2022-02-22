"""
Configuration file for the Sphinx documentation builder.

Created originally using sphinx-quickstart on 2022-02-21.
"""

from datetime import datetime
from pathlib import Path
import sys

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
#  add these directories to sys.path here - **using absolute paths**.

source_code_root = (Path(__file__).parents[2]).absolute()
sys.path.append(str(source_code_root))


# -- Project information -----------------------------------------------------

from esmf_regrid import __version__ as esmf_r_version

copyright_years = f"2020 - {datetime.now().year}"

project = "iris-esmf-regrid"
copyright = f"{copyright_years}, SciTools-incubator"
author = "iris-esmf-regrid Contributors"

# The full version, including alpha/beta/rc tags
release = esmf_r_version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    # "sphinx.ext.autosummary",
    "sphinx_copybutton",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "github_url": "https://github.com/SciTools-incubator/iris-esmf-regrid",
    "show_prev_next": False,
    "icon_links": [
        {
            "name": "Support",
            "url": "https://github.com/SciTools-incubator/iris-esmf-regrid/discussions",
            "icon": "fa fa-comments fa-fw",
        }
    ],
}


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


# -- api generation configuration ---------------------------------------------
autodoc_typehints = "none"
# autosummary_imported_members = True
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
        "https://github.com/SciTools-incubator/iris-esmf-regrid/issues/%s",
        "Issue #",
    ),
    "pull": ("https://github.com/SciTools-incubator/iris-esmf-regrid/pull/%s", "PR #"),
}


# -- intersphinx extension ----------------------------------------------------
# See https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html
intersphinx_mapping = {
    "iris": ("https://scitools-iris.readthedocs.io/en/latest/", None),
    "cartopy": ("https://scitools.org.uk/cartopy/docs/latest/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}


# -- todo_ configuration -----------------------------------------------
# See https://www.sphinx-doc.org/en/master/usage/extensions/todo.html
todo_include_todos = True
