# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'iris-esmf-regrid'
copyright = '2021, iris-esmf-regrid Developers'
author = 'iris-esmf-regrid Developers'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.todo",
    "sphinx.ext.duration",
    "sphinx.ext.coverage",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.extlinks",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx.ext.napoleon",
    "sphinx_panels",
    # TODO: Spelling extension disabled until the dependencies can be included
    # "sphinxcontrib.spelling",
    "sphinx_gallery.gen_gallery",
    "matplotlib.sphinxext.mathmpl",
    "matplotlib.sphinxext.plot_directive",
    # better api documentation (custom)
    "custom_class_autodoc",
    "custom_data_autodoc",
    "generate_package_rst",
]
# -- panels extension ---------------------------------------------------------
# See https://sphinx-panels.readthedocs.io/en/latest/

# -- Napoleon extension -------------------------------------------------------
# See https://sphinxcontrib-napoleon.readthedocs.io/en/latest/sphinxcontrib.napoleon.html
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True  # includes dunders in api doc
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_keyword = True
napoleon_custom_sections = None

# -- spellingextension --------------------------------------------------------
# See https://sphinxcontrib-spelling.readthedocs.io/en/latest/customize.html
spelling_lang = "en_GB"
# The lines in this file must only use line feeds (no carriage returns).
spelling_word_list_filename = ["spelling_allow.txt"]
spelling_show_suggestions = False
spelling_show_whole_line = False
spelling_ignore_importable_modules = True
spelling_ignore_python_builtins = True

# -- copybutton extension -----------------------------------------------------
# See https://sphinx-copybutton.readthedocs.io/en/latest/
copybutton_prompt_text = ">>> "

# sphinx.ext.todo configuration -----------------------------------------------
# See https://www.sphinx-doc.org/en/master/usage/extensions/todo.html
todo_include_todos = True

# api generation configuration
autodoc_member_order = "groupwise"
autodoc_default_flags = ["show-inheritance"]
autosummary_generate = True
autosummary_imported_members = True
autopackage_name = ["iris"]
autoclass_content = "init"
modindex_common_prefix = ["iris"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# -- intersphinx extension ----------------------------------------------------
# See https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html
intersphinx_mapping = {
    "cartopy": ("https://scitools.org.uk/cartopy/docs/latest/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "iris": ("https://scitools-iris.readthedocs.io/en/latest/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Extlinks extension -------------------------------------------------------
# See https://www.sphinx-doc.org/en/master/usage/extensions/extlinks.html

extlinks = {
    "issue": ("https://github.com/SciTools-incubator/iris-esmf-regrid/issues/%s", "Issue #"),
    "pull": ("https://github.com/SciTools-incubator/iris-esmf-regrid/pull/%s", "PR #"),
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']