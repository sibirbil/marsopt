import os
import sys
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath("../.."))

from marsopt import __version__

# Project information
project = "marsopt"
copyright = "2025, Samet Çopur"
author = "Samet Çopur"
version = __version__
release = __version__

# Extensions configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_rtd_theme",
    "sphinx.ext.mathjax",
    "myst_parser",
]

# MyST configuration
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
]

# Napoleon settings
napoleon_use_param = True
napoleon_use_ivar = True
napoleon_use_rtype = True
napoleon_preprocess_types = True

autodoc_type_aliases = {
    "NDArray": "numpy.typing.NDArray",
}

# Type hints settings
autodoc_typehints = "description"
autodoc_typehints_format = "short"
always_document_param_types = True
typehints_fully_qualified = False
typehints_document_rtype = True

# Additional settings for numpy docstring format
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_keyword = True

# Sphinx autodoc ayarları
autodoc_member_order = "bysource"


# Theme configuration
templates_path = ["_templates"]
html_theme = "furo"
html_static_path = ["_static"]
