import os
import sys

sys.path.insert(0, os.path.abspath('../..'))  # points to DSEngine/ root

project = 'DSEngine'
author = 'DSEngine Contributors'
release = '1.0.0'

extensions = [
    'sphinx.ext.autodoc',       # Auto-reads Python docstrings
    'sphinx.ext.napoleon',       # Parses Google-style docstrings
    'sphinx.ext.autosummary',    # Summary tables per module
    'sphinx.ext.viewcode',       # 'View Source' links
    'nbsphinx',                  # Renders Jupyter notebooks
]

# Napoleon: use Google style, not NumPy style
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

# Autodoc: show members and inherited members by default
autodoc_default_options = {
    'members': True,
    'undoc-members': False,    # Skip functions with no docstring
    'private-members': False,  # Skip _private functions
    'show-inheritance': True,
}

# Autosummary: auto-generate stub .rst files
autosummary_generate = True

# HTML theme
html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'navigation_depth': 4,
    'titles_only': False,
}

# Exclude private and test files from docs
exclude_patterns = ['_build', '**.ipynb_checkpoints']

# Static files
html_static_path = ['_static']
