project = 'libpy'
copyright = '2020, Quantopian Inc.'
author = 'Quantopian Inc.'

# The full version, including alpha/beta/rc tags
release = '0.1.0'

extensions = [
    'breathe',
    'IPython.sphinxext.ipython_console_highlighting',
    'IPython.sphinxext.ipython_directive',
]

breathe_projects = {'libpy': '../doxygen-build/xml'}
breathe_default_project = 'libpy'


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

highlight_language = 'cpp'
