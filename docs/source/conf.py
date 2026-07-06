# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pr_PINN'
copyright = '2026, Francesco-Colombo'
author = 'Francesco-Colombo'
release = '0.0.2'

master_doc = 'index'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx_rtd_theme',
    # 'rst2pdf.pdfbuilder',
    'nbsphinx',
    'IPython.sphinxext.ipython_console_highlighting',
]

templates_path = ['_templates']
# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', '**.ipynb_checkpoints']

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "private-members": True,
    "show-inheritance": True,
}

autodoc_typehints = "description"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Options for PDF output --------------------------------------------------

# Grouping the document tree into LaTeX files. List of tuples# (source start file, target name, title, author, documentclass [howto/manual]).
latex_engine = 'xelatex'
latex_documents = [('index', 'pr_PINN.tex',
                    u'pr_PINN - PINN for pattern recognition course', u'Francesco Colombo', 'manual'),]
latex_show_pagerefs = True
latex_domain_indices = False

pdf_documents = [
    ('index', u'pr_PINN', u'PINN for pattern recognition course', u'Francesco Colombo'),]

nbsphinx_input_prompt = 'In [%s]:'
nbsphinx_kernel_name = 'python3'
nbsphinx_output_prompt = 'Out[%s]:'
