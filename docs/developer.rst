Developer Documentation
========================

My personal docs for things to do for this project

.. contents::
    :depth: 3

Packaging
-----------

PyPI
^^^^^

Install the required packages for building the wheels.

.. code-block:: bash

    conda install -c conda-forge hatch twine

Ensure that the ``pyproject.toml`` file is up to date. Build and upload the package to PyPI using

.. code-block:: bash

    python -m build
    python -m twine upload --verbose ./dist/*

Sphinx
------

The following commands were used to create the docs

.. code-block:: bash

    conda install -c conda-forge sphinx sphinx-rtd-theme sphinx-copybutton
    pip install sphinx-reload
    sphinx-quickstart docs

The above commands were installed using ``conda``, but the ``requirements.txt`` is populated using ``pip`` like entries in parallel. This is to install only sphinx packages in the build pipeline for the docs.

Build the docs using

.. code-block:: bash

    # Traditional
    cd docs
    make html

.. code-block:: bash

    # Live reload
    sphinx-reload docs

References
----------

- Sphinx
    - `Quickstart <https://www.sphinx-doc.org/en/master/usage/quickstart.html>`_
        - `Primer <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`
    - Code Documentation
        - `Autodoc code Documentation <https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html>`_: Main extension
        - `Domains <https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html>`_ for referencing
- Packaging
    - PyPI
        - `Getting Started <https://packaging.python.org/en/latest/tutorials/packaging-projects/>`_
        - `PyPI Classifiers <https://pypi.org/classifiers/>`_
    - Conda
        - `conda-build <https://docs.conda.io/projects/conda-build/en/latest/index.html>`_: Building packages
        - `Building a package from scratch <https://docs.conda.io/projects/conda-build/en/latest/user-guide/tutorials/build-pkgs.html>`_
- Blog
    - `An idiot's guide to Python documentation with Sphinx and ReadTheDocs <https://samnicholls.net/2016/06/15/how-to-sphinx-readthedocs/>`_

.. image:: https://img.shields.io/badge/Developer-TheProjectsGuy-blue
    :target: https://github.com/TheProjectsGuy
