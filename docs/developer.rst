Developer Documentation
========================

My personal docs for things to do for this project. Currently **not** accepting contributions for this personal project.

.. contents::
    :depth: 3

Packaging
-----------

Run all packaging and development commands from the repository's root folder.

Versioning
^^^^^^^^^^^

This repository follows `Semantic Versioning 2.0.0 <https://semver.org/>`_. When creating new releases, ensure that the version number is updated in the following places

- ``meta.yaml``: For conda builds
- ``src/featmf/__about__.py``: For the python package
- ``docs/conf.py``: For the documentation

The package dependencies have to be updated in the following places

- ``setup_conda.sh``: For developers using conda
- ``meta.yaml``: For conda builds
- ``docs/requirements.txt``: For building the docs (sphinx).

PyPI
^^^^^

Install the required packages for building the wheels. You need to do this only once.

.. code-block:: bash

    conda install -c conda-forge hatch twine

Ensure that the ``pyproject.toml`` file is up to date. Build and install the project using

.. code-block:: bash

    # Manual build
    python -m build
    # Build using hatch
    hatch build
    # Install (and uninstall) the wheel directly
    pip install ./dist/featmf-0.1.0-py3-none-any.whl
    pip uninstall featmf

Upload the package to PyPI using

.. code-block:: bash

    # Upload to PyPI
    python -m twine upload --verbose ./dist/*

Conda
^^^^^^^

Install the required build tools. You need to do this only once.

.. code-block:: bash

    conda install conda-build anaconda-client

When developing, it's better to not build, but just add folder to path. This way, modifications you make are reflected (live) without re-building package.
You can manage the *developer* install (add to path and test the package) using

.. code-block:: bash

    # Install (add the folder to conda.pth)
    conda develop ./src
    # Verify if this worked (path should be present)
    cat $CONDA_PREFIX/lib/python3.9/site-packages/conda.pth
    # Remove this (after testing is over)
    conda develop -u ./src

Build and install the package (locally) using

.. code-block:: bash

    # Set channels (arguments to the build call)
    chlist="-c conda-forge -c pytorch -c nvidia ..."
    # Build (default output in: ~/anaconda3/conda-bld)
    conda build . $chlist
    # See the path (verify the file)
    conda build --output . $chlist
    # Install
    conda install --use-local featmf
    # Clear builds (if you want to rebuild later)
    conda build purge

Once the package is build, upload the package to Anaconda (personal user `avneesh-mishra <https://anaconda.org/avneesh-mishra/repo>`_)

.. code-block:: bash

    # Login (if not done already)
    anaconda login -h
    anaconda login --hostname HOSTNAME --username ANACONDA_USERNAME \
        --password ANACONDA_PASSWORD
    # Upload the tar ball as a package (see --output of build for path)
    anaconda upload $HOME/anaconda3/conda-bld/linux-64/featmf-0.1.0-py39_0.tar.bz2
    # Verify the package by local install
    conda install -c avneesh-mishra featmf


Sphinx
------

The following dependencies were used to create the docs (one time install)

.. code-block:: bash

    conda install -c conda-forge sphinx sphinx-rtd-theme sphinx-copybutton
    pip install sphinx-reload
    sphinx-quickstart docs

The above commands were installed using ``conda``, but the ``requirements.txt`` (in the ``docs`` folder) is populated using ``pip`` like entries in parallel. 
If you get an error saying that a package was not found, populate it with an appropriate entry and try again. This is to install only sphinx packages in the build pipeline for the docs (don't add everything here).

Build the docs using

.. code-block:: bash

    # Traditional
    cd docs
    make html
    # Live reload
    sphinx-reload docs

References
----------

- Sphinx
    - `Quickstart <https://www.sphinx-doc.org/en/master/usage/quickstart.html>`_
        - `Primer <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_
    - `Sphinx Design <https://sphinx-design.readthedocs.io/en/latest/dropdowns.html>`_
    - `Directives <https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html>`_ for markups
    - Code Documentation
        - `Autodoc code Documentation <https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html>`_: Main extension
        - `Domains <https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html>`_ for referencing
            - `Info fields <https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html#info-field-lists>`_ for declaring the type of function parameters and variables
        - `Describing code in Sphinx <https://www.sphinx-doc.org/en/master/tutorial/describing-code.html>`_ tutorial
- Packaging
    - PyPI
        - `Getting Started <https://packaging.python.org/en/latest/tutorials/packaging-projects/>`_
        - `PyPI Classifiers <https://pypi.org/classifiers/>`_
    - Conda
        - `conda-build <https://docs.conda.io/projects/conda-build/en/latest/index.html>`_: Building packages
            - `Defining metadata <https://docs.conda.io/projects/conda-build/en/latest/resources/build-scripts.html>`_
        - `Building a package from scratch <https://docs.conda.io/projects/conda-build/en/latest/user-guide/tutorials/build-pkgs.html>`_
        - `Anaconda.org User Guide <https://docs.anaconda.com/anacondaorg/user-guide/getting-started/>`_: Getting started
            - `Working with packages <https://docs.anaconda.com/anacondaorg/user-guide/tasks/work-with-packages/>`_
- Blog
    - `An idiot's guide to Python documentation with Sphinx and ReadTheDocs <https://samnicholls.net/2016/06/15/how-to-sphinx-readthedocs/>`_

.. image:: https://img.shields.io/badge/Developer-TheProjectsGuy-blue
    :target: https://github.com/TheProjectsGuy
