Developer Documentation
========================

My personal docs for things to do for this project

.. contents::
    :depth: 3

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

.. image:: https://img.shields.io/badge/Developer-TheProjectsGuy-blue
    :target: https://github.com/TheProjectsGuy
