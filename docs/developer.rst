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
