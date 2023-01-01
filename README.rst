FeatMF - Feature Matching Framework
=====================================

Feature Matching Framework has multiple feature detection, description, and matching pipelines under one hood.

.. contents:: Table of contents
    :depth: 2

Setup
--------

Python Poetry
^^^^^^^^^^^^^^^^

Using python version ``3.8`` for development. Using `pyenv <https://github.com/pyenv/pyenv>`_ to manage multiple environments.
Using `python poetry <https://python-poetry.org/>`_ to manage packages and dependency.

.. code-block:: bash

    # Initialize pyenv
    pyenv-init || eval "$(pyenv init -)"
    # If you haven't installed Python 3.8 through this already
    pyenv install 3.8
    # Create the Python Poetry packages
    poetry install
    poetry update


.. image:: https://img.shields.io/badge/Developer-TheProjectsGuy-blue
    :target: https://github.com/TheProjectsGuy
