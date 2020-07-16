``libpy``
=========

.. image:: https://github.com/quantopian/libpy/workflows/CI/badge.svg
    :alt: GitHub Actions status
    :target: https://github.com/quantopian/libpy/actions?query=workflow%3ACI+branch%3Amaster

.. image:: https://badge.fury.io/py/libpy.svg
    :target: https://badge.fury.io/py/libpy

``libpy`` is a library to help you write amazing Python extensions in C++.
``libpy`` makes it easy to expose C++ code to Python.
``libpy`` lets you automatically wrap functions and classes.
``libpy`` is designed for high performance and safety: libpy extension modules should be both faster and safer than using the C API directly.

`Full documentation <https://quantopian.github.io/libpy/>`_

Requirements
------------

libpy supports:

- macOS/Linux
- Python >=3.5

libpy requires:

- gcc>=9 or clang>=10
- numpy>=1.11.3

Optional Requirements
---------------------

libpy optionally provides wrappers for the following libraries:

- google sparsehash


Install
-------

To install for development:

.. code-block:: bash

   $ make

Otherwise, ``pip install libpy``, making sure ``CC`` and ``CXX`` environment variables are set to the the right compiler.

**Note**: The installation of ``libpy`` will use the ``python`` executable to
figure out information about your environment. If you are not using a virtual
environment or ``python`` does not point to the Python installation you want
to use (checked with ``which python`` and ``python --version``) you must
point to your Python exacutable using the ``PYTHON`` environment variable,
i.e. ``PYTHON=python3 make`` or ``PYTHON=python3 pip3 install libpy``.

Tests
-----

To run the unit tests, invoke:

.. code-block:: bash

   $ make test
