Setup
=====

Requirements
------------

lipy supports:

- macOS/Linux
- Python >=3.5

lipy requires:

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

.. note::
   The installation of ``libpy`` will use the ``python`` executable to
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
