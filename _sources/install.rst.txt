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


Tests
-----

To run the unit tests, invoke:

.. code-block:: bash

   $ make test
