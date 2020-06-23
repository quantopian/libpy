Setup
=====

Requirements
------------

lipy supports:

- macOS/Linux
- Python >=3.5

lipy requires:

- gcc>=8 or clang>=10
- numpy>=1.11.3

libpy also depends on:

- pcre
- google sparsehash

To install these dependencies:

ubuntu
~~~~~~

.. code-block:: bash

    $ sudo apt install libpcre2-dev libsparsehash-dev

macOS
~~~~~

.. code-block:: bash

    $ brew install pcre2 google-sparsehash

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
