====================
Shock and Awe [#f1]_
====================

A *concise* overview of ``libpy``. For an introduction to extending Python with C or C++ please see `the Python documentation <https://docs.python.org/3/extending/extending.html>`_ or `Joe Jevnik's C Extension Tutorial <https://llllllllll.github.io/c-extension-tutorial/>`_.

Simple Scalar Functions
=======================

We start by building simple scalar functions in C++ which we can call from Python.

.. ipython:: python

  from libpy_tutorial import scalar_functions

A simple scalar function:

.. literalinclude:: tutorial/libpy_tutorial/scalar_functions.cc
   :lines: 11-13

.. ipython:: python

  scalar_functions.bool_scalar(False)

A great way to use ``libpy`` is to write the code that needs to be fast in C++ and expose that code via Python. Let's estimate ``pi`` using a monte carlo simulation:

.. literalinclude:: tutorial/libpy_tutorial/scalar_functions.cc
   :lines: 15-30

.. ipython:: python

  scalar_functions.monte_carlo_pi(10000000)

Of course, we can build C++ functions that support all the features of regular Python functions.

``libpy`` supports optional args:

.. literalinclude:: tutorial/libpy_tutorial/scalar_functions.cc
   :lines: 34-36

.. ipython:: python

  scalar_functions.optional_arg(b"An argument was passed")
  scalar_functions.optional_arg()

and keyword/optional keyword arguments:

.. literalinclude:: tutorial/libpy_tutorial/scalar_functions.cc
   :lines: 38-44

.. ipython:: python

  scalar_functions.keyword_args(kw_arg_kwd=1)
  scalar_functions.keyword_args(kw_arg_kwd=1, opt_kw_arg_kwd=55)


Working With Arrays
===================

In order to write performant code it is often useful to write vectorized functions that act on arrays. Thus, libpy has extenstive support for ``numpy`` arrays.

.. ipython:: python

  from libpy_tutorial import arrays
  import numpy as np

We can take ``numpy`` arrays as input:

.. literalinclude:: tutorial/libpy_tutorial/arrays.cc
   :lines: 11-17

.. ipython:: python

  some_numbers = np.arange(20000)
  arrays.simple_sum(some_numbers)

and return them as output:

.. literalinclude:: tutorial/libpy_tutorial/arrays.cc
   :lines: 23-43

.. ipython:: python

  prime_mask = arrays.is_prime(some_numbers)
  some_numbers[prime_mask][:100]

.. note:: ``numpy`` arrays passed to C++ are `ranges <https://en.cppreference.com/w/cpp/ranges/range>`_.

.. literalinclude:: tutorial/libpy_tutorial/arrays.cc
   :lines: 19-21

.. ipython:: python

  arrays.simple_sum_iterator(some_numbers)

N Dimensional Arrays
====================

We can also work with n-dimensional arrays. As a motivating example, let's sharpen an image. Specifically - we will sharpen:

.. ipython:: python

  from PIL import Image
  import matplotlib.pyplot as plt # to show the image in documenation
  import numpy as np
  import pkg_resources
  img_file = pkg_resources.resource_stream("libpy_tutorial", "data/original.png")
  img = Image.open(img_file)
  @savefig original.png width=200px
  plt.imshow(img)

.. literalinclude:: tutorial/libpy_tutorial/ndarrays.cc
   :lines: 10-55

.. ipython:: python

  pixels = np.array(img)
  kernel = np.array([
      [0, -1, 0],
      [-1, 5, -1],
      [0, -1, 0]
  ]) # already normalized
  from libpy_tutorial import ndarrays
  res = ndarrays.apply_kernel(pixels, kernel)
  @savefig sharpened.png width=200px
  plt.imshow(res)


.. note:: We are able to pass a shaped n-dimensional array as input and return one as output.


Creating Classes
================

``libpy`` also allows you to construct C++ classes and then easily expose them as if they are regular Python classes.

.. ipython:: python

  from libpy_tutorial.classes import Vec3d

C++ classes are able to emulate all the features of Python classes:

.. literalinclude:: tutorial/libpy_tutorial/classes.cc
   :lines: 9-67

.. literalinclude:: tutorial/libpy_tutorial/classes.cc
   :lines: 93-106

.. ipython:: python

  Vec3d.__doc__
  v = Vec3d(1, 2, 3)
  v
  str(v)
  v.x(), v.y(), v.z()
  w = Vec3d(4, 5, 6); w
  v + w
  v * w
  v.magnitude()

Exceptions
==========

Working with exceptions is also important.

.. ipython:: python

  from libpy_tutorial import exceptions

We can throw exceptions in C++ that will then be dealt with in Python. Two patterns:

1. Throw your own exception: ``throw py::exception(type, msg...)``, maybe in response to an exception from a C-API function.
2. Throw a C++ exception directly.

.. literalinclude:: tutorial/libpy_tutorial/exceptions.cc
   :lines: 11-17

::

    In [40]: exceptions.throw_value_error(4)
    ---------------------------------------------------------------------------
    ValueError                                Traceback (most recent call last)
    <ipython-input-40-6cfdcf9a1ea9> in <module>
    ----> 1 exceptions.throw_value_error(4)

    ValueError: You passed 4 and this is the exception

    In [41]: exceptions.raise_from_cxx()
    ---------------------------------------------------------------------------
    RuntimeError                              Traceback (most recent call last)
    <ipython-input-41-ee2345413222> in <module>
    ----> 1 exceptions.raise_from_cxx()

    RuntimeError: a C++ exception was raised: Supposedly a bad argument was used

.. rubric:: Footnotes

.. [#f1] With naming credit to the intorduction of `Q for Mortals <https://code.kx.com/q4m3/1_Q_Shock_and_Awe/>`_.

=================
Python Extensions
=================

In order to create and use a Python Extension we must do four basic things:

First, we use :cpp:func:`py::autofunction` to create an array of `PyMethoddef <https://docs.python.org/3/c-api/structures.html#c.PyMethodDef>`_.

.. literalinclude:: tutorial/libpy_tutorial/scalar_functions.cc
   :lines: 47-53

Second, we create a `PyModuleDef <https://docs.python.org/3/c-api/module.html#c.PyModuleDef>`_ module.

.. literalinclude:: tutorial/libpy_tutorial/scalar_functions.cc
   :lines: 55-65

Then we intialize the module (`see also <https://docs.python.org/3/extending/extending.html#the-module-s-method-table-and-initialization-function>`_):

.. literalinclude:: tutorial/libpy_tutorial/scalar_functions.cc
   :lines: 67-74

.. note:: The initialization function must be named ``PyInit_name()``, where name is the name of the module.

Finally, we must tell ``setup.py`` to build our module using the ``LibpyExtension`` helper:

.. literalinclude:: tutorial/setup.py
   :lines: 18-28,37,50-54,71-72
