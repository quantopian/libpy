====================
Shock and Awe [#f1]_
====================

A *concise* overview of ``libpy``.

Simple Scalar Functions
=======================

We start by building simple scalar functions in C++ which we can call from Python. For an introduction to extending Python with C or C++ please see `the Python documentation <https://docs.python.org/3/extending/extending.html>`_.

**TODO**: some sort of skeleton reference or something. Embed some code here.

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

  scalar_functions.monte_carlo_pi(10_000_000)

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

It is also very useful to be able to work with ``numpy`` arrays.

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

.. note:: ``numpy`` arrays passed to C++ can be operatated on as iterable types

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

  from libpy_tutorial.classes import SampleClass

C++ classes are able to emulate all the features of Python classes:

.. literalinclude:: tutorial/libpy_tutorial/classes.cc
   :lines: 10-50, 76-88

.. ipython:: python

  SampleClass.__doc__
  sample = SampleClass(5, 10)
  sample.b()
  sample.sum()
  sample(100, 1)
  other_sample = SampleClass(500, 10)
  sample + other_sample
  sample > other_sample
  -sample
  int(sample)

Exceptions
==========

Working with exceptions is also important.

.. ipython:: python

  from libpy_tutorial import exceptions

We can throw exceptions in C++ that will then be dealt with in Python. Three patterns:

1. ``py::raise`` a  Python exception and then ``throw`` it with ``py::exception{}``
2. Construct and throw an exception at the same time
3. ``py::raise_from_cxx_exception`` to raise a exception from a C++ exception, and then throw ``py::exception{}``. This will happen automatically if you ``throw`` a C++ exception to Python.

.. literalinclude:: tutorial/libpy_tutorial/exceptions.cc
   :lines: 11-25

::

    In [39]: exceptions.raise_a_value_error()
    ---------------------------------------------------------------------------
    ValueError                                Traceback (most recent call last)
    <ipython-input-39-5010484f9a47> in <module>
    ----> 1 exceptions.raise_a_value_error()

    ValueError: failed to do something because: wargl bargle

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
