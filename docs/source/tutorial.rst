====================
Shock and Awe [#f1]_
====================

A *concise* overview of ``libpy``.

Simple Scalar Functions
=======================

We start by building simple scalar functions in C++ which we can call from Python.

**TODO**: some sort of skeleton reference or something. Embed some code here.

.. ipython:: python

  from libpy_tutorial import scalar_functions

A simple scalar function:

.. ipython:: python

  scalar_functions.bool_scalar(False)

A great way to use ``libpy`` is to write the code that needs to be fast in C++ and expose that code via Python. Let's estimate ``pi`` using a monte carlo simulation:

.. ipython:: python

  scalar_functions.monte_carlo_pi(10_000_000)

Of course, we can build C++ functions that support all the features of regular Python functions.

``libpy`` supports optional args:

.. ipython:: python

  scalar_functions.optional_arg(b"An argument was passed")
  scalar_functions.optional_arg()

and keyword/optional keyword arguments:

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

.. ipython:: python

  some_numbers = np.arange(20000)
  arrays.simple_sum(some_numbers)

and return them as output:

.. ipython:: python

  prime_mask = arrays.is_prime(some_numbers)
  some_numbers[prime_mask][:100]

.. note:: ``numpy`` arrays passed to C++ can be operatated on as iterable types

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

.. error:: I cannot get my exception examples included without segfaulting sphinx. Unlcear why.



.. rubric:: Footnotes

.. [#f1] With naming credit to the intorduction of `Q for Mortals <https://code.kx.com/q4m3/1_Q_Shock_and_Awe/>`_.
