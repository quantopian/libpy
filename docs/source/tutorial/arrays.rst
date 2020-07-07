======
Arrays
======

To get the full power out of a C++ extension, you will often need to pass arrays of data between Python and C++.
Libpy has native support for integrating with numpy, the most popular ndarray library for Python.

Libpy supports receiving arrays as views so that no data needs to be copied.
Libpy array views can also be const to guarantee that the underlying data isn't mutated.
Libpy also supports creating Numpy arrays as views over C++ containers without copying the underlying data.

``py::array_view``
==================

Libpy can accept numpy arrays, or generally any buffer-like object, through a :cpp:struct:`py::ndarray_view`.
:cpp:struct:`py::ndarray_view` is a template type which takes as a parameter the C++ type of the elements of the array and the number of dimensions.
For example: ``py::ndarray_view<std::int32_t, 3>`` is a view of a 3d array of signed 32 bit integers.
The type of the elements of a :cpp:struct:`py::ndarray_view` are fixed at compile time, but the shape is determined at runtime.

As a convenience, :cpp:struct:`py::array_view` is an alias of :cpp:struct:`py::ndarray_view` for 1 dimensional arrays.

Shape and Strides
-----------------

Like numpy, an array view is composed of three parts:

- shape :: ``std::array<std::size_t>``
- strides :: ``std::array<std::int64_t>``
- buffer :: ``(const) std::byte*``

The shape array contains the number of elements along each axis.
For example: ``{2, 3}`` would be an array with 2 rows and 3 columns.

The strides array contains the number of bytes needed to move one step along each axis.
For example: given a ``{2, 3}`` shaped array of 4 byte elements, then strides of ``{12, 4}`` would be a C-contiguous array because the rows are contiguous.
Given the same ``{2, 3}`` shaped array of 4 byte elements, then strides of ``{4, 8}`` would be a Fortran-contiguous array because the rows are contiguous.

Non-contiguous views
--------------------

Array views do not need to view contiguous arrays.
For example, given a C-contiguous ``{4, 5}`` array of 2 byte values, we could take a view of first column by producing an array view with strides ``{10}``.

Simple Array Input
==================

Let's write function to sum an array:

.. code-block:: c++

   std::int64_t simple_sum(py::array_view<const std::int64_t> values) {
       std::int64_t out = 0;
       for (auto value : values) {
           out += value;
       }
       return out;
   }

This function has one parameter, ``values`` which is a view over the data being summed.
This parameter should be passed by value because it is only a view, and therefore small, like a :cpp:struct:`std::string_view`.

From C++
--------

:cpp:struct:`py::array_view` has an implicit constructor from any type that exposes both ``data()`` and ``size()`` member functions, like :cpp:struct:`std::vector`.
This means we can call ``simple_sum`` directly from C++, for example:

.. code-block:: c++

   std::vector<std::int64_t> vs(100);
   std::iota(vs.begin(), vs.end(), 0);

   std::int64_t sum = simple_sum(vs);

From Python
-----------

To call ``simple_sum`` from Python, we must first use :cpp:func:`py::automethod` to adapt the function and then attach it to a module.
For example:

.. code-block::

   LIBPY_AUTOMODULE(libpy_tutorial,
                    arrays,
                    ({py::autofunction<simple_sum>("simple_sum")}))
   (py::borrowed_ref<>) {
       return false;
   }

Now, we can import the function and pass it numpy arrays:

.. ipython:: python

   import numpy as np
   from libpy_tutorial.arrays import simple_sum
   arr = np.arange(10); arr
   simple_sum(arr)

``py::array_view`` interface
============================

:cpp:struct:`py::ndarray_view` has the interface of a standard fixed-size C++ container, like :cpp:struct:`std::array`.
:cpp:struct:`py::ndarray_view` does have a few additions to the standard methods:

Constructors
------------

- :cpp:func:`py::ndarray_view::from_buffer_protocol`
- :cpp:func:`py::ndarray_view::virtual_array`

Extra Member Accessors
----------------------

- :cpp:func:`py::ndarray_view::shape`
- :cpp:func:`py::ndarray_view::strides`
- :cpp:func:`py::ndarray_view::buffer`
- :cpp:func:`py::ndarray_view::rank`
- :cpp:func:`py::ndarray_view::ssize`

Contiguity
----------

Methods are helpers for checking if a view is over a contiguous array.

- :cpp:func:`py::ndarray_view::is_c_contig`
- :cpp:func:`py::ndarray_view::is_f_contig`
- :cpp:func:`py::ndarray_view::is_contig`

Derived Views
-------------

- :cpp:func:`py::ndarray_view::freeze`
- :cpp:func:`py::ndarray_view::slice`
