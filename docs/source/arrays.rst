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

Libpy can accept numpy arrays, or generally any buffer-like object, through a :cpp:class:`py::ndarray_view`.
:cpp:class:`py::ndarray_view` is a template type which takes as a parameter the C++ type of the elements of the array and the number of dimensions.
For example: ``py::ndarray_view<std::int32_t, 3>`` is a view of a 3d array of signed 32 bit integers.
The type of the elements of a :cpp:class:`py::ndarray_view` are fixed at compile time, but the shape is determined at runtime.

As a convenience, :cpp:type:`py::array_view` is an alias of :cpp:class:`py::ndarray_view` for one dimensional arrays.

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

The buffer must be a ``(const) std::byte*`` and not a ``(const) T*``

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
This parameter should be passed by value because it is only a view, and therefore small, like a :cpp:class:`std::string_view`.

From C++
--------

:cpp:type:`py::array_view` has an implicit constructor from any type that exposes both ``data()`` and ``size()`` member functions, like :cpp:class:`std::vector`.
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

Shallow Constness
=================

:cpp:class:`py::ndarray_view` implements shallow constness.
Shallow constness means that a ``const py::ndarray_view`` allows mutation to the underlying data, but not mutation of what is being pointed to.
Shallow constness means that :cpp:class:`py::ndarray_view` acts like a pointer, not a reference.
One may have a ``const`` pointer to non ``const`` data.

To create an immutable view, the ``const`` must be injected into the viewed type.
Instead of having a ``const`` view of ``int``, have a view of ``const int``.

.. code-block:: c++

   py::ndarray_view<T, n>        // mutable elements
   const py::ndarray_view<T, n>  // mutable elements
   py::ndarray_view<const T, n>  // immutable elements


Freeze
------

Given a mutable view, the :cpp:func:`py::ndarray_view::freeze` member function returns an immutable view over the same data.
This is useful for ensuring that a particular component doesn't mutate a view that is otherwise mutable.
:cpp:func:`py::ndarray_view::freeze` exists for immutable views, but is a nop.


``py::array_view`` extended interface
=====================================

:cpp:class:`py::ndarray_view` has the interface of a standard fixed-size C++ container, like :cpp:class:`std::array`.
:cpp:class:`py::ndarray_view` does have a few additions to the standard member functions:

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

Member functions that are helpers for checking if a view is over a contiguous array.

- :cpp:func:`py::ndarray_view::is_c_contig`
- :cpp:func:`py::ndarray_view::is_f_contig`
- :cpp:func:`py::ndarray_view::is_contig`

Derived Views
-------------

- :cpp:func:`py::ndarray_view::freeze`
- :cpp:func:`py::ndarray_view::slice`

Free Functions
--------------

- :cpp:func:`py::for_each_unordered`

Constructing Array Views
========================

Ndarray views may be constructed from C++ in a few ways.
The easiest way to get an ndarray view is to accept one as a parameter from a function which has been :cpp:func:`py::automethod` converted.
Libpy will take care of type and dimensionality checking and extracting the buffer from the underlying Python object.

From C++
--------

From Contiguous C++ Containers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One dimensional array views, or :cpp:type:`py::array_view`, objects may be constructed from any C++ object that exposes both a ``data()`` and ``size()`` member functions.
``data()`` must return a ``T*`` which points to an array of ``T`` elements of size ``size()``.
Example containers that can be implicitly constructed from are :cpp:class:`std::vector` and :cpp:class:`std::array`.

Example Usage
`````````````

.. code-block:: c++

   void from_vector() {
       std::vector vec = {1, 2, 3};
       py::array_view view(vec);
   }

   void from_array() {
       std::array arr = {1, 2, 3};
       py::array_view view(arr);
   }

Low Level Constructor
~~~~~~~~~~~~~~~~~~~~~

If one wishes to construct a view from C++ directly, the most fundamental constructor takes the buffer as a ``(const) std::byte*``, the shape array, and the strides array.
It is the user's responsibility to ensure that the buffer is compatible with the provided shape and strides, no checking will or can be done.

From Buffer-like Objects
~~~~~~~~~~~~~~~~~~~~~~~~

To construct an array view from a Python object that exports the buffer protocol, like a :class:`memoryview` or numpy array, there is a static member function :cpp:func:`py::ndarray_view::from_buffer_protocol`.
Unlike a normal constructor, :cpp:func:`py::ndarray_view::from_buffer_protocol` returns a tuple of two parts: the array view instance and a :cpp:type:`py::buffer`.
The :cpp:type:`py::buffer` is an RAII object which manages the lifetime of the underlying buffer which the view is over.
The returned view is only valid as long as the paired :cpp:type:`py::buffer` is alive.
Accessing through the view outside the lifetime of the :cpp:type:`py::buffer`c may trigger a use after free and is undefined behavior.

:cpp:func:`py::ndarray_view::from_buffer_protocol` will check that the runtime type of the Python buffer matches the static type of the C++ array view.
:cpp:func:`py::ndarray_view::from_buffer_protocol` will also check that the runtime dimensionality of the Python buffer matches the static dimensionality of the C++ array view.

Virtual Array Views
~~~~~~~~~~~~~~~~~~~

A virtual array view is a scalar which is broadcasted to present as an array view.
Concretely, a virtual array uses the ``buffer`` member to hold a pointer to a single value, and has strides of all zeros.
By setting all of the strides to zero, this means that the single scalar can satisfy any shape.

A virtual array view is useful when one must satisfy and interface that requires an array view but would like to pass a constant value.
A virtual array view is considerably more efficient than allocating an array and filling it with a constant.
No memory must be allocated, and each access will go to the same cache line.

Because all elements of the view share the same underlying memory, mutable virtual arrays can have unexpected results.
If any value in the array view is mutated, all of the elements would change.
This can have unexpected consequences when passing the views to functions that are not prepared for that behavior.
For this reason, it is recommended to only use const virtual array views.

Virtual array views do not copy nor move from the element being viewed.
For that reason, the view must not outlive the element being broadcasted.

Example Usage
`````````````

.. code-block:: c++

   // Library code

   /** A function which adds two array views, storing the result in the first
       array view.
    */
   void add_inplace(py::array_view<int> a, py::array_view<const int> b) {
       std::transform(a.cbegin(), a.cend(), b.cbegin(), a.begin(), std::plus<>{});
   }

   // User code

   /** The user defined function which wants to call `add_inplace` with a
       scalar.
    */
   void f(py::array_view<int> a) {
       int rhs = 5;
       auto rhs_view = py::array_view<const int>::virtual_array(rhs, a.shape());

       // `rhs_view` points to the same data as `rhs`
       assert(rhs_view.buffer() == reinterpret_cast<const std::byte*>(&rhs));

       add_inplace(a, rhs_view);

       // ...
   }

Here, it is critical not to use ``rhs_view`` after ``rhs`` has gone out of scope because the buffer points to the memory owned by ``rhs``.

Type Erased Views
=================

:cpp:class:`py::ndarray_view` normally have a static type for the elements; however, Python users of numpy arrays might not always think of arrays in this way.
Libpy currently only supports exporting a single overload of a function, so some functions which could be written generically need to have a single signature which can accept arrays of any type.
In addition to the restriction of having a single overload exposed, for some functions, adding a lot of template expansions to have static types doesn't meaningfully improve the performance to justify the increased compile times.

To provide static type-erased values, there are types :cpp:class:`py::any_ref` and :cpp:class:`py::any_cref`.
:cpp:class:`py::any_ref` values act like references, and :cpp:class:`py::any_cref` act like ``const`` references.
Unlike a ``void*``, :cpp:class:`py::any_ref` and :cpp:class:`py::any_cref` hold a virtual method table which implements some basic functionality.
The vtable for both type-erased reference types is a :cpp:class:`py::any_vtable`.
:cpp:class:`py::any_vtable` supports constructing new values, copying, moving, checking equality, and getting the numpy dtype for the type.
:cpp:class:`py::any_vtable` can also provide information about the type like the size and alignment.

``py::array_view<py::any_ref>`` and ``py::array_view<py::any_cref>`` have more specific meaning than "view of an array of any ref objects".
Instead, ``py::array_view<py::any_ref>`` and ``py::array_view<py::any_cref>`` are always homogeneous, meaning all of the elements are the same type.
``py::array_view<py::any_ref>`` and ``py::array_view<py::any_cref>`` have the following members:

- shape :: ``std::array<std::size_t>``
- strides :: ``std::array<std::int64_t>``
- buffer :: ``(const) std::byte*``
- vtable :: :cpp:class:`py::any_vtable`

The shape and strides are the same as a normal :cpp:class:`py::ndarray`.
The buffer is now a pointer to an untyped block of data which should be interpreted based on the vtable.
The vtable member encodes the type of the elements in the array and provides access to the operations on the elements.

Type Casting
------------

For performance reasons, it is still useful to convert to a statically typed array view sometimes.
There is a :cpp:func:`py::ndarray_view::cast` template member function which 
