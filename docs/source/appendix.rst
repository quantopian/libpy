========
Appendix
========

C++
===

``<libpy/abi.h>``
-----------------

.. doxygenstruct:: py::abi::abi_version
   :undoc-members:

.. doxygenvariable:: py::abi::libpy_abi_version

.. doxygenfunction:: py::abi::compatible_versions

.. doxygenfunction:: py::abi::ensure_compatible_libpy_abi

``<libpy/any.h>``
-----------------

.. doxygenclass:: py::any_vtable
   :members:
   :undoc-members:

.. doxygenclass:: py::any_ref
   :members:
   :undoc-members:

.. doxygenfunction:: py::make_any_ref

.. doxygenclass:: py::any_cref
   :members:
   :undoc-members:

.. doxygenfunction:: py::make_any_cref

.. doxygenfunction:: py::dtype_to_vtable

``<libpy/any_vector.h>``
------------------------

.. doxygenclass:: py::any_vector
   :members:
   :undoc-members:

``<libpy/autoclass.h>``
-----------------------

.. doxygenstruct:: py::autoclass
   :members:

.. doxygenstruct:: py::autoclass_interface
   :members:

.. doxygenstruct:: py::autoclass_interface_instance
   :members:

``<libpy/autofunction.h>``
--------------------------

.. doxygenfunction:: py::autofunction

.. doxygenfunction:: py::automethod

.. doxygenclass:: py::arg::keyword
   :members:

.. doxygentypedef:: py::arg::kwd

.. doxygenclass:: py::arg::optional
   :members:

.. doxygentypedef:: py::arg::opt

.. doxygenclass:: py::arg::optional< keyword< Name, T >, none_is_missing >
   :members:

.. doxygentypedef:: py::arg::opt_kwd

.. doxygenclass:: py::dispatch::adapt_argument
   :members:


``<libpy/automodule.h>``
------------------------

.. doxygendefine:: LIBPY_AUTOMODULE

``<libpy/borrowed_ref.h>``
--------------------------

 .. doxygenclass:: py::borrowed_ref
   :members:
   :undoc-members:

``<libpy/buffer.h>``
--------------------

.. doxygentypedef:: py::buffer

.. doxygenvariable:: py::buffer_format

.. doxygenfunction:: py::get_buffer

.. doxygenfunction:: py::buffer_type_compatible(buffer_format_code)

.. doxygenfunction:: py::buffer_type_compatible(const py::buffer&)

``<libpy/build_tuple.h>``
-------------------------

.. doxygenfunction:: py::build_tuple

``<libpy/call_function.h>``
---------------------------

.. doxygenfunction:: py::call_function

.. doxygenfunction:: py::call_function_throws

.. doxygenfunction:: py::call_method

.. doxygenfunction:: py::call_method_throws

``<libpy/char_sequence.h>``
---------------------------

.. doxygentypedef:: py::cs::char_sequence

.. doxygenfunction:: py::cs::literals::operator""_cs

.. doxygenfunction:: py::cs::literals::operator""_arr

.. doxygenfunction:: py::cs::cat(Cs)

.. doxygenfunction:: py::cs::cat(Cs, Ds)

.. doxygenfunction:: py::cs::cat(Cs, Ds, Ts...)

.. doxygenfunction:: py::cs::to_array

.. doxygenfunction:: py::cs::intersperse

.. doxygenfunction:: py::cs::join

``<libpy/datetime64.h>``
------------------------

.. doxygenclass:: py::datetime64
   :members:
   :undoc-members:

.. doxygentypedef:: py::chrono::ns
.. doxygentypedef:: py::chrono::us
.. doxygentypedef:: py::chrono::ms
.. doxygentypedef:: py::chrono::s
.. doxygentypedef:: py::chrono::m
.. doxygentypedef:: py::chrono::h
.. doxygentypedef:: py::chrono::D

.. doxygenfunction:: py::to_chars(py::datetime64char*, char*, const datetime64<unit>&, bool)

.. doxygenfunction:: py::chrono::is_leapyear

.. doxygenfunction:: py::chrono::time_since_epoch

``<libpy/demangle.h>``
----------------------

.. doxygenfunction:: py::util::demangle_string(const char*)

.. doxygenfunction:: py::util::demangle_string(const std::string&)

.. doxygenfunction:: py::util::type_name

.. doxygenclass:: py::util::demangle_error

``<libpy/dict_range.h>``
------------------------

.. doxygenclass:: py::dict_range
   :members:
   :undoc-members:

``<libpy/exception.h>``
-----------------------

.. doxygenclass:: py::exception
   :members:

.. doxygenfunction:: py::raise

.. doxygenfunction:: raise_from_cxx_exception

.. doxygenstruct:: py::dispatch::raise_format

``<libpy/from_object.h>``
-------------------------

.. doxygenfunction:: py::from_object

.. doxygenvariable:: py::has_from_object

.. doxygenstruct:: py::dispatch::from_object

``<libpy/getattr.h>``
---------------------

.. doxygenfunction:: py::getattr

.. doxygenfunction:: py::getattr_throws

.. doxygenfunction:: py::nested_getattr(py::borrowed_ref<>, const T&, const Ts&...)

.. doxygenfunction:: py::nested_getattr_throws

``<libpy/gil.h>``
-----------------

.. doxygenstruct:: py::gil
   :members:

``<libpy/hash.h>``
------------------

.. doxygenfunction:: py::hash_combine(T, Ts...)

.. doxygenfunction:: py::hash_many(const Ts&...)

.. doxygenfunction:: py::hash_tuple

.. doxygenfunction:: py::hash_buffer

``<libpy/itertools.h>``
-----------------------

.. doxygenfunction:: py::zip

.. doxygenfunction:: py::enumerate

.. doxygenfunction:: py::imap

``<libpy/meta.h>``
------------------

.. doxygenstruct:: py::meta::print_t

.. doxygenstruct:: py::meta::print_v

.. doxygentypedef:: py::meta::remove_cvref

.. doxygenvariable:: py::meta::element_of

.. doxygenvariable:: py::meta::search_tuple

.. doxygentypedef:: py::meta::type_cat

.. doxygentypedef:: py::meta::set_diff

``op`` operator function objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each of these types implements ``operator()`` to defer to the named operator while attempting to preserve all the observable properties of calling the underlying operator directly.

.. doxygenstruct:: py::meta::op::add
.. doxygenstruct:: py::meta::op::sub
.. doxygenstruct:: py::meta::op::mul
.. doxygenstruct:: py::meta::op::rem
.. doxygenstruct:: py::meta::op::div
.. doxygenstruct:: py::meta::op::lshift
.. doxygenstruct:: py::meta::op::rshift
.. doxygenstruct:: py::meta::op::and_
.. doxygenstruct:: py::meta::op::xor_
.. doxygenstruct:: py::meta::op::or_
.. doxygenstruct:: py::meta::op::gt
.. doxygenstruct:: py::meta::op::ge
.. doxygenstruct:: py::meta::op::eq
.. doxygenstruct:: py::meta::op::le
.. doxygenstruct:: py::meta::op::lt
.. doxygenstruct:: py::meta::op::ne

.. doxygenstruct:: py::meta::op::neg
.. doxygenstruct:: py::meta::op::pos
.. doxygenstruct:: py::meta::op::inv

``<libpy/ndarray_view.h>``
--------------------------

.. doxygenclass:: py::ndarray_view
   :members:
   :undoc-members:

.. doxygenclass:: py::ndarray_view< T, 1, false >
   :members:
   :undoc-members:

.. doxygentypedef:: py::array_view

.. doxygenfunction:: py::for_each_unordered

Type Erased Views
-----------------

These views use a :cpp:class:`py::any_vtable` object to view a type-erased buffer of data.
Operations that would normally return references return :cpp:class:`py::any_ref` or :cpp:class:`py::any_ref` objects.
These partial specializations implement the same protocol as the non type-erased version; however, extra methods are added to interact with the vtable or to cast the data back to some statically known type.

.. doxygenclass:: py::ndarray_view< any_cref, ndim, higher_dimensional >
   :members:
   :undoc-members:

.. doxygenclass:: py::ndarray_view< any_ref, ndim, higher_dimensional >
   :members:
   :undoc-members:

.. doxygenclass:: py::ndarray_view< any_cref, 1, false >
   :members:
   :undoc-members:

.. doxygenclass:: py::ndarray_view< any_ref, 1, false >
   :members:
   :undoc-members:

Python
======

Miscellaneous
-------------
.. autodata:: libpy.version_info

   The ABI version of libpy.

Build
-----

.. autoclass:: libpy.build.LibpyExtension
