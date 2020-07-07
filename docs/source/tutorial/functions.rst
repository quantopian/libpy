=========
Functions
=========

The simplest unit of code that can be exposed to Python from C++ is a function.
``libpy`` supports automatically converting C++ functions into Python functions by adapting the parameter types and return type.

``libpy`` uses :cpp:func:`py::autofunction` to convert a C++ function into a Python function definition [#f1]_.
The result of :cpp:func:`py::autofunction` can be attached to a Python module object and made available to Python.

A Simple C++ Function
=====================

Let's start by writing a simple C++ function to expose to Python:

.. code-block:: c++

   double fma(double a, double b, double c) {
       return a * b + c;
   }


``fma`` is a standard C++ function with no knowledge of Python.


Adapting a Function
===================

To adapt ``fma`` into a Python function, we need to use :cpp:func:`py::autofunction`.

.. code-block:: c++

   PyMethodDef fma_methoddef = py::autofunction<fma>("fma");

:cpp:func:`py::autofunction` is a template function which takes as a template argument the C++ function to adapt.
:cpp:func:`py::autofunction` also takes a string which is the name of the function as it will be exposed to Python.
The Python function name does not need to match the C++ name.
:cpp:func:`py::autofunction` takes an optional second argument: a string to use as the Python docstring.
For example, a docstring could be added to ``fma`` with:

.. code-block:: c++

   PyMethodDef fma_methoddef = py::autofunction<fma>("fma", "Fused Multiply Add");

.. warning::

   Currently the ``name`` and ``doc`` string parameters **must outlive** the resulting :c:struct:`PyMethodDef`.
   In practice, this means it should be a static string, or string literal.

Adding the Function to a Module
===============================

To use an adapted function from Python, it must be attached to a module so that it may be imported by Python code.
To create a Python method, we can use :c:macro:`LIBPY_AUTOMETHOD`.
:c:macro:`LIBPY_AUTOMETHOD` is a macro which takes in the package name, the module name, and the set of functions to add.
Following the call to :c:macro:`LIBPY_AUTOMETHOD`, we must provide a function which is called when the module is first imported.
To just add functions, our body can be a simple ``return false`` to indicate that no errors occurred.

.. code-block:: c++

   LIBPY_AUTOMODULE(libpy_tutorial, function, ({fma_methoddef}))
       (py::borrowed_ref<>) {
       return false;
   }

Building and Importing the Module
=================================

To build a libpy extension, we can use ``setup.py`` and libpy's :class:`~libpy.build.LibpyExtension` class.

In the ``setup.py``\'s ``setup`` call, we can add a list of ``ext_modules`` to be built:

.. code-block:: python

   from libpy.build import LibpyExtension

   setup(
       # ...
       ext_modules=[
           LibpyExtension(
               'libpy_tutorial.function',
               ['libpy_tutorial/function.cc'],
           ),
       ],
       # ...
   )

Now, the extension can be built with:

.. code-block:: bash

   $ python setup.py build_ext --inplace

Finally, the function can be imported and used from python:


.. ipython:: python

   import libpy  # we need to ensure we import libpy before importing our extension
   from libpy_tutorial.function import fma
   fma(2.0, 3.0, 4.0)

.. rubric:: Footnotes

.. [#f1] :cpp:func:`py::autofunction` creates a :c:struct:`PyMethodDef` instance, which is not yet a Python object.
