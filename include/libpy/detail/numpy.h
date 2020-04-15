#pragma once

// Numpy expects code is being used in a way that has a 1:1 correspondence from source
// file to Python extension (as a shared object), for example:
//
//    // my_extension.c
//    #include arrayobject.h
//    ...
//    PyMODINIT_FUNC PyInit_myextension() {
//        import_array();
//        // Rest of module setup.
//    }
//
// Normally when writing a C++ library, header files provide declarations for functions
// and types but not the definitions (except for templates and inline functions) and the
// actual definition lives in one or more C++ files that get linked together into a single
// shared object. Code that just wants to consume the library can include the files so
// that the compiler knows the types and signatures provided by the library, but it
// doesn't need to include the definitions in the consumer's compiled output. Instead,
// when the object is loaded, it will also load the shared object and resolve all of the
// symbols to the definitions in the shared object. This allows the implementation to
// changed in backwards compatible ways and ensures that in a single process, all users of
// the library have the same version. One downside is that the linker uses it's own path
// resolution machinery to find the actual shared object file to use. Also, the resolution
// happens when the shared object is loaded, no code can run before the symbols are
// resolved. Numpy doesn't want users to need to deal with C/C++ linker stuff, and just
// wants to be able to use the Python import system to find the numpy shared object(s). To
// do this, numpy uses it's own linking system. The `numpy/arrayobject.h` will put a
// `static void** PyArray_API = nullptr` name into *each* object that includes it. Many if
// not all of the API functions in numpy are actually macros that resolve to something
// like: `((PyArrayAPIObject*) PyArray_API)->function`. The `import_array()` macro will
// import (through Python) the needed numpy extension modules to get the `PyArray_API` out
// of a capsule-like object. `import_array()` is doing something very similar to what a
// linker does, but without special compiler/linker assistance.
//
// This whole system works fine for when a single TU turns into a single object; however,
// the test suite for libpy links all the test files together along with `main.cc` into a
// single program. This has made it very hard to figure out when and how to initialize the
// `PyArray_API` value. Instead, we now set a macro when compiling for the tests
// (`LIBPY_COMPILING_FOR_TESTS`) which will control the `NO_IMPORT_ARRAY` flag. This flag
// tells numpy to declare the `PyArray_API` flag as an `extern "C" void** PyArray_API`,
// meaning we expect to have this symbol defined by another object we are to be linked
// with. In `main.cc` we also set `LIBPY_TEST_MAIN` to disable `NO_IMPORT_ARRAY` which
// causes changes the declaration of `PyArray_API` to change to: `#define PyArray_API
// PY_ARRAY_UNIQUE_SYMBOL` and then `void** PyArray_API`. Importantly, this removes the
// `static` and `extern` causing the symbol to have external linkage. Then, because the
// tests are declaring the same symbol as extern, they will all resolve to the same
// `PyArray_API` instance and we only need to call `import_array` once in `main.cc`.
#if LIBPY_COMPILING_FOR_TESTS
#define PY_ARRAY_UNIQUE_SYMBOL PyArray_API_libpy
#ifndef LIBPY_TEST_MAIN
#define NO_IMPORT_ARRAY
#endif
#endif
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "libpy/detail/python.h"
#include <numpy/arrayobject.h>
