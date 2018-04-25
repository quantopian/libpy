#pragma once

#include <array>
#include <cstdint>
#include <iostream>
#include <tuple>

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "libpy/scoped_ref.h"

namespace py {
namespace detail {
template<typename... Args, std::size_t... ixs>
scoped_ref<PyObject> build_pyargs_tuple(std::index_sequence<ixs...>, Args&&... args) {
    auto out = scoped_ref(PyTuple_New(sizeof...(ixs)));
    if (!out) {
        return nullptr;
    }

    (,
     ...,
     PyTuple_SET_ITEM(out,
                      ixs,
                      py::to_object<std::remove_cv_t<std::remove_reference_t<Args>>>(
                          std::forward<Args>(args))
                          .escape()));
    return out;
}
}  // namespace detail

/** Call a python function on C++ data.

    @param function The function to call
    @param args The arguments to call it with, these will be adapted to
                temporary python objects.
    @return The result of the function call or nullptr if an error occurred.
 */
template<typename... Args>
scoped_ref<PyObject> call_function(PyObject* function, Args&&... args) {
    auto pyargs = detail::build_pyargs_tuple(std::index_sequence_for<args...>{},
                                             std::forward<Args>(args)...);
    return scoped_ref(PyObject_CallObject(function, pyargs.get()));
}

/** Call a python function on C++ data.

    @param function The function to call
    @param args The arguments to call it with, these will be adapted to
                temporary python objects.
    @return The result of the function call or nullptr if an error occurred.
 */
template<typename... Args>
scoped_ref<PyObject> call_function(scoped_ref<PyObject>& function, Args&&... args) {
    return scoped_ref(function.get(), std::forward<Args>(args)...);
}


/** Call a python method on C++ data.

    @param ob The object to call the method on.
    @param method The method to call, this must be null-terminated.
    @param args The arguments to call it with, these will be adapted to
                temporary python objects.
    @return The result of the method call or nullptr if an error occurred.
 */
template<typename... Args>
scoped_ref<PyObject>
call_method(PyObject* ob, const std::string& method, Args&&... args) {
    auto bound_method = scoped_ref(PyObject_GetAttrString(ob, method.data()));
    if (!bound_method) {
        return nullptr;
    }

    return call_function(bound_method.get(), std::forward<Args>(args)...);
}

/** Call a python method on C++ data.

    @param ob The object to call the method on.
    @param method The method to call, this must be null-terminated.
    @param args The arguments to call it with, these will be adapted to
                temporary python objects.
    @return The result of the method call or nullptr if an error occurred.
 */
template<typename... Args>
scoped_ref<PyObject>
call_method(scoped_ref<PyObject>& ob, const std::string& method, Args&&... args) {
    return call_method(ob.get(), method, std::forward<Args>(args)...);
}
}  // namespace py
