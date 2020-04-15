#pragma once

#include <array>
#include <cstdint>
#include <iostream>
#include <tuple>

#include "libpy/borrowed_ref.h"
#include "libpy/build_tuple.h"
#include "libpy/detail/python.h"
#include "libpy/exception.h"
#include "libpy/numpy_utils.h"
#include "libpy/owned_ref.h"
#include "libpy/to_object.h"

namespace py {
/** Call a python function on C++ data.

    @param function The function to call
    @param args The arguments to call it with, these will be adapted to
                temporary python objects.
    @return The result of the function call or nullptr if an error occurred.
 */
template<typename... Args>
owned_ref<> call_function(py::borrowed_ref<> function, Args&&... args) {
    auto pyargs = py::build_tuple(std::forward<Args>(args)...);
    return owned_ref(PyObject_CallObject(function.get(), pyargs.get()));
}

/** Call a python function on C++ data.

    @param function The function to call
    @param args The arguments to call it with, these will be adapted to
                temporary python objects.
    @return The result of the function call. If the function throws a Python
            exception, a `py::exception` will be thrown.
 */
template<typename... Args>
owned_ref<> call_function_throws(py::borrowed_ref<> function, Args&&... args) {
    auto pyargs = py::build_tuple(std::forward<Args>(args)...);
    owned_ref res(PyObject_CallObject(function.get(), pyargs.get()));
    if (!res) {
        throw py::exception{};
    }
    return res;
}

/** Call a python method on C++ data.

    @param ob The object to call the method on.
    @param method The method to call, this must be null-terminated.
    @param args The arguments to call it with, these will be adapted to
                temporary python objects.
    @return The result of the method call or nullptr if an error occurred.
 */
template<typename... Args>
owned_ref<>
call_method(py::borrowed_ref<> ob, const std::string& method, Args&&... args) {
    owned_ref bound_method(PyObject_GetAttrString(ob.get(), method.data()));
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
    @return The result of the method call or nullptr. If the method throws a
            Python exception, a `py::exception` will be thrown.
 */
template<typename... Args>
owned_ref<>
call_method_throws(py::borrowed_ref<> ob, const std::string& method, Args&&... args) {
    owned_ref bound_method(PyObject_GetAttrString(ob.get(), method.data()));
    if (!bound_method) {
        throw py::exception{};
    }

    return call_function_throws(bound_method.get(), std::forward<Args>(args)...);
}
}  // namespace py
