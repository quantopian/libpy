#pragma once

#include <array>
#include <cstdint>
#include <iostream>
#include <tuple>

#include "libpy/detail/python.h"
#include "libpy/numpy_utils.h"
#include "libpy/scoped_ref.h"
#include "libpy/to_object.h"

namespace py {
namespace detail {
template<typename... Args, std::size_t... ixs>
scoped_ref<> build_pyargs_tuple(std::index_sequence<ixs...>, Args&&... args) {
    py::scoped_ref out(PyTuple_New(sizeof...(ixs)));
    if (!out) {
        return nullptr;
    }

    (...,
     PyTuple_SET_ITEM(out.get(), ixs, py::to_object(std::forward<Args>(args)).escape()));
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
scoped_ref<> call_function(PyObject* function, Args&&... args) {
    auto pyargs = detail::build_pyargs_tuple(std::index_sequence_for<Args...>{},
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
scoped_ref<PyObject> call_function(const scoped_ref<>& function, Args&&... args) {
    return call_function(function.get(), std::forward<Args>(args)...);
}

/** Call a python method on C++ data.

    @param ob The object to call the method on.
    @param method The method to call, this must be null-terminated.
    @param args The arguments to call it with, these will be adapted to
                temporary python objects.
    @return The result of the method call or nullptr if an error occurred.
 */
template<typename... Args>
scoped_ref<> call_method(PyObject* ob, const std::string& method, Args&&... args) {
    scoped_ref bound_method(PyObject_GetAttrString(ob, method.data()));
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
scoped_ref<>
call_method(const scoped_ref<>& ob, const std::string& method, Args&&... args) {
    return call_method(ob.get(), method, std::forward<Args>(args)...);
}
}  // namespace py
