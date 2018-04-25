#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <functional>
#include <tuple>
#include <utility>

#include <Python.h>

#include "libpy/array_view.h"
#include "libpy/char_sequence.h"
#include "libpy/exception.h"
#include "libpy/from_object.h"
#include "libpy/to_object.h"

namespace py {
namespace detail {
/** Struct for extracting traits about the function being wrapped.
 */
template<typename F, typename Self = PyObject*>
struct function_traits;

template<typename R, typename Self, typename... Args>
struct remove_cref_traits {
private:
    template<std::size_t... ixs>
    std::tuple<Args...> inner_build_arg_tuple(PyObject* t, std::index_sequence<ixs...>) {
        return {py::from_object<Args>(PyTuple_GET_ITEM(t, ixs))...};
    }

public:
    using return_type = R;

    static constexpr std::size_t arity = sizeof...(Args);
    static constexpr auto flags = arity ? METH_VARARGS : METH_NOARGS;

    std::tuple<Args...> build_arg_tuple(Self self, PyObject* t) {
        return std::tuple_cat(std::make_tuple(self),
                              inner_build_arg_tuple(t,
                                                    std::index_sequence_for<Args...>{}));
    }
};

template<typename R, typename... Args, typename Self>
struct function_traits<R (*)(Self, Args...), Self>
    : public remove_cref_traits<R,
                                Self,
                                std::remove_const_t<std::remove_reference_t<Args>>...> {};

/** Struct which provides a single function `f` which is the actual implementation of
    `_automethod_wrapper` to use. This is implemented as a struct to allow for partial
    template specialization to optimize for the `METH_NOARGS` case.
 */
template<std::size_t arity, auto impl, typename Self = PyObject*>
struct automethodwrapper_impl {
    static PyObject* f(Self self, PyObject* args) {
        using f = function_traits<decltype(impl), Self>;

        if (PyTuple_GET_SIZE(args) != f::arity) {
            py::raise(PyExc_TypeError)
                << "function expects " << f::arity << " arguments but received "
                << PyTuple_GET_SIZE(args);
            return nullptr;
        }

        return py::to_object(std::apply(impl, f::build_arg_tuple(self, args)));
    }
};

/** `METH_NOARGS` handler for `_automethodwrapper_impl`, hit when `arity == 0`.
 */
template<auto impl, typename Self>
struct automethodwrapper_impl<0, impl, Self> {
    static PyObject* f(Self self, PyObject*) {
        return py::to_object(impl(self));
    }
};

/** The actual funtion that will be registered with the automatically created
    PyMethodDef. This has the signature expected for a python function and will handle
    unpacking the arguments.

    @param self The module or instance this is a method of.
    @param args The arguments to the method as a `PyTupleObject*`.
    @return The result of calling our method.
*/
template<auto impl, typename Self = PyObject*>
PyObject* automethodwrapper(Self self, PyObject* args) {
    using traits = function_traits<decltype(impl), Self>;
    try {
        return automethodwrapper_impl<traits::arity, impl, Self>::f(self, args);
    }
    catch (const std::exception& e) {
        return raise_from_cxx_exception(e);
    }
}
}  // namespace detail

/** Wrap a function such that it is suitable for passing to the `tp_new` slot of a type.

    @tparam impl The function to wrap.
 */
template<auto impl>
PyObject* wrap_new(PyTypeObject* cls, PyObject* args, PyObject* kwargs) {
    if (kwargs && PyDict_Size(kwargs)) {
        raise(PyExc_TypeError) << "type does not accept keyword arguments";
        return nullptr;
    }

    using f = detail::function_traits<decltype(impl), PyTypeObject*>;
    if (f::arity == 0 and PyTuple_GET_SIZE(args)) {
        raise(PyExc_TypeError) << "type does not accept any arguments";
        return nullptr;
    }

    try {
        return detail::automethodwrapper<impl, PyTypeObject*>(cls, args);
    }
    catch (const std::exception& e) {
        return raise_from_cxx_exception(e);
    }
}

/** Wrap a function as a PyMethodDef with automatic argument unpacking and validations.

    @tparam impl The function to wrap.
    @param name The name of the method to expose the Python.
    @param doc The docstring to use. If not provided, the Python `__doc__` attribute
               will be None.
    @return A PyMethodDef which defines the wrapped function.
 */
template<auto impl>
constexpr PyMethodDef automethod(const char* const name,
                                 const char* const doc = nullptr) {
    PyMethodDef out{};
    out.ml_name = name;
    out.ml_meth = reinterpret_cast<PyCFunction>(detail::automethodwrapper<impl>);
    out.ml_flags = detail::function_traits<decltype(impl)>::flags;
    out.ml_doc = doc;
    return out;
}

/** The marker used to indicate the end of the array of `PyMethodDef`s.
 */
constexpr PyMethodDef end_method_list = {nullptr, nullptr, 0, nullptr};
}  // namespace py
