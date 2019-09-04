#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <functional>
#include <tuple>
#include <utility>

#include "libpy/char_sequence.h"
#include "libpy/detail/python.h"
#include "libpy/exception.h"
#include "libpy/from_object.h"
#include "libpy/ndarray_view.h"
#include "libpy/to_object.h"

namespace py {
namespace arg {
/** A wrapper for specifying that a type may be passed by keyword in Python.

    @tparam Name A `py::char_sequence` encoding the name.
    @tparam T The name of the argument in Python.
 */
template<typename Name, typename T>
struct keyword {
private:
    T m_value;

public:
    using name = Name;
    using type = T;

    template<typename... Args>
    keyword(Args&&... args) : m_value(std::forward<Args>(args)...) {}
    keyword(const keyword&) = default;
    keyword(keyword&&) = default;
    keyword& operator=(keyword&&) = default;
    keyword(keyword& cpfrom) : m_value(cpfrom.m_value) {}
    keyword& operator=(const keyword&) = default;

    const T& get() const {
        return m_value;
    }

    T& get() {
        return m_value;
    }
};

/** A wrapper for specifying that a function accepts an optional argument
    from Python.
 */
template<typename T>
class optional {
private:
    std::optional<T> m_value;

public:
    using type = T;

    template<typename... Args>
    optional(Args&&... args) : m_value(std::forward<Args>(args)...) {}
    optional(const optional&) = default;
    optional(optional&&) = default;
    optional& operator=(optional&&) = default;
    optional(optional& cpfrom) : m_value(cpfrom.m_value) {}
    optional& operator=(const optional&) = default;

    const std::optional<T>& get() const {
        return m_value;
    }

    std::optional<T>& get() {
        return m_value;
    }
};

// partial specialization to peek through the `keyword` type to avoid
// `.get().get()` calls.
template<typename Name, typename T>
struct optional<keyword<Name, T>> {
private:
    std::optional<T> m_value;

public:
    using type = T;

    template<typename... Args>
    optional(Args&&... args) : m_value(std::forward<Args>(args)...) {}
    optional(const optional&) = default;
    optional(optional&&) = default;
    optional& operator=(optional&&) = default;
    optional(optional& cpfrom) : m_value(cpfrom.m_value) {}
    optional& operator=(const optional&) = default;

    const std::optional<T>& get() const {
        return m_value;
    }

    std::optional<T>& get() {
        return m_value;
    }
};
}  // namespace arg

namespace dispatch {
template<typename Name, typename T>
struct from_object<arg::keyword<Name, T>> {
    static arg::keyword<Name, T> f(PyObject* ob) {
        if (!ob) {
            auto name = py::cs::to_array(Name{});
            throw py::exception(PyExc_TypeError,
                                "functon missing required argument '",
                                std::string_view(name.data(), name.size() - 1),
                                "'");
        }
        return {py::from_object<T>(ob)};
    }
};

template<typename T>
struct from_object<arg::optional<T>> {
    static arg::optional<T> f(PyObject* ob) {
        if (ob) {
            return {py::from_object<T>(ob)};
        }
        else {
            return {};
        }
    }
};

template<typename Name, typename T>
struct from_object<arg::optional<arg::keyword<Name, T>>> {
    static arg::optional<arg::keyword<Name, T>> f(PyObject* ob) {
        if (ob) {
            return {py::from_object<T>(ob)};
        }
        else {
            return {};
        }
    }
};
}  // namespace dispatch

namespace detail {
/** Helper for extracting and validating the optional and keyword parameters
    from a signature.
 */
template<std::size_t ix, typename... Ts>
struct optionals_and_keywords;

template<typename Name, std::size_t ix>
struct keyword {};

template<std::size_t ix, typename T, typename... Ts>
struct optionals_and_keywords<ix, T, Ts...> {
private:
    using R = optionals_and_keywords<ix + 1, Ts...>;

public:
    using optionals = typename R::optionals;
    using keywords = typename R::keywords;
};

template<std::size_t ix, typename T, typename... Ts>
struct optionals_and_keywords<ix, arg::optional<T>, Ts...> {
private:
    using R = optionals_and_keywords<ix + 1, Ts...>;

    static_assert(std::tuple_size_v<typename R::optionals> == sizeof...(Ts),
                  "non-optional follows optional argument");

public:
    using optionals = meta::type_cat<std::tuple<T>, typename R::optionals>;
    using keywords = typename R::keywords;
};

template<std::size_t ix, typename Name, typename T, typename... Ts>
struct optionals_and_keywords<ix, arg::keyword<Name, T>, Ts...> {
private:
    using R = optionals_and_keywords<ix + 1, Ts...>;

    static_assert(std::tuple_size_v<typename R::keywords> == sizeof...(Ts),
                  "non-keyword follows keyword argument");

public:
    using optionals = typename R::optionals;
    using keywords = meta::type_cat<std::tuple<keyword<Name, ix>>, typename R::keywords>;
};

template<std::size_t ix, typename Name, typename T, typename... Ts>
struct optionals_and_keywords<ix, arg::optional<arg::keyword<Name, T>>, Ts...> {
private:
    using R = optionals_and_keywords<ix + 1, Ts...>;

    static_assert(std::tuple_size_v<typename R::optionals> == sizeof...(Ts),
                  "non-optional follows optional argument");
    static_assert(std::tuple_size_v<typename R::keywords> == sizeof...(Ts),
                  "non-keyword follows keyword argument");

public:
    using optionals = meta::type_cat<std::tuple<T>, typename R::optionals>;
    using keywords = meta::type_cat<std::tuple<keyword<Name, ix>>, typename R::keywords>;
};

template<std::size_t ix>
struct optionals_and_keywords<ix> {
    using optionals = std::tuple<>;
    using keywords = std::tuple<>;
};

template<typename... Args>
struct argument_parser {
public:
    using signature =
        std::tuple<decltype(py::dispatch::from_object<Args>::f(nullptr))...>;

private:
    using parsed = optionals_and_keywords<0, Args...>;
    static constexpr std::size_t arity = sizeof...(Args);
    static constexpr std::size_t nkeywords = std::tuple_size_v<typename parsed::keywords>;
    static constexpr std::size_t nposonly = arity - nkeywords;
    static constexpr std::size_t nrequired =
        arity - std::tuple_size_v<typename parsed::optionals>;

    template<typename Arg, std::size_t ix>
    static decltype(py::from_object<Arg>(nullptr)) positional_arg(PyObject* args) {
        if (static_cast<Py_ssize_t>(ix) < PyTuple_GET_SIZE(args)) {
            return py::from_object<Arg>(PyTuple_GET_ITEM(args, ix));
        }
        else if constexpr (ix < nrequired) {
            throw py::exception(PyExc_AssertionError, "cannot be missing a required arg");
        }
        else {
            return Arg{};
        }
    }

    template<std::size_t... ixs>
    static signature positional_args([[maybe_unused]] PyObject* args,
                                     std::index_sequence<ixs...>) {
        return {positional_arg<Args, ixs>(args)...};
    }

    template<typename Name, std::size_t ix, typename... Keywords>
    static Py_ssize_t get_keyword_ix(std::string_view name,
                                     std::tuple<keyword<Name, ix>, Keywords...>) {
        auto buf = py::cs::to_array(Name{});
        if (name == std::string_view(buf.data(), buf.size() - 1)) {
            return ix;
        }
        return get_keyword_ix(name, std::tuple<Keywords...>{});
    }

    static Py_ssize_t get_keyword_ix(std::string_view name, std::tuple<>) {
        throw py::exception(PyExc_TypeError,
                            "'",
                            name,
                            "' is an invalid keyword argument for this function");
    }

public:
    constexpr static int flags = METH_KEYWORDS | METH_VARARGS;

    static signature parse(PyObject* args, PyObject* kwargs) {
        if (!arity && (PyTuple_GET_SIZE(args) || (kwargs && PyDict_Size(kwargs)))) {
            throw py::exception(PyExc_TypeError, "function takes no arguments");
        }
        if (nkeywords == 0) {
            if (kwargs && PyDict_Size(kwargs)) {
                throw py::exception(PyExc_TypeError,
                                    "function takes no keyword arguments");
            }
            if (PyTuple_GET_SIZE(args) < static_cast<Py_ssize_t>(nrequired)) {
                throw py::exception(PyExc_TypeError,
                                    "function expects at least ",
                                    nrequired,
                                    " arguments but received ",
                                    PyTuple_GET_SIZE(args));
            }
            return positional_args(args, std::make_index_sequence<arity>{});
        }
        // flatten the args + kwargs into a single tuple to pass to
        // `positional_args`
        py::scoped_ref<> flat(PyTuple_New(arity));
        if (!flat) {
            throw py::exception{};
        }
        for (Py_ssize_t ix = 0; ix < PyTuple_GET_SIZE(args); ++ix) {
            PyObject* ob = PyTuple_GET_ITEM(args, ix);
            Py_INCREF(ob);                         // add a new ref
            PyTuple_SET_ITEM(flat.get(), ix, ob);  // steals the new ref
        }
        if (kwargs) {
            for (auto [key, value] : py::dict_range(kwargs)) {
                std::string_view key_view = py::util::pystring_to_string_view(key);
                Py_ssize_t ix = get_keyword_ix(key_view, typename parsed::keywords{});
                if (PyTuple_GET_ITEM(flat.get(), ix)) {
                    throw py::exception(PyExc_TypeError,
                                        "function got multiple values for argument '",
                                        key,
                                        "'");
                }
                Py_INCREF(value);
                PyTuple_SET_ITEM(flat.get(), ix, value);
            }
        }

        return positional_args(flat.get(), std::make_index_sequence<arity>{});
    }
};

template<int flags>
using default_self = std::conditional_t<METH_CLASS & flags, PyTypeObject*, PyObject*>;

/** Struct for extracting traits about the function being wrapped.
 */
template<typename F, int base_flags, typename Self>
struct method_traits;

template<typename R, typename... Args, int base_flags>
struct method_traits<R (*)(Args...), base_flags, void> {
private:
    using parser = argument_parser<Args...>;

public:
    using return_type = R;

    static constexpr auto flags = base_flags | parser::flags;

    static auto build_arg_tuple(PyObject*, PyObject* args, PyObject* kwargs) {
        return parser::parse(args, kwargs);
    }
};

template<typename R, typename... Args, int base_flags, typename Self>
struct method_traits<R (*)(Self, Args...), base_flags, Self> {
private:
    using parser = argument_parser<Args...>;

public:
    using return_type = R;

    static constexpr auto flags = base_flags | parser::flags;

    static auto build_arg_tuple(Self self, PyObject* args, PyObject* kwargs) {
        return std::tuple_cat(std::make_tuple(self), parser::parse(args, kwargs));
    }
};

/** Struct which provides a single function `f` which is the actual implementation of
    `_automethod_wrapper` to use. This is implemented as a struct to allow for partial
    template specialization to optimize for the `METH_NOARGS` case.
 */
template<auto impl, int flags, typename Self>
struct automethodwrapper_impl {
    static PyObject*
    f(std::conditional_t<std::is_same_v<Self, void>, PyObject*, Self> self,
      PyObject* args,
      PyObject* kwargs) {
        using f = method_traits<decltype(impl), flags, Self>;

        if constexpr (std::is_same_v<typename f::return_type, void>) {
            std::apply(impl, f::build_arg_tuple(self, args, kwargs));
            // Allow auto method with void return. This will return a new reference of
            // None to the calling Python.
            Py_RETURN_NONE;
        }
        else if constexpr (std::is_same_v<typename f::return_type, PyObject*>) {
            return std::apply(impl, f::build_arg_tuple(self, args, kwargs));
        }
        else {
            return py::to_object(std::apply(impl, f::build_arg_tuple(self, args, kwargs)))
                .escape();
        }
    }
};

/** The actual funtion that will be registered with the automatically created
    PyMethodDef. This has the signature expected for a python function and will handle
    unpacking the arguments.

    @param self The module or instance this is a method of.
    @param args The arguments to the method as a `PyTupleObject*`.
    @return The result of calling our method.
*/
template<auto impl, int flags, typename Self>
PyObject*
automethodwrapper(std::conditional_t<std::is_same_v<Self, void>, PyObject*, Self> self,
                  PyObject* args,
                  PyObject* kwargs) {
    try {
        return automethodwrapper_impl<impl, flags, Self>::f(self, args, kwargs);
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
    return detail::automethodwrapper<impl, 0, PyTypeObject*>(cls, args, kwargs);
}

namespace detail {
template<typename Self>
PyCFunction unsafe_cast_to_pycfunction(PyObject* (*f)(Self, PyObject*, PyObject*) ) {
    return reinterpret_cast<PyCFunction>(reinterpret_cast<void*>(f));
}
}  // namespace detail

/** Wrap a function as a PyMethodDef with automatic argument unpacking and
    validations. This function will not receive Python's `self` argument.

    @tparam impl The function to wrap.
    @param name The name of the method to expose the Python.
    @param doc The docstring to use. If not provided, the Python `__doc__` attribute
               will be None.
    @return A PyMethodDef which defines the wrapped function.
 */
template<auto impl, int flags = 0>
constexpr PyMethodDef autofunction(const char* const name,
                                   const char* const doc = nullptr) {
    PyMethodDef out{};
    out.ml_name = name;
    out.ml_meth = detail::unsafe_cast_to_pycfunction(
        &detail::automethodwrapper<impl, flags, void>);
    out.ml_flags = detail::method_traits<decltype(impl), flags, void>::flags;
    out.ml_doc = doc;
    return out;
}

/** Wrap a function as a PyMethodDef with automatic argument unpacking and
 * validations. This function must take a `PyObject* self` as a first argument.

    @tparam impl The function to wrap.
    @param name The name of the method to expose the Python.
    @param doc The docstring to use. If not provided, the Python `__doc__` attribute
               will be None.
    @return A PyMethodDef which defines the wrapped function.
 */
template<auto impl, int flags = 0>
constexpr PyMethodDef automethod(const char* const name,
                                 const char* const doc = nullptr) {
    using Self = detail::default_self<flags>;
    PyMethodDef out{};
    out.ml_name = name;
    out.ml_meth = detail::unsafe_cast_to_pycfunction(
        &detail::automethodwrapper<impl, flags, Self>);
    out.ml_flags = detail::method_traits<decltype(impl), flags, Self>::flags;
    out.ml_doc = doc;
    return out;
}

/** The marker used to indicate the end of the array of `PyMethodDef`s.
 */
constexpr PyMethodDef end_method_list = {nullptr, nullptr, 0, nullptr};
}  // namespace py
