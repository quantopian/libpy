#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <functional>
#include <tuple>
#include <utility>

#include "libpy/borrowed_ref.h"
#include "libpy/buffer.h"
#include "libpy/char_sequence.h"
#include "libpy/detail/numpy.h"
#include "libpy/detail/python.h"
#include "libpy/exception.h"
#include "libpy/from_object.h"
#include "libpy/ndarray_view.h"
#include "libpy/to_object.h"

namespace py {
namespace dispatch {
/** Extension point to adapt Python arguments to C++ arguments for
    `autofunction` adapted function.

    By default, all types are adapted with `py::from_object<T>` where T is the
    type of the argument including cv qualifiers and referenceness. This can be
    extended to support adapting arguments that require some extra state in the
    transformation, for example adapting a buffer-like object into a
    `std::string_view`.

    To add a custom adapted argument type, create an explicit specialization of
    `py::dispatch::adapt_argument`. The specialization should support an
    `adapt_argument(py::borrowed_ref<>)` constructor which is passed the Python
    input. The constructor should throw an exception if the Python object cannot
    be adapted to the given C++ type. The specialization should also implement a
    `get()` method which returns the C++ value to pass to the C++ function
    implementation. Instances of `adapt_argument` will outlive the C++
    implementation function call, so they may be used to manage state like a
    `Py_buffer` or a lock.

    The `get` method should either return an rvalue reference to the adapted
    data or move the data out of `*this` so that it can be forwarded to the
    implementation function. If the type is small like a string view, then a
    value can be returned directly. `get()` will only be called exactly once.
 */
template<typename T>
class adapt_argument {
private:
    decltype(py::from_object<T>(nullptr)) m_memb;

public:
    /** Construct an adapter for a given Python object.

        @param ob The object to adapt.
     */
    adapt_argument(py::borrowed_ref<> ob) : m_memb(py::from_object<T>(ob)) {}

    /** Get the C++ value.
     */
    decltype(m_memb) get() {
        return std::forward<decltype(m_memb)>(m_memb);
    }
};

template<>
class adapt_argument<std::string_view> {
private:
    py::buffer m_buf;
    std::string_view m_memb;

public:
    adapt_argument(py::borrowed_ref<> ob) {
        if (PyBytes_Check(ob.get())) {
            Py_ssize_t size = PyBytes_GET_SIZE(ob.get());
            const char* data = PyBytes_AS_STRING(ob.get());

            m_memb = {data, static_cast<std::size_t>(size)};
        }
        else {
            m_buf = py::get_buffer(ob, PyBUF_CONTIG_RO | PyBUF_FORMAT);
            if (!py::buffer_type_compatible<char>(m_buf)) {
                throw py::exception(PyExc_TypeError,
                                    "cannot adapt convert Python object of type ",
                                    Py_TYPE(ob.get()),
                                    " to std::string_view");
            }

            if (m_buf->ndim != 1) {
                throw py::exception(
                    PyExc_TypeError,
                    "cannot adapt multi-dimensional buffer to a std::string_view");
            }

            m_memb = {static_cast<char*>(m_buf->buf),
                      static_cast<std::size_t>(m_buf->shape[0])};
        }
    }

    std::string_view get() const {
        return m_memb;
    }
};

template<>
class adapt_argument<const std::string_view> : public adapt_argument<std::string_view> {
    using adapt_argument<std::string_view>::adapt_argument;
};

template<>
class adapt_argument<std::string_view&> : public adapt_argument<std::string_view> {
    using adapt_argument<std::string_view>::adapt_argument;
};

template<>
class adapt_argument<const std::string_view&> : public adapt_argument<std::string_view> {
    using adapt_argument<std::string_view>::adapt_argument;
};

template<typename T, std::size_t ndim>
class adapt_argument<py::ndarray_view<T, ndim>> {
    py::buffer m_buf;
    py::ndarray_view<T, ndim> m_memb;

public:
    adapt_argument(py::borrowed_ref<> ob) {
        if (PyArray_Check(ob.get())) {
            // Special case for when `ob` is an ndarray. This is not just an
            // optimization, Python's buffer protocol doesn't support datetime64
            // dtypes, so we need to go to the array directly for that.
            m_memb = py::from_object<py::ndarray_view<T, ndim>>(ob);
        }
        else if constexpr (std::is_same_v<T, py::any_ref> ||
                           std::is_same_v<T, py::any_cref> ||
                           py::buffer_format<T> != '\0') {
            // Guard this branch in an if-constexpr because
            // `from_buffer_protocol` only exists for views over types that are
            // supported by the Python buffer protocol.
            auto [memb, buf] = py::ndarray_view<T, ndim>::from_buffer_protocol(ob);
            std::swap(m_buf, buf);
            m_memb = memb;
        }
        else {
            throw py::invalid_conversion::make<py::ndarray_view<T, ndim>>(ob);
        }
    }

    py::ndarray_view<T, ndim> get() {
        return m_memb;
    }
};

template<typename T, std::size_t ndim>
class adapt_argument<const py::ndarray_view<T, ndim>>
    : public adapt_argument<py::ndarray_view<T, ndim>> {
    using adapt_argument<py::ndarray_view<T, ndim>>::adapt_argument;
};

template<typename T, std::size_t ndim>
class adapt_argument<py::ndarray_view<T, ndim>&>
    : public adapt_argument<py::ndarray_view<T, ndim>> {
    using adapt_argument<py::ndarray_view<T, ndim>>::adapt_argument;
};

template<typename T, std::size_t ndim>
class adapt_argument<const py::ndarray_view<T, ndim>&>
    : public adapt_argument<py::ndarray_view<T, ndim>> {
    using adapt_argument<py::ndarray_view<T, ndim>>::adapt_argument;
};
}  // namespace dispatch

namespace arg {
/** A wrapper for specifying that a type may be passed by keyword in Python.

    @tparam Name A `py::cs::char_sequence` encoding the name.
    @tparam T The name of the argument in Python.
 */
template<typename Name, typename T>
class keyword {
private:
    T m_value;

public:
    /// The name of the keyword argument as a `py::cs::char_sequence` type.
    using name = Name;
    /// The type of the argument.
    using type = T;

    template<typename... Args>
    keyword(Args&&... args) : m_value(std::forward<Args>(args)...) {}
    keyword(const keyword&) = default;
    keyword(keyword&&) = default;
    keyword& operator=(keyword&&) = default;
    keyword(keyword& cpfrom) : m_value(cpfrom.m_value) {}
    keyword& operator=(const keyword&) = default;

    /** Get the argument value.
     */
    T& get() {
        return m_value;
    }

    /** Get the argument value.
     */
    const T& get() const {
        return m_value;
    }
};

/** A wrapper for specifying that a function accepts an optional argument
    from Python.
 */
template<typename T, bool none_is_missing = true>
class optional {
private:
    std::optional<T> m_value;

public:
    /// The type of the argument.
    using type = T;

    template<typename... Args>
    optional(Args&&... args) : m_value(std::forward<Args>(args)...) {}
    optional(const optional&) = default;
    optional(optional&&) = default;
    optional& operator=(optional&&) = default;
    optional(optional& cpfrom) : m_value(cpfrom.m_value) {}
    optional& operator=(const optional&) = default;

    /** Get the argument value.
     */
    std::optional<T>& get() {
        return m_value;
    }

    /** Get the argument value.
     */
    const std::optional<T>& get() const {
        return m_value;
    }
};

/* partial specialization to peek through the `keyword` type to avoid
   `.get().get()` calls.
*/
template<typename Name, typename T, bool none_is_missing>
class optional<keyword<Name, T>, none_is_missing> {
private:
    std::optional<T> m_value;

public:
    /// The name of the keyword argument as a `py::cs::char_sequence` type.
    using name = Name;
    /// The type of the argument.
    using type = T;

    template<typename... Args>
    optional(Args&&... args) : m_value(std::forward<Args>(args)...) {}
    optional(const optional&) = default;
    optional(optional&&) = default;
    optional& operator=(optional&&) = default;
    optional(optional& cpfrom) = default;
    optional& operator=(const optional&) = default;

    /** Get the argument value.

        @note This combines both the `py::arg::optional::get` and `py::arg::keyword::get`
              to just give the `std::optional` directly.
     */
    std::optional<T>& get() {
        return m_value;
    }

    /** Get the argument value.

        @note This combines both the `py::arg::optional::get` and `py::arg::keyword::get`
              to just give the `std::optional` directly.
     */
    const std::optional<T>& get() const {
        return m_value;
    }
};

// helper aliases

template<typename Name, typename T>
using kwd = keyword<Name, T>;

template<typename T, bool none_is_missing = true>
using opt = optional<T, none_is_missing>;

template<typename Name, typename T, bool none_is_missing = true>
using opt_kwd = opt<kwd<Name, T>, none_is_missing>;
}  // namespace arg

namespace dispatch {
template<typename Name, typename T>
class adapt_argument<arg::keyword<Name, T>> {
private:
    dispatch::adapt_argument<T> m_adapted;

public:
    adapt_argument(py::borrowed_ref<> ob) : m_adapted(ob) {}

    arg::keyword<Name, T> get() {
        return arg::keyword<Name, T>(m_adapted.get());
    }
};

template<typename T, bool none_is_missing>
class adapt_argument<arg::optional<T, none_is_missing>> {
private:
    std::optional<dispatch::adapt_argument<T>> m_adapted;

public:
    adapt_argument() = default;

    adapt_argument(py::borrowed_ref<> ob) {
        if (ob && !(none_is_missing && ob.get() == Py_None)) {
            m_adapted = adapt_argument<T>{ob};
        }
    }

    arg::optional<T, none_is_missing> get() {
        if (m_adapted) {
            return m_adapted->get();
        }
        return std::nullopt;
    }
};

template<typename Name, typename T, bool none_is_missing>
class adapt_argument<arg::optional<arg::keyword<Name, T>, none_is_missing>> {
private:
    std::optional<dispatch::adapt_argument<T>> m_adapted;

public:
    adapt_argument() = default;

    adapt_argument(py::borrowed_ref<> ob) {
        if (ob && !(none_is_missing && ob.get() == Py_None)) {
            m_adapted = adapt_argument<T>{ob};
        }
    }

    arg::optional<arg::keyword<Name, T>, none_is_missing> get() {
        if (m_adapted) {
            return m_adapted->get();
        }
        return std::nullopt;
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
    constexpr static std::size_t noptionals = R::noptionals;
    using keywords = typename R::keywords;
};

template<std::size_t ix, typename T, typename... Ts, bool none_is_missing>
struct optionals_and_keywords<ix, arg::optional<T, none_is_missing>, Ts...> {
private:
    using R = optionals_and_keywords<ix + 1, Ts...>;

    static_assert(R::noptionals == sizeof...(Ts),
                  "non-optional follows optional argument");

public:
    constexpr static std::size_t noptionals = R::noptionals + 1;
    using keywords = typename R::keywords;
};

template<std::size_t ix, typename Name, typename T, typename... Ts>
struct optionals_and_keywords<ix, arg::keyword<Name, T>, Ts...> {
private:
    using R = optionals_and_keywords<ix + 1, Ts...>;

    static_assert(std::tuple_size_v<typename R::keywords> == sizeof...(Ts),
                  "non-keyword follows keyword argument");

public:
    constexpr static std::size_t noptionals = R::noptionals;
    using keywords = meta::type_cat<std::tuple<keyword<Name, ix>>, typename R::keywords>;
};

template<std::size_t ix, typename Name, typename T, typename... Ts, bool none_is_missing>
struct optionals_and_keywords<ix,
                              arg::optional<arg::keyword<Name, T>, none_is_missing>,
                              Ts...> {
private:
    using R = optionals_and_keywords<ix + 1, Ts...>;

    static_assert(R::noptionals == sizeof...(Ts),
                  "non-optional follows optional argument");
    static_assert(std::tuple_size_v<typename R::keywords> == sizeof...(Ts),
                  "non-keyword follows keyword argument");

public:
    constexpr static std::size_t noptionals = R::noptionals + 1;
    using keywords = meta::type_cat<std::tuple<keyword<Name, ix>>, typename R::keywords>;
};

template<std::size_t ix>
struct optionals_and_keywords<ix> {
    constexpr static std::size_t noptionals = 0;
    using keywords = std::tuple<>;
};

template<typename... Args>
struct argument_parser {
public:
    using signature = std::tuple<dispatch::adapt_argument<Args>...>;

private:
    using parsed = optionals_and_keywords<0, py::meta::remove_cvref<Args>...>;
    static constexpr std::size_t arity = sizeof...(Args);
    static constexpr std::size_t nkeywords = std::tuple_size_v<typename parsed::keywords>;
    static constexpr std::size_t nposonly = arity - nkeywords;
    static constexpr std::size_t nrequired = arity - parsed::noptionals;

    template<typename Arg, std::size_t ix>
    static dispatch::adapt_argument<Arg> positional_arg(PyObject* args) {
        if (static_cast<Py_ssize_t>(ix) < PyTuple_GET_SIZE(args)) {
            return dispatch::adapt_argument<Arg>(PyTuple_GET_ITEM(args, ix));
        }
        else if constexpr (ix < nrequired) {
            throw py::exception(PyExc_AssertionError, "cannot be missing a required arg");
        }
        else {
            return dispatch::adapt_argument<Arg>{};
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

        auto mismatched_args = [](Py_ssize_t nargs) {
            if (nrequired == arity) {
                return py::exception(PyExc_TypeError,
                                     "function takes ",
                                     arity,
                                     " argument",
                                     (arity != 1) ? "s" : "",
                                     " but ",
                                     nargs,
                                     " were given");
            }
            return py::exception(PyExc_TypeError,
                                 "function takes from ",
                                 nrequired,
                                 " to ",
                                 arity,
                                 " arguments but ",
                                 nargs,
                                 " were given");
        };

        if (nkeywords == 0) {
            if (kwargs && PyDict_Size(kwargs)) {
                throw py::exception(PyExc_TypeError,
                                    "function takes no keyword arguments");
            }

            if (PyTuple_GET_SIZE(args) < static_cast<Py_ssize_t>(nrequired) ||
                PyTuple_GET_SIZE(args) > static_cast<Py_ssize_t>(arity)) {
                throw mismatched_args(PyTuple_GET_SIZE(args));
            }
            return positional_args(args, std::make_index_sequence<arity>{});
        }

        // flatten the args + kwargs into a single tuple to pass to
        // `positional_args`
        py::owned_ref<> flat(PyTuple_New(arity));
        if (!flat) {
            throw py::exception{};
        }
        for (Py_ssize_t ix = 0; ix < PyTuple_GET_SIZE(args); ++ix) {
            PyObject* ob = PyTuple_GET_ITEM(args, ix);
            Py_INCREF(ob);                         // add a new ref
            PyTuple_SET_ITEM(flat.get(), ix, ob);  // steals the new ref
        }

        Py_ssize_t total_args = PyTuple_GET_SIZE(args);
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
                Py_INCREF(value.get());
                PyTuple_SET_ITEM(flat.get(), ix, value.get());
            }

            total_args += PyDict_Size(kwargs);
        }
        if (total_args < static_cast<Py_ssize_t>(nrequired)) {
            throw mismatched_args(total_args);
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

    static R apply(R (*f)(Args...), PyObject*, PyObject* args, PyObject* kwargs) {
        auto parsed = parser::parse(args, kwargs);
        return std::apply([&](auto&... args) -> R { return f(args.get()...); }, parsed);
    }
};

template<typename R, typename... Args, int base_flags, typename Self>
struct method_traits<R (*)(Self, Args...), base_flags, Self> {
private:
    using parser = argument_parser<Args...>;

public:
    using return_type = R;

    static constexpr auto flags = base_flags | parser::flags;

    static R apply(R (*f)(Self, Args...), Self self, PyObject* args, PyObject* kwargs) {
        auto parsed = parser::parse(args, kwargs);
        return std::apply([&](auto&... args) -> R { return f(self, args.get()...); },
                          parsed);
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
            f::apply(impl, self, args, kwargs);
            // Allow auto method with void return. This will return a new reference of
            // None to the calling Python.
            Py_RETURN_NONE;
        }
        else if constexpr (std::is_same_v<typename f::return_type, PyObject*>) {
            return f::apply(impl, self, args, kwargs);
        }
        else {
            return py::to_object(f::apply(impl, self, args, kwargs)).escape();
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
