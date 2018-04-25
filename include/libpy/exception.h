#pragma once

#include <string>
#include <tuple>

#include <Python.h>

#include "libpy/char_sequence.h"

namespace py {
namespace dispatch {
/** Handlers for formatting C++ data types into Python exception messages.

    To add a new formatter, add a new explicit template specialization for the
    given type with a `static constexpr cs::char_sequence` member called `fmt`
    which is the format string part for this object. The format strings come
    from https://docs.python.org/3.6/c-api/unicode.html#c.PyUnicode_FromFormat.
    Also add a static function called `prepare` which returns the object
    as it should be fed into the `PyErr_Format` call.
 */
template<typename T>
struct raise_format;

template<>
struct raise_format<char> {
    using fmt = cs::char_sequence<'c'>;

    static auto prepare(char c) {
        return c;
    }
};

template<>
struct raise_format<int> {
    using fmt = cs::char_sequence<'d'>;

    static auto prepare(int i) {
        return i;
    }
};

template<>
struct raise_format<unsigned int> {
    using fmt = cs::char_sequence<'u'>;

    static auto prepare(unsigned int u) {
        return u;
    }
};

template<>
struct raise_format<long> {
    using fmt = cs::char_sequence<'u'>;

    static auto prepare(long l) {
        return l;
    }
};

template<>
struct raise_format<unsigned long> {
    using fmt = cs::char_sequence<'l', 'u'>;

    static auto prepare(unsigned long ul) {
        return ul;
    }
};

template<>
struct raise_format<const char*> {
    using fmt = cs::char_sequence<'s'>;

    static auto prepare(const char* cs) {
        return cs;
    }
};

template<std::size_t n>
struct raise_format<const char[n]> {
    using fmt = cs::char_sequence<'s'>;

    static auto prepare(const char* cs) {
        return cs;
    }
};

template<>
struct raise_format<std::string> {
    using fmt = cs::char_sequence<'s'>;

    static auto prepare(const std::string& cs) {
        return cs.c_str();
    }
};

template<>
struct raise_format<void*> {
    using fmt = cs::char_sequence<'p'>;

    static auto prepare(void* p) {
        return p;
    }
};

template<>
struct raise_format<PyObject*> {
    using fmt = cs::char_sequence<'R'>;

    static auto prepare(PyObject* ob) {
        return ob;
    }
};

template<>
struct raise_format<PyTypeObject*> {
    using fmt = cs::char_sequence<'R'>;

    static auto prepare(PyTypeObject* ob) {
        return ob;
    }
};
}  // namespace dispatch

namespace detail {
template<typename... Ts>
struct raise_buffer;
}  // namespace detail

// forward declare for friend function
detail::raise_buffer<> raise(PyObject* exc);

namespace detail {
/** Compile-time type for holding the exception and types to feed to
    `PyErr_Format`.
 */
template<typename... Ts>
struct raise_buffer {
private:
    PyObject* m_exc;
    std::tuple<Ts...> m_elements;

    /** Prepare the elements to be passed as the variadic arguments to
        `PyErr_Format`.

        @param elements The elements to prepare.
     */
    static auto prepare(Ts&&... elements) {
        return std::make_tuple(
            dispatch::raise_format<std::remove_reference_t<Ts>>::prepare(
                std::forward<Ts>(elements))...);
    }

protected:
    friend raise_buffer<> py::raise(PyObject*);

    /** Construct a `raise_buffer` from an exception type and some elements.

        @param exc The type of the exception to raise
        @param elements The elements to format into the exception message.
     */
    constexpr raise_buffer(PyObject* exc, Ts&&... elements)
        : m_exc(exc), m_elements(std::forward<Ts>(elements)...) {}

public:
    /** Move construct a `raise_buffer`, this invalidates the old
        `raise_buffer`.
     */
    constexpr raise_buffer(raise_buffer&& mvfrom) noexcept
        : m_exc(mvfrom.m_exc), m_elements(std::move(mvfrom.m_elements)) {
        std::move(mvfrom).invalidate();
    }

    /** Construct a new `raise_buffer` which appends a new element to the
        elements.

        @param old The old `raise_buffer` to append to. This is invalidated.
        @param new The element to add.
     */
    template<typename... Old, typename New>
    constexpr raise_buffer(raise_buffer<Old...>&& old, New&& new_) noexcept
        : m_exc(old.exc()),
          m_elements(std::tuple_cat(std::move(old.elements()),
                                    std::make_tuple(std::forward<New>(new_)))) {
        std::move(old).invalidate();
    }

    // cannot copy construct a raise buffer
    raise_buffer(const raise_buffer&) = delete;

    // cannot assign a raise buffer
    raise_buffer& operator=(const raise_buffer&) = delete;

    /** Get the type of the exception to be raised. This is `nullptr` when
        invalidated.
     */
    constexpr PyObject* exc() {
        return m_exc;
    }

    /** Get the elements to format into the message.
     */
    constexpr std::tuple<Ts...>& elements() {
        return m_elements;
    }

    /** Invalidates the `raise_buffer`. This causes `PyErr_Format` to **not**
        get called on destruction.
     */
    void invalidate() && {
        m_exc = nullptr;
    }

    /** Raise the exception with the given elements if not invalidated.
     */
    ~raise_buffer() {
        if (m_exc) {
            auto fmt = cs::to_array(cs::join(
                cs::char_sequence<'%'>{},
                // Prepend the sequence with an empty string to get a %
                // before all of the formatters. If `Ts` is empty this
                // will become `join('%', '')` which is still empty.
                cs::char_sequence<>{},
                typename dispatch::raise_format<std::remove_reference_t<Ts>>::fmt{}...));

            std::apply(PyErr_Format,
                       std::tuple_cat(std::make_tuple(m_exc, fmt.data()),
                                      std::apply(prepare, std::move(m_elements))));
        }
    }

    /** Add an element to the message.

        Invalidates `this`.

        @param element The element to add to the message.
        @return The new `raise_buffer` with the element added.
     */
    template<typename T>
    constexpr auto operator<<(T&& element) && {
        return detail::raise_buffer<Ts..., std::decay_t<T>>(std::move(*this),
                                                            std::forward<T>(element));
    }
};
}  // namespace detail

/** Raise an exception with a format string that is type-safe formatted.

    ## Example

    ```
    if (failed) {
        py::raise(PyExc_ValueError)
            << "failed to do something because: "
            << reason;
        return nullptr;
    }
    ```

    @param exc The Python exception type to raise.
    @return A buffer to write the error message to like a `std::ostream`.
    @see py::raise_format
*/
inline detail::raise_buffer<> raise(PyObject* exc) {
    return detail::raise_buffer<>(exc);
}

/** Raise a python exception from a C++ exception. If a python exception is already raised
    then merge their messages.

    @param e The C++ exception the Python exception is being raised from.
    @return Always `nullptr`. This allows users to write:
            `return raise_from_cxx_exception(e)`
 */
inline std::nullptr_t raise_from_cxx_exception(const std::exception& e) {
    if (!PyErr_Occurred()) {
        py::raise(PyExc_AssertionError) << "a C++ exception was raised: " << e.what();
        return nullptr;
    }
    PyObject* type;
    PyObject* value;
    PyObject* tb;
    PyErr_Fetch(&type, &value, &tb);
    Py_XDECREF(tb);
    raise(type) << value << " (raised from C++ exception: " << e.what() << ")";
    Py_DECREF(type);
    Py_DECREF(value);
    return nullptr;
}

/** A C++ exception that can be used to communicate a Python error.
 */
class exception : public std::exception {
private:
    std::string m_msg;

public:
    /** Default constructor assumes that an exception has already been thrown.
     */
    inline exception(const std::string& msg = "a Python exception occurred")
        : m_msg(msg) {}

    /** Create a wrapping exception and raise it in Python.

        @param exc The exception type to raise.
        @param args The arguments to forward to `py::raise`.
     */
    template<typename... Args>
    inline exception(PyObject* exc, Args&&... args) {
        (raise(exc) << ... << args);
    }

    inline const char* what() const noexcept {
        return m_msg.data();
    }
};
}  // namespace py
