#pragma once

#include <sstream>
#include <string>
#include <tuple>

#include "libpy/borrowed_ref.h"
#include "libpy/char_sequence.h"
#include "libpy/detail/api.h"
#include "libpy/detail/python.h"
#include "libpy/meta.h"
#include "libpy/owned_ref.h"
#include "libpy/util.h"

namespace py {
namespace dispatch {
/** Handlers for formatting C++ data types into Python exception messages.

    To add a new formatter, add a new explicit template specialization for the type and
    implement `static std::ostream& f(std::ostream& out, T value)` which
    should append the new data to the output message.
 */
template<typename T>
struct raise_format {
    static std::ostream& f(std::ostream& out, const T& value) {
        return out << value;
    }
};

template<>
struct raise_format<char> {
    static std::ostream& f(std::ostream& out, char value) {
        out.put(value);
        return out;
    }
};

template<>
struct raise_format<bool> {
    static std::ostream& f(std::ostream& out, bool value) {
        if (value) {
            return out << "true";
        }
        return out << "false";
    }
};

template<>
struct raise_format<PyObject*> {
    static std::ostream& f(std::ostream& out, PyObject* value) {
        py::owned_ref as_str(PyObject_Str(value));
        if (!as_str) {
            out << "<error calling str on id=" << static_cast<void*>(value) << '>';
        }
        else {
            out << py::util::pystring_to_cstring(as_str.get());
        }
        return out;
    }
};

template<>
struct raise_format<PyTypeObject*> {
    static std::ostream& f(std::ostream& out, PyTypeObject* value) {
        return raise_format<PyObject*>::f(out, reinterpret_cast<PyObject*>(value));
    }
};

template<typename T>
struct raise_format<py::borrowed_ref<T>> {
    static std::ostream& f(std::ostream& out, py::borrowed_ref<T> value) {
        return raise_format<PyObject*>::f(out, static_cast<PyObject*>(value));
    }
};

template<typename T>
struct raise_format<py::owned_ref<T>> {
    static std::ostream& f(std::ostream& out, const py::owned_ref<T>& value) {
        return raise_format<PyObject*>::f(out, static_cast<PyObject*>(value));
    }
};
}  // namespace dispatch

namespace detail {
template<typename... Ts>
struct raise_buffer;
}  // namespace detail

// forward declare for friend function
detail::raise_buffer<> raise(py::borrowed_ref<> exc);

namespace detail {
/** Compile-time type for holding the exception and types to feed to
    `PyErr_Format`.
 */
template<typename... Ts>
struct raise_buffer {
private:
    py::borrowed_ref<> m_exc;
    std::tuple<Ts...> m_elements;

protected:
    friend raise_buffer<> py::raise(py::borrowed_ref<>);

    /** Construct a `raise_buffer` from an exception type and some elements.

        @param exc The type of the exception to raise
        @param elements The elements to format into the exception message.
     */
    constexpr raise_buffer(py::borrowed_ref<> exc, Ts&&... elements)
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
    constexpr py::borrowed_ref<> exc() {
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
            std::stringstream msg;
            std::apply(
                [&](auto&&... elements) {
                    (dispatch::raise_format<
                         py::meta::remove_cvref<decltype(elements)>>::f(msg, elements),
                     ...);
                },
                std::move(m_elements));
            PyErr_SetString(m_exc.get(), msg.str().c_str());
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
inline detail::raise_buffer<> raise(py::borrowed_ref<> exc) {
    return detail::raise_buffer<>(exc);
}

/** Raise a python exception from a C++ exception. If a python exception is already raised
    then merge their messages.

    @param e The C++ exception the Python exception is being raised from.
    @return Always `nullptr`. This allows users to write:
            `return raise_from_cxx_exception(e)`
 */
LIBPY_EXPORT std::nullptr_t raise_from_cxx_exception(const std::exception& e);

LIBPY_BEGIN_EXPORT
/** A C++ exception that can be used to communicate a Python error.
 */
class exception : public std::exception {
private:
    std::string m_msg;

    static std::string msg_from_current_pyexc();

public:
    /** Default constructor assumes that an exception has already been thrown.
     */
    inline exception() : m_msg(msg_from_current_pyexc()) {}

    inline exception(const std::string& msg) : m_msg(msg) {}

    /** Create a wrapping exception and raise it in Python.

        @param exc The exception type to raise.
        @param args The arguments to forward to `py::raise`.
     */
    template<typename... Args>
    exception(py::borrowed_ref<> exc, Args&&... args) {
        (raise(exc) << ... << std::forward<Args>(args));
        m_msg = msg_from_current_pyexc();
    }

    virtual inline const char* what() const noexcept override {
        return m_msg.data();
    }
};
LIBPY_END_EXPORT
}  // namespace py
