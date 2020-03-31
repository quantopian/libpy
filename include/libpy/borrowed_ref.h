#pragma once

#include "libpy/detail/python.h"

namespace py {
template<typename T>
class owned_ref;

/** A type that explicitly indicates that a Python object is a borrowed
    reference. This is implicitly convertible from a regular `PyObject*` or a
    `py::owned_ref`. This type may be used as a Python object parameter like:

    \code
    int f(borrowed_ref a, borrowed_ref b);
    \endcode

    This allows calling this function with either `py::owned_ref` or
    `PyObject*`.

    @note A `borrowed_ref` may still hold a value of `nullptr`.
 */
template<typename T = PyObject>
class borrowed_ref {
private:
    T* m_ref;

public:
    constexpr borrowed_ref() : m_ref(nullptr) {}
    constexpr borrowed_ref(std::nullptr_t) : m_ref(nullptr) {}
    constexpr borrowed_ref(T* ref) : m_ref(ref) {}
    constexpr borrowed_ref(const py::owned_ref<T>& ref) : m_ref(ref.get()) {}

    constexpr borrowed_ref(const borrowed_ref&) = default;
    constexpr borrowed_ref& operator=(const borrowed_ref& ob) = default;

    constexpr T* get() const {
        return m_ref;
    }

    explicit constexpr operator T*() const {
        return m_ref;
    }

    // use an enable_if to resolve the ambiguous dispatch when T is PyObject
    template<typename U = T,
             typename = std::enable_if_t<!std::is_same<U, PyObject>::value>>
    explicit operator PyObject*() const {
        return reinterpret_cast<PyObject*>(m_ref);
    }

    T& operator*() const {
        return *m_ref;
    }

    T* operator->() const {
        return m_ref;
    }

    explicit operator bool() const {
        return m_ref;
    }

    bool operator==(borrowed_ref other) const {
        return m_ref == other.get();
    }

    bool operator!=(borrowed_ref other) const {
        return m_ref != other.get();
    }
};
}  // namespace py
