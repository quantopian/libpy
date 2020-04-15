#pragma once

#include <type_traits>

#include "libpy/borrowed_ref.h"
#include "libpy/detail/python.h"

namespace py {
/** An RAII wrapper for ensuring an object is cleaned up in a given scope.
 */
template<typename T = PyObject>
class owned_ref final {
private:
    T* m_ref;

public:
    /** The type of the underlying pointer.
     */
    using element_type = T;

    /** Default construct a scoped ref to a `nullptr`.
     */
    constexpr owned_ref() : m_ref(nullptr) {}

    constexpr owned_ref(std::nullptr_t) : m_ref(nullptr) {}

    /** Manage a new reference. `ref` should not be used outside of the
        `owned_ref`.

        @param ref The reference to manage
     */
    constexpr explicit owned_ref(T* ref) : m_ref(ref) {}

    constexpr owned_ref(const owned_ref& cpfrom) : m_ref(cpfrom.m_ref) {
        Py_XINCREF(m_ref);
    }

    constexpr owned_ref(owned_ref&& mvfrom) noexcept : m_ref(mvfrom.m_ref) {
        mvfrom.m_ref = nullptr;
    }

    constexpr owned_ref& operator=(const owned_ref& cpfrom) {
        // we need to incref before we decref to support self assignment
        Py_XINCREF(cpfrom.m_ref);
        Py_XDECREF(m_ref);
        m_ref = cpfrom.m_ref;
        return *this;
    }

    constexpr owned_ref& operator=(owned_ref&& mvfrom) noexcept {
        std::swap(m_ref, mvfrom.m_ref);
        return *this;
    }

    /** Create a scoped ref that is a new reference to `ref`.

        @param ref The Python object to create a new managed reference to.
     */
    constexpr static owned_ref new_reference(py::borrowed_ref<T> ref) {
        Py_INCREF(ref.get());
        return owned_ref{ref.get()};
    }

    /** Create a scoped ref that is a new reference to `ref` if `ref` is non-null.

        @param ref The Python object to create a new managed reference to. If
       `ref`
               is `nullptr`, then the resulting object just holds `nullptr` also.
     */
    constexpr static owned_ref xnew_reference(py::borrowed_ref<T> ref) {
        Py_XINCREF(ref.get());
        return owned_ref{ref.get()};
    }

    /** Decref the managed pointer if it is not `nullptr`.
     */
    ~owned_ref() {
        Py_XDECREF(m_ref);
    }

    /** Return the underlying pointer and invalidate the `owned_ref`.

        This allows the reference to "escape" the current scope.

        @return The underlying pointer.
        @see get
     */
    T* escape() && {
        T* ret = m_ref;
        m_ref = nullptr;
        return ret;
    }

    /** Get the underlying managed pointer.

        @return The pointer managed by this `owned_ref`.
        @see escape
     */
    constexpr T* get() const {
        return m_ref;
    }

    explicit operator T*() const {
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

    bool operator==(py::borrowed_ref<> other) const {
        return m_ref == other.get();
    }

    bool operator!=(py::borrowed_ref<> other) const {
        return m_ref != other.get();
    }
};
static_assert(std::is_standard_layout<owned_ref<>>::value,
              "owned_ref<> should be standard layout");
static_assert(sizeof(owned_ref<>) == sizeof(PyObject*),
              "alias type should be the same size as aliased type");
}  // namespace py
