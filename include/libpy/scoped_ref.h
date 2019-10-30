#pragma once

#include <type_traits>

#include "libpy/detail/python.h"

namespace py {
/** An RAII wrapper for ensuring an object is cleaned up in a given scope.
 */
template<typename T = PyObject>
class scoped_ref final {
private:
    T* m_ref;

public:
    /** The type of the underlying pointer.
     */
    using element_type = T;

    /** Default construct a scoped ref to a `nullptr`.
     */
    constexpr scoped_ref() : m_ref(nullptr) {}

    constexpr scoped_ref(std::nullptr_t) : m_ref(nullptr) {}

    /** Manage a new reference. `ref` should not be used outside of the
        `scoped_ref`.

        @param ref The reference to manage
     */
    constexpr explicit scoped_ref(T* ref) : m_ref(ref) {}

    constexpr scoped_ref(const scoped_ref& cpfrom) : m_ref(cpfrom.m_ref) {
        Py_XINCREF(m_ref);
    }

    constexpr scoped_ref(scoped_ref&& mvfrom) noexcept : m_ref(mvfrom.m_ref) {
        mvfrom.m_ref = nullptr;
    }

    constexpr scoped_ref& operator=(const scoped_ref& cpfrom) {
        // we need to incref before we decref to support self assignment
        Py_XINCREF(cpfrom.m_ref);
        Py_XDECREF(m_ref);
        m_ref = cpfrom.m_ref;
        return *this;
    }

    constexpr scoped_ref& operator=(scoped_ref&& mvfrom) noexcept {
        std::swap(m_ref, mvfrom.m_ref);
        return *this;
    }

    /** Create a scoped ref that is a new reference to `ref`.

        @param ref The Python object to create a new managed reference to.
     */
    constexpr static scoped_ref new_reference(T* ref) {
        Py_INCREF(ref);
        return scoped_ref{ref};
    }

    /** Create a scoped ref that is a new reference to `ref` if `ref` is non-null.

        @param ref The Python object to create a new managed reference to. If `ref`
               is `nullptr`, then the resulting object just holds `nullptr` also.
     */
    constexpr static scoped_ref xnew_reference(T* ref) {
        Py_XINCREF(ref);
        return scoped_ref{ref};
    }


    /** Decref the managed pointer if it is not `nullptr`.
     */
    ~scoped_ref() {
        Py_XDECREF(m_ref);
    }

    /** Return the underlying pointer and invalidate the `scoped_ref`.

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

        @return The pointer managed by this `scoped_ref`.
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

    bool operator==(T* other) const {
        return m_ref == other;
    }

    bool operator==(const scoped_ref& other) const {
        return m_ref == other.get();
    }

    bool operator!=(T* other) const {
        return m_ref != other;
    }

    bool operator!=(const scoped_ref& other) const {
        return m_ref != other.get();
    }
};
static_assert(std::is_standard_layout<scoped_ref<>>::value,
              "scoped_ref<> should be standard layout");
static_assert(sizeof(scoped_ref<>) == sizeof(PyObject*),
              "alias type should be the same size as aliased type");
}  // namespace py
