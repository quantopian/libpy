#pragma once

#include <type_traits>

#include <Python.h>

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
    explicit scoped_ref(T* ref) : m_ref(ref) {}

    scoped_ref(const scoped_ref& cpfrom) : m_ref(cpfrom.m_ref) {
        Py_XINCREF(m_ref);
    }

    scoped_ref(scoped_ref&& mvfrom) noexcept : m_ref(mvfrom.m_ref) {
        mvfrom.m_ref = nullptr;
    }

    scoped_ref& operator=(const scoped_ref& cpfrom) {
        // we need to incref before we decref to support self assignment
        Py_XINCREF(cpfrom.m_ref);
        Py_XDECREF(m_ref);
        m_ref = cpfrom.m_ref;
        return *this;
    }

    scoped_ref& operator=(scoped_ref&& mvfrom) noexcept {
        std::swap(m_ref, mvfrom.m_ref);
        return *this;
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
    T* get() const {
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
}  // namespace py
