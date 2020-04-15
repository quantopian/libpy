#pragma once

#include "libpy/borrowed_ref.h"
#include "libpy/detail/python.h"
#include "libpy/exception.h"
#include "libpy/from_object.h"
#include "libpy/owned_ref.h"
#include "libpy/to_object.h"

namespace py {

LIBPY_BEGIN_EXPORT
/** A wrapper that allows a `py::owned_ref` to be used as a key in a mapping structure.

    `object_map_key` overloads `operator==` to dispatch to the underlying Python object's
    `__eq__`.
    `object_map_key` is specialized for `std::hash` to dispatch to the underlying Python
    object's `__hash__`.

    If either operation would throw a Python exception, a C++ `py::exception` is raised.
*/
class object_map_key {
private:
    py::owned_ref<> m_ob;

public:
    inline object_map_key(std::nullptr_t) : m_ob(nullptr) {}
    inline object_map_key(py::borrowed_ref<> ob)
        : m_ob(py::owned_ref<>::xnew_reference(ob)) {}
    inline object_map_key(py::owned_ref<> ob) : m_ob(std::move(ob)) {}

    object_map_key() = default;
    object_map_key(const object_map_key&) = default;
    object_map_key(object_map_key&&) = default;

    object_map_key& operator=(const object_map_key&) = default;
    object_map_key& operator=(object_map_key&&) = default;

    inline PyObject* get() const {
        return m_ob.get();
    }

    inline explicit operator bool() const noexcept {
        return static_cast<bool>(m_ob);
    }

    inline operator const py::owned_ref<>&() const noexcept {
        return m_ob;
    }

    inline operator py::borrowed_ref<>() const {
        return m_ob;
    }

    bool operator==(py::borrowed_ref<> other) const;
    bool operator!=(py::borrowed_ref<> other) const;
    bool operator<(py::borrowed_ref<> other) const;
    bool operator<=(py::borrowed_ref<> other) const;
    bool operator>=(py::borrowed_ref<> other) const;
    bool operator>(py::borrowed_ref<> other) const;
};
LIBPY_END_EXPORT

namespace dispatch {
template<>
struct from_object<object_map_key> {
    static object_map_key f(py::borrowed_ref<> ob) {
        return object_map_key{ob};
    }
};

template<>
struct to_object<object_map_key> {
    static py::owned_ref<> f(const object_map_key& ob) {
        return py::owned_ref<>::new_reference(ob.get());
    }
};
}  // namespace dispatch
}  // namespace py

namespace std {
template<>
struct hash<py::object_map_key> {
    auto operator()(const py::object_map_key& ob) const {
        using out_type = decltype(PyObject_Hash(ob.get()));

        if (!ob.get()) {
            return out_type{0};
        }

        out_type r = PyObject_Hash(ob.get());
        if (r == -1) {
            throw py::exception{};
        }

        return r;
    }
};
}  // namespace std
