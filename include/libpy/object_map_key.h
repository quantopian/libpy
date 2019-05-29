#pragma once

#include <Python.h>

#include "libpy/exception.h"
#include "libpy/scoped_ref.h"
#include "libpy/from_object.h"
#include "libpy/to_object.h"

namespace py {

/** A wrapper that allows a `py::scoped_ref` to be used as a key in a mapping structure.

    `object_map_key` overloads `operator==` to dispatch to the underlying Python object's
    `__eq__`.
    `object_map_key` is specialized for `std::hash` to dispatch to the underlying Python
    object's `__hash__`.

    If either operation would throw a Python exception, a C++ `py::exception` is raised.
*/
class object_map_key {
private:
    py::scoped_ref<> m_ob;

public:
    inline object_map_key(const py::scoped_ref<>& ob) : m_ob(ob) {}

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

    inline bool operator==(const object_map_key& other) const {
        if (!m_ob) {
            return !static_cast<bool>(other.m_ob);
        }
        if (!other.m_ob) {
            return false;
        }

        int r = PyObject_RichCompareBool(m_ob.get(), other.get(), Py_EQ);
        if (r < 0) {
            throw py::exception{};
        }

        return r;
    }

    inline bool operator!=(const object_map_key& other) const {
        if (!m_ob) {
            return static_cast<bool>(other.m_ob);
        }
        if (!other.m_ob) {
            return true;
        }

        int r = PyObject_RichCompareBool(m_ob.get(), other.get(), Py_NE);
        if (r < 0) {
            throw py::exception{};
        }

        return r;
    }
};

namespace dispatch {
template<>
struct from_object<object_map_key> {
    static object_map_key f(PyObject* ob) {
        Py_INCREF(ob);
        return object_map_key{py::scoped_ref<>{ob}};
    }
};

template<>
struct to_object<object_map_key> {
    static PyObject* f(const object_map_key& ob) {
        PyObject* out = ob.get();
        Py_INCREF(out);
        return out;
    }
};
}  // namespace dispatch
}  // namespace py

namespace std {
template<>
struct hash<py::object_map_key> {
    auto operator()(const py::object_map_key& ob) const {
        // this returns a different type in Python 2 and Python 3
        using out_type = decltype(PyObject_Hash(ob.get()));

        if (!ob.get()) {
            return out_type{0};
        }

        out_type r = PyObject_Hash(ob.get());;
        if (r == -1) {
            throw py::exception{};
        }

        return r;
    }
};
}  // namespace std
