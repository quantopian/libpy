#pragma once

#include "libpy/detail/python.h"
#include "libpy/scoped_ref.h"

namespace py::detail {
template<typename T, typename base = PyObject>
struct autoclass_object : public base {
    T value;

    static T& unbox(base* ob) {
        return static_cast<autoclass_object*>(ob)->value;
    }

    static T& unbox(const py::scoped_ref<base>& self) {
        return unbox(self.get());
    }

    static T& unbox(const py::scoped_ref<autoclass_object>& self) {
        return unbox(static_cast<base*>(self));
    }

    template<typename b = base, typename = std::enable_if_t<!std::is_same_v<b, PyObject>>>
    static T& unbox(const py::scoped_ref<>& self) {
        return unbox(self.get());
    }

    template<typename b = base, typename = std::enable_if_t<!std::is_same_v<b, PyObject>>>
    static T& unbox(PyObject* self) {
        return unbox(reinterpret_cast<base*>(self));
    }
};
}  // namespace py::detail
