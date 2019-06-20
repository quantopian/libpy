#pragma once

#include <Python.h>

#include <libpy/scoped_ref.h>

namespace py::detail {
template<typename T>
struct autoclass_object : public PyObject {
    T value;

    static T& unbox(PyObject* ob) {
        return static_cast<autoclass_object*>(ob)->value;
    }

    static T& unbox(const py::scoped_ref<>& self) {
        return unbox(self.get());
    }

    static T& unbox(const py::scoped_ref<autoclass_object>& self) {
        return unbox(static_cast<PyObject*>(self));
    }

};
}  // namespace py::detail
