#pragma once

#include <Python.h>

#include <libpy/scoped_ref.h>

namespace py::detail {
template<typename T>
struct autoclass_object {
    PyObject base;
    T cxx_ob;

    static T& unbox(PyObject* ob) {
        return reinterpret_cast<autoclass_object*>(ob)->cxx_ob;
    }

    static T& unbox(const py::scoped_ref<>& self) {
        return unbox(self.get());
    }

    static T& unbox(const py::scoped_ref<autoclass_object>& self) {
        return unbox(reinterpret_cast<PyObject*>(self.get()));
    }

};
}  // namespace py::detail
