#include "libpy/exception.h"

namespace py {
std::nullptr_t raise_from_cxx_exception(const std::exception& e) {
    if (!PyErr_Occurred()) {
        py::raise(PyExc_RuntimeError) << "a C++ exception was raised: " << e.what();
        return nullptr;
    }
    PyObject* type;
    PyObject* value;
    PyObject* tb;
    PyErr_Fetch(&type, &value, &tb);
    PyErr_NormalizeException(&type, &value, &tb);
    Py_XDECREF(tb);
    const char* what = e.what();
    if (!what[0]) {
        raise(type) << value << " (raised from C++ exception)";
    }
    else {
        raise(type) << value << " (raised from C++ exception: " << what << ')';
    }
    Py_DECREF(type);
    Py_DECREF(value);
    return nullptr;
}
}  // namespace py
