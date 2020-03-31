#include "libpy/exception.h"

namespace py {
namespace {
void deep_what_recursive_helper(std::string& out, const std::exception& e, int level) {
    if (level) {
        out.push_back('\n');
    }
    out.insert(out.end(), level * 2, ' ');
    const char* what = e.what();
    if (*what) {
        if (level) {
            out += "raised from: ";
        }
        out += what;
    }
    else {
        out += "raised from exception with empty what()";
    }
    try {
        std::rethrow_if_nested(e);
    }
    catch (const std::exception& next) {
        deep_what_recursive_helper(out, next, level + 1);
    }
}

std::string deep_what(const std::exception& e) {
    std::string out;
    deep_what_recursive_helper(out, e, 0);
    return out;
}
}  // namespace

std::nullptr_t raise_from_cxx_exception(const std::exception& e) {
    if (!PyErr_Occurred()) {
        py::raise(PyExc_RuntimeError) << "a C++ exception was raised: " << deep_what(e);
        return nullptr;
    }
    if (dynamic_cast<const py::exception*>(&e)) {
        // this already raised an exception with the message we want to show to
        // Python
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
        raise(type) << value << "; raised from C++ exception";
    }
    else {
        raise(type) << value << "; raised from C++ exception: " << deep_what(e);
    }
    Py_DECREF(type);
    Py_DECREF(value);
    return nullptr;
}

std::string exception::msg_from_current_pyexc() {
    PyObject* type;
    PyObject* value;
    PyObject* tb;
    PyErr_Fetch(&type, &value, &tb);
    PyErr_NormalizeException(&type, &value, &tb);

    py::owned_ref as_str(PyObject_Str(value));
    std::string out;
    if (!as_str) {
        out = "<failed to convert Python message to C++ message>";
    }
    else {
        out = reinterpret_cast<PyTypeObject*>(type)->tp_name;
        out += ": ";
        out += py::util::pystring_to_string_view(as_str);
    }
    PyErr_Restore(type, value, tb);
    return out;
}
}  // namespace py
