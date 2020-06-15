#include <string>
#include <vector>

#include <libpy/abi.h>
#include <libpy/automethod.h>
#include <libpy/exception.h>

namespace libpy_tutorial {

void raise_a_value_error() {
    auto reason = "wargl bargle";
    py::raise(PyExc_ValueError) << "failed to do something because: " << reason;
    // do some other things

    throw py::exception{};
}

void throw_value_error(int a) {
    throw py::exception(PyExc_ValueError, "You passed ", a, " and this is the exception");
}

namespace {
PyMethodDef methods[] = {
    py::autofunction<raise_a_value_error>("raise_a_value_error"),
    py::autofunction<throw_value_error>("throw_value_error"),
    py::end_method_list,
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "libpy_tutorial.exceptions",
    nullptr,
    -1,
    methods,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};

PyMODINIT_FUNC PyInit_exceptions() {
    if (py::abi::ensure_compatible_libpy_abi()) {
        return nullptr;
    }
    import_array();
    return PyModule_Create(&module);
}
}  // namespace
}  // namespace libpy_tutorial
