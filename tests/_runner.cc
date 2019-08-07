#include "gtest/gtest.h"

#include "libpy/detail/python.h"
#define LIBPY_TEST_MAIN
#include "libpy/numpy_utils.h"

#include "libpy/automethod.h"
#include "libpy/csv.h"

namespace test {
PyObject* run_tests(PyObject*, PyObject* py_argv) {
    if (!PyTuple_Check(py_argv)) {
        PyErr_SetString(PyExc_TypeError, "py_argv must be a tuple of strings");
        return nullptr;
    }
    std::vector<char*> argv;
    for (Py_ssize_t ix = 0; ix < PyTuple_GET_SIZE(py_argv); ++ix) {
        PyObject* cs = PyTuple_GET_ITEM(py_argv, ix);
        if (!PyBytes_Check(cs)) {
            PyErr_SetString(PyExc_TypeError, "py_argv must be a tuple of strings");
            return nullptr;
        }
        argv.push_back(PyBytes_AS_STRING(cs));
    }
    int argc = argv.size();
    argv.push_back(nullptr);
    testing::InitGoogleTest(&argc, argv.data());
    // print a newline to start output fresh from the partial line that pytest starts
    // us with
    std::cout << '\n';
    int out = RUN_ALL_TESTS();
    PyErr_Clear();
    return PyLong_FromLong(out);
}

PyMethodDef methods[] = {
    {"run_tests", run_tests, METH_O, nullptr},
    py::automethod<py::csv::py_parse>("parse_csv"),
    {nullptr, nullptr, 0, nullptr},
};

#if PY_MAJOR_VERSION == 2
PyMODINIT_FUNC init_runner() {
    import_array();
    PyObject* mod = Py_InitModule("_runner", methods);
    if (!mod) {
        return;
    }
    py::csv::add_parser_pytypes(mod);
}
#else
PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_runner",
    nullptr,
    -1,
    methods,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};

PyMODINIT_FUNC PyInit__runner() {
    import_array();
    py::scoped_ref mod(PyModule_Create(&module));
    if (!mod) {
        return nullptr;
    }
    if (py::csv::add_parser_pytypes(mod.get())) {
        return nullptr;
    }
    return std::move(mod).escape();
}
#endif
}  // namespace test
