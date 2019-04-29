#pragma once

#include <sstream>
#include <string>

#include "Python.h"
#include "gtest/gtest.h"

#include "libpy/exception.h"
#include "libpy/scoped_ref.h"
#include "libpy/util.h"

inline std::string format_current_python_exception() {
    PyObject* exc[3];
    PyErr_Fetch(&exc[0], &exc[1], &exc[2]);

    if (PyErr_ExceptionMatches(PyExc_SystemExit)) {
        // PyErr_PrintEx() will call exit (without printing) when the exception is a
        // SystemExit
        return "<SystemExit>";
    }

    py::scoped_ref sys(PyImport_ImportModule("sys"));
    if (!sys) {
        throw py::exception();
    }

    py::scoped_ref buf(PyObject_GetAttrString(sys.get(), "stderr"));
    if (!buf) {
        throw py::exception();
    }

    PyErr_Restore(exc[0], exc[1], exc[2]);
    PyErr_PrintEx(false);
    py::scoped_ref contents(PyObject_CallMethod(buf.get(), "getvalue", nullptr));
    if (!contents) {
        return "<unknown>";
    }
    return py::util::pystring_to_cstring(contents.get());
}

class with_python_interpreter : public testing::Test {
public:
    inline static void SetUpTestCase() {
        py::scoped_ref sys(PyImport_ImportModule("sys"));
        if (!sys) {
            throw py::exception();
        }
        py::scoped_ref io(PyImport_ImportModule("io"));
        if (!io) {
            throw py::exception();
        }
        py::scoped_ref buf(PyObject_CallMethod(io.get(), "StringIO", nullptr));
        if (!buf) {
            throw py::exception();
        }

        if (PyObject_SetAttrString(sys.get(), "stderr", buf.get())) {
            throw py::exception();
        }
    }

    virtual void TearDown() {
        EXPECT_FALSE(PyErr_Occurred()) << format_current_python_exception();
        PyErr_Clear();
    }
};

namespace detail {
inline py::scoped_ref<>
run_python(const std::string_view& python_source,
           const std::string_view& file,
           std::size_t line,
           py::scoped_ref<> python_namespace = py::scoped_ref(nullptr)) {
    if (!python_namespace) {
        py::scoped_ref main_module(PyImport_ImportModule("__main__"));
        if (!main_module) {
            throw py::exception();
        }
        py::scoped_ref main_dict(PyModule_GetDict(main_module.get()));
        if (!main_dict) {
            return nullptr;
        }

        if (!(python_namespace = py::scoped_ref(PyDict_New()))) {
            return nullptr;
        }

        if (PyDict_Update(python_namespace.get(), main_dict.get())) {
            return nullptr;
        }
    }

    py::scoped_ref io(PyImport_ImportModule("io"));
    if (!io) {
        return nullptr;
    }

    py::scoped_ref buf(PyObject_CallMethod(io.get(), "StringIO", nullptr));
    if (!buf) {
        return nullptr;
    }

    std::stringstream full_source;

    // the line number reported is the *last* line of the macro, we need to
    // subtract out the newlines from the python_source
    std::size_t lines_in_source =
        std::count(python_source.begin(), python_source.end(), '\n');

    // Add a bunch of newlines so that the errors in the tests correspond to
    // the line of the files. Subtract some lines to account for the code we
    // inject around the test source.
    for (std::size_t n = 0; n < line - lines_in_source - 2; ++n) {
        full_source << '\n';
    }

    // Put the user's test in a function. We share a module dict in this test
    // suite so assignments in a test should not bleed into other tests.
    full_source << "if True:\n" << python_source;
    py::scoped_ref code_object(
        Py_CompileString(full_source.str().data(), file.data(), Py_file_input));
    if (!code_object) {
        return nullptr;
    }

#if PY_MAJOR_VERSION == 2
#define LIBPY_CODE_CAST(x) reinterpret_cast<PyCodeObject*>(x)
#else
#define LIBPY_CODE_CAST(x) (x)
#endif

    py::scoped_ref result(PyEval_EvalCode(LIBPY_CODE_CAST(code_object.get()),
                                          python_namespace.get(),
                                          python_namespace.get()));

#undef LIBPY_CODE_CAST

    if (!result) {
        return nullptr;
    }
    return python_namespace;
}
}  // namespace detail

/** Run some Python code.

    @param python_source The Python source code to run.
    @param namespace (optional) The Python namespace to evaluate in.
    @return namespace The namespace after running the Python code.
*/
#define RUN_PYTHON(python_source, ...)                                                   \
    ::detail::run_python(python_source, __FILE__, __LINE__, ##__VA_ARGS__)
