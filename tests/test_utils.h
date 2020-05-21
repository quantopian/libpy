#pragma once

#include <sstream>
#include <string>

#include "gtest/gtest.h"

#include "libpy/call_function.h"
#include "libpy/detail/python.h"
#include "libpy/exception.h"
#include "libpy/owned_ref.h"
#include "libpy/util.h"

inline void gc_collect() {
    while (PyGC_Collect())
        ;
}

inline void expect_pyerr_type_and_message(PyObject* ptype, std::string_view pmsg) {
    ASSERT_TRUE(PyErr_Occurred()) << "no exception was thrown";
    EXPECT_TRUE(PyErr_ExceptionMatches(ptype));

    PyObject *type, *value, *traceback;
    PyErr_Fetch(&type, &value, &traceback);
    PyErr_Restore(type, value, traceback);

    py::owned_ref py_msg(PyObject_Str(value));
    ASSERT_TRUE(py_msg);
    std::string_view c_msg = py::util::pystring_to_string_view(py_msg);

    // convert the values to std::string to get *much* nicer error output
    EXPECT_EQ(std::string{c_msg}, std::string{pmsg});
}

inline std::string format_current_python_exception() {
    PyObject* exc[3];
    PyErr_Fetch(&exc[0], &exc[1], &exc[2]);

    if (PyErr_ExceptionMatches(PyExc_SystemExit)) {
        // PyErr_PrintEx() will call exit (without printing) when the exception is a
        // SystemExit
        return "<SystemExit>";
    }

    py::owned_ref sys(PyImport_ImportModule("sys"));
    if (!sys) {
        throw py::exception();
    }

    py::owned_ref buf(PyObject_GetAttrString(sys.get(), "stderr"));
    if (!buf) {
        throw py::exception();
    }

    PyErr_Restore(exc[0], exc[1], exc[2]);
    PyErr_PrintEx(false);
    py::owned_ref contents = py::call_method(buf, "getvalue");
    if (!contents) {
        return "<unknown>";
    }
    return py::util::pystring_to_cstring(contents.get());
}

class with_python_interpreter : public testing::Test {
public:
    virtual void SetUp() override {
        py::owned_ref sys(PyImport_ImportModule("sys"));
        if (!sys) {
            throw py::exception();
        }
        py::owned_ref io_module(PyImport_ImportModule("io"));
        if (!io_module) {
            throw py::exception();
        }
        py::owned_ref buf = py::call_method(io_module, "StringIO");
        if (!buf) {
            throw py::exception();
        }

        if (PyObject_SetAttrString(sys.get(), "stderr", buf.get())) {
            throw py::exception();
        }
    }

    virtual void TearDown() override {
        EXPECT_FALSE(PyErr_Occurred()) << format_current_python_exception();
        PyErr_Clear();
    }
};

namespace detail {
inline py::owned_ref<> run_python(
    const std::string_view& python_source,
    const std::string_view& file,
    std::size_t line,
    bool eval,
    const std::unordered_map<std::string, py::owned_ref<>>& python_namespace = {}) {
    py::owned_ref py_ns{PyDict_New()};

    for (const auto& [k, v] : python_namespace) {
        if (PyDict_SetItemString(py_ns.get(), k.data(), v.get())) {
            return nullptr;
        }
    }

    py::owned_ref main_module(PyImport_ImportModule("__main__"));
    if (!main_module) {
        return nullptr;
    }

    // PyModule_GetDict returns a borrowed reference
    PyObject* main_dict = PyModule_GetDict(main_module.get());
    if (!main_dict) {
        return nullptr;
    }

    if (PyDict_Update(py_ns.get(), main_dict)) {
        return nullptr;
    }

    py::owned_ref io(PyImport_ImportModule("io"));
    if (!io) {
        return nullptr;
    }

    py::owned_ref buf = py::call_method(io, "StringIO");
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
    for (std::size_t n = 0; n < line - lines_in_source - 2 - eval; ++n) {
        full_source << '\n';
    }

    // Put the user's test in a function. We share a module dict in this test
    // suite so assignments in a test should not bleed into other tests.
    full_source << "if True:\n";
    if (eval) {
        full_source << "    __libpy_output = \\\n";
    }
    full_source << python_source;
    py::owned_ref code_object(
        Py_CompileString(full_source.str().data(), file.data(), Py_file_input));
    if (!code_object) {
        return nullptr;
    }

    py::owned_ref result(PyEval_EvalCode(code_object.get(), py_ns.get(), py_ns.get()));

    if (!result) {
        return nullptr;
    }
    if (!eval) {
        return py_ns;
    }
    return py::owned_ref<>::xnew_reference(
        PyDict_GetItemString(py_ns.get(), "__libpy_output"));
}
}  // namespace detail

/** Run some Python code.

    @param python_source The Python source code to run.
    @param namespace (optional) The Python namespace to evaluate in, this should be
           an `unordered_map<std::string, py::owned_ref<>>`
    @return namespace The namespace after running the Python code.
*/
#define RUN_PYTHON(python_source, ...)                                                   \
    ::detail::run_python(python_source, __FILE__, __LINE__, false, ##__VA_ARGS__)

/** Eval some Python code.

    @param python_source The Python source code to evaluate.
    @param namespace (optional) The Python namespace to evaluate in, this should be
           an `unordered_map<std::string, py::owned_ref<>>`
    @return The result of evaluating the given Python expression.
*/
#define EVAL_PYTHON(python_source, ...)                                                  \
    ::detail::run_python(python_source, __FILE__, __LINE__, true, ##__VA_ARGS__)
