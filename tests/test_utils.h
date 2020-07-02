#pragma once

#include <sstream>
#include <string>

#include "gtest/gtest.h"

#include "libpy/call_function.h"
#include "libpy/detail/python.h"
#include "libpy/exception.h"
#include "libpy/itertools.h"
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

namespace py_test {

template<typename T>
std::array<T, 3> examples();

template<>
inline std::array<std::int64_t, 3> examples() {
    return {-200, 0, 1000};
}

template<>
inline std::array<std::string, 3> examples() {
    return {"foo", "", "arglebargle"};
}

template<>
inline std::array<std::array<char, 3>, 3> examples() {
    std::array<char, 3> foo{'f', 'o', 'o'};
    std::array<char, 3> bar{'b', 'a', 'r'};
    std::array<char, 3> baz{'b', 'a', 'z'};
    return {foo, bar, baz};
}

template<>
inline std::array<bool, 3> examples() {
    return {true, false, true};
}

template<>
inline std::array<double, 3> examples() {
    return {-1.0, -0.0, 100.0};
}

template<>
inline std::array<py::owned_ref<>, 3> examples() {
    Py_INCREF(Py_True);
    Py_INCREF(Py_False);
    Py_INCREF(Py_None);
    return {py::owned_ref<>(Py_True),
            py::owned_ref<>(Py_False),
            py::owned_ref<>(Py_None)};
}

template<typename M>
void test_map_to_object_impl(M m) {

    // Fill the map with some example values.
    auto it = py::zip(examples<typename M::key_type>(),
                      examples<typename M::mapped_type>());
    for (auto [key, value] : it) {
        m[key] = value;
    }

    auto check_python_map = [&](py::owned_ref<PyObject> ob) {
        ASSERT_TRUE(ob) << "to_object should not return null";
        EXPECT_TRUE(PyDict_Check(ob.get()));

        // Python map should be the same length as C++ map.
        Py_ssize_t len = PyDict_Size(ob.get());
        EXPECT_EQ(std::size_t(len), m.size())
            << "Python dict length should match C++ map length.";

        // Key/Value pairs in the python map should match the result of calling
        // to_object on each key/value pair in the C++ map.
        for (auto& [cxx_key, cxx_value] : m) {
            auto py_key = py::to_object(cxx_key);
            auto py_value = py::to_object(cxx_value);

            py::borrowed_ref result = PyDict_GetItem(ob.get(), py_key.get());
            ASSERT_TRUE(result) << "Key should have been in the map";

            bool values_equal =
                PyObject_RichCompareBool(py_value.get(), result.get(), Py_EQ);
            EXPECT_EQ(values_equal, 1) << "Dict values were not equal";
        }
    };

    // Check to_object with value, const value, and rvalue reference.

    py::owned_ref<PyObject> result = py::to_object(m);
    check_python_map(result);

    const M& const_ref = m;
    py::owned_ref<PyObject> constref_result = py::to_object(const_ref);
    check_python_map(constref_result);

    M copy = m;  // Make a copy before moving b/c the lambda above uses ``m``.
    py::owned_ref<PyObject> rvalueref_result = py::to_object(std::move(copy));
    check_python_map(rvalueref_result);
}

template<typename V>
void test_sequence_to_object_impl(V v) {
    auto check_python_list = [&](py::owned_ref<PyObject> ob) {
        ASSERT_TRUE(ob) << "to_object should not return null";
        EXPECT_EQ(PyList_Check(ob.get()), 1) << "ob should be a list";

        Py_ssize_t len = PyList_GET_SIZE(ob.get());
        EXPECT_EQ(std::size_t(len), v.size())
            << "Python list length should match C++ vector length.";

        // Values in Python list should be the result of calling to_object on each entry
        // in the C++ vector.
        for (auto [i, cxx_value] : py::enumerate(v)) {
            auto py_value = py::to_object(cxx_value);

            py::borrowed_ref result = PyList_GetItem(ob.get(), i);
            ASSERT_TRUE(result) << "Should have had a value at index " << i;

            bool values_equal =
                PyObject_RichCompareBool(py_value.get(), result.get(), Py_EQ);
            EXPECT_EQ(values_equal, 1)
                << "List values at index " << i << " were not equal";
        }
    };

    // Check to_object with value, const value, and rvalue reference.

    py::owned_ref<PyObject> result = py::to_object(v);
    check_python_list(result);

    const V& const_ref = v;
    py::owned_ref<PyObject> constref_result = py::to_object(const_ref);
    check_python_list(constref_result);

    V copy = v;  // Make a copy before moving b/c the lambda above uses ``v``.
    py::owned_ref<PyObject> rvalueref_result = py::to_object(std::move(copy));
    check_python_list(rvalueref_result);
}

template<typename V>
void test_set_to_object_impl(V v) {
    auto check_python_set = [&](py::owned_ref<PyObject> ob) {
        ASSERT_TRUE(ob) << "to_object should not return null";
        EXPECT_EQ(PySet_Check(ob.get()), 1) << "ob should be a set";

        Py_ssize_t len = PySet_GET_SIZE(ob.get());
        EXPECT_EQ(std::size_t(len), v.size())
            << "Python set length should match C++ set length.";

        // Values in Python list should be the result of calling to_object on each entry
        // in the C++ vector.
        for (auto cxx_value : v) {
            auto py_value = py::to_object(cxx_value);

            auto result = PySet_Contains(ob.get(), py_value.get());
            ASSERT_TRUE(result) << "Should contain value " << cxx_value;

            // bool values_equal =
            //     PyObject_RichCompareBool(py_value.get(), result.get(), Py_EQ);
            // EXPECT_EQ(values_equal, 1)
            //     << "List values at index " << i << " were not equal";
        }
    };

    // Check to_object with value, const value, and rvalue reference.

    py::owned_ref<PyObject> result = py::to_object(v);
    check_python_set(result);

    const V& const_ref = v;
    py::owned_ref<PyObject> constref_result = py::to_object(const_ref);
    check_python_set(constref_result);

    V copy = v;  // Make a copy before moving b/c the lambda above uses ``v``.
    py::owned_ref<PyObject> rvalueref_result = py::to_object(std::move(copy));
    check_python_set(rvalueref_result);
}

}  // namespace py_test
