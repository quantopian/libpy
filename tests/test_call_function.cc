#include "test_utils.h"

#include "libpy/call_function.h"

namespace test_call_function {
using namespace std::literals;
class call_function : public with_python_interpreter {};

TEST_F(call_function, basic) {
    py::scoped_ref ns = RUN_PYTHON(R"(
        def f(a, b):
            return a + b
    )");
    ASSERT_TRUE(ns);

    PyObject* f = PyDict_GetItemString(ns.get(), "f");
    ASSERT_TRUE(f);

    // Python functions are duck-typed, `f` should be callable with both ints and strings
    // (and more)
    {
        auto result_ob = py::call_function(f, 1, 2);
        ASSERT_TRUE(result_ob);
        EXPECT_EQ(py::from_object<std::int64_t>(result_ob), 3);
    }

    {
        auto result_ob = py::call_function(f, "abc", "def");
        ASSERT_TRUE(result_ob);
        EXPECT_EQ(py::from_object<std::string>(result_ob), "abcdef"s);
    }
}

TEST_F(call_function, exception) {
    py::scoped_ref ns = RUN_PYTHON(R"(
        def f():
            raise ValueError('ayy lmao');
    )");
    ASSERT_TRUE(ns);

    PyObject* f = PyDict_GetItemString(ns.get(), "f");
    ASSERT_TRUE(f);

    py::scoped_ref<> result = py::call_function(f);
    EXPECT_FALSE(result);
    expect_pyerr_type_and_message(PyExc_ValueError, "ayy lmao");
    PyErr_Clear();
}

TEST_F(call_function, method) {
    py::scoped_ref ns = RUN_PYTHON(R"(
        class C(object):
            def __init__(self):
                self.a = 1

            def f(self, b):
                return self.a + b

        ob = C()
    )");
    ASSERT_TRUE(ns);

    PyObject* ob = PyDict_GetItemString(ns.get(), "ob");
    ASSERT_TRUE(ob);

    // Python functions are duck-typed, `f` should be callable with both ints and doubles
    // (and more)
    {
        auto result_ob = py::call_method(ob, "f", 2);
        ASSERT_TRUE(result_ob);
        EXPECT_EQ(py::from_object<std::int64_t>(result_ob), 3);
    }

    {
        auto result_ob = py::call_method(ob, "f", 2.5);
        ASSERT_TRUE(result_ob);
        EXPECT_EQ(py::from_object<double>(result_ob), 3.5);
    }
}

TEST_F(call_function, method_exception) {
    py::scoped_ref ns = RUN_PYTHON(R"(
        class C(object):
            def f(self):
                raise ValueError('ayy lmao')

        ob = C()
    )");
    ASSERT_TRUE(ns);

    PyObject* ob = PyDict_GetItemString(ns.get(), "ob");
    ASSERT_TRUE(ob);

    py::scoped_ref<> result = py::call_method(ob, "f");
    EXPECT_FALSE(result);
    expect_pyerr_type_and_message(PyExc_ValueError, "ayy lmao");
    PyErr_Clear();

    // should still throw if the method doesn't exist
    result = py::call_method(ob, "g");
    EXPECT_FALSE(result);
    expect_pyerr_type_and_message(PyExc_AttributeError,
                                  "'C' object has no attribute 'g'");
    PyErr_Clear();
}

class call_function_throws : public with_python_interpreter {};

TEST_F(call_function_throws, basic) {
    py::scoped_ref ns = RUN_PYTHON(R"(
        def f(a, b):
            return a + b
    )");
    ASSERT_TRUE(ns);

    PyObject* f = PyDict_GetItemString(ns.get(), "f");
    ASSERT_TRUE(f);

    // Python functions are duck-typed, `f` should be callable with both ints and strings
    // (and more)
    {
        auto result_ob = py::call_function_throws(f, 1, 2);
        ASSERT_TRUE(result_ob);
        EXPECT_EQ(py::from_object<std::int64_t>(result_ob), 3);
    }

    {
        auto result_ob = py::call_function_throws(f, "abc", "def");
        ASSERT_TRUE(result_ob);
        EXPECT_EQ(py::from_object<std::string>(result_ob), "abcdef"s);
    }
}

TEST_F(call_function_throws, exception) {
    py::scoped_ref ns = RUN_PYTHON(R"(
        def f():
            raise ValueError('ayy lmao');
    )");
    ASSERT_TRUE(ns);

    PyObject* f = PyDict_GetItemString(ns.get(), "f");
    ASSERT_TRUE(f);

    EXPECT_THROW(py::call_function_throws(f), py::exception);
    expect_pyerr_type_and_message(PyExc_ValueError, "ayy lmao");
    PyErr_Clear();
}

TEST_F(call_function_throws, method) {
    py::scoped_ref ns = RUN_PYTHON(R"(
        class C(object):
            def __init__(self):
                self.a = 1

            def f(self, b):
                return self.a + b

        ob = C()
    )");
    ASSERT_TRUE(ns);

    PyObject* ob = PyDict_GetItemString(ns.get(), "ob");
    ASSERT_TRUE(ob);

    // Python functions are duck-typed, `f` should be callable with both ints and doubles
    // (and more)
    {
        auto result_ob = py::call_method_throws(ob, "f", 2);
        ASSERT_TRUE(result_ob);
        EXPECT_EQ(py::from_object<std::int64_t>(result_ob), 3);
    }

    {
        auto result_ob = py::call_method_throws(ob, "f", 2.5);
        ASSERT_TRUE(result_ob);
        EXPECT_EQ(py::from_object<double>(result_ob), 3.5);
    }
}

TEST_F(call_function_throws, method_exception) {
    py::scoped_ref ns = RUN_PYTHON(R"(
        class C(object):
            def f(self):
                raise ValueError('ayy lmao')

        ob = C()
    )");
    ASSERT_TRUE(ns);

    PyObject* ob = PyDict_GetItemString(ns.get(), "ob");
    ASSERT_TRUE(ob);

    EXPECT_THROW(py::call_method_throws(ob, "f"), py::exception);
    expect_pyerr_type_and_message(PyExc_ValueError, "ayy lmao");
    PyErr_Clear();
}

}  // namespace test_call_function
