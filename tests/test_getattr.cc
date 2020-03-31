#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "libpy/detail/python.h"
#include "libpy/getattr.h"
#include "test_utils.h"

namespace test_getattr {
class getattr : public with_python_interpreter {};

TEST_F(getattr, simple) {
    py::owned_ref ns = RUN_PYTHON(R"(
        class A(object):
            b = 1

        expected = A.b
    )");
    ASSERT_TRUE(ns);

    py::borrowed_ref A = PyDict_GetItemString(ns.get(), "A");
    ASSERT_TRUE(A);

    py::owned_ref<> actual = py::getattr(A, "b");
    ASSERT_TRUE(actual);

    py::borrowed_ref expected = PyDict_GetItemString(ns.get(), "expected");
    ASSERT_TRUE(expected);

    // compare them using object identity
    EXPECT_EQ(actual, expected);
}

TEST_F(getattr, attribute_error) {
    py::owned_ref ns = RUN_PYTHON(R"(
        class A(object):
            pass
    )");
    ASSERT_TRUE(ns);

    py::borrowed_ref A = PyDict_GetItemString(ns.get(), "A");
    ASSERT_TRUE(A);

    py::owned_ref<> actual = py::getattr(A, "b");
    ASSERT_FALSE(actual);
    expect_pyerr_type_and_message(PyExc_AttributeError,
                                  "type object 'A' has no attribute 'b'");
    PyErr_Clear();
}

TEST_F(getattr, nested) {
    py::owned_ref ns = RUN_PYTHON(R"(
        class A(object):
            class B(object):
                class C(object):
                    d = 1

        expected = A.B.C.d
    )");
    ASSERT_TRUE(ns);

    py::borrowed_ref A = PyDict_GetItemString(ns.get(), "A");
    ASSERT_TRUE(A);

    py::owned_ref<> actual = py::nested_getattr(A, "B", "C", "d");
    ASSERT_TRUE(actual);

    py::borrowed_ref expected = PyDict_GetItemString(ns.get(), "expected");
    ASSERT_TRUE(expected);

    // compare them using object identity
    EXPECT_EQ(actual, expected);
}

TEST_F(getattr, nested_failure) {
    py::owned_ref ns = RUN_PYTHON(R"(
        class A(object):
            class B(object):
                class C(object):
                    pass
    )");
    ASSERT_TRUE(ns);

    py::borrowed_ref A = PyDict_GetItemString(ns.get(), "A");
    ASSERT_TRUE(A);

    // attempt to access a few fields past the end of the real attribute chain.
    py::owned_ref<> actual = py::nested_getattr(A, "B", "C", "D", "E");
    ASSERT_FALSE(actual);
    expect_pyerr_type_and_message(PyExc_AttributeError,
                                  "type object 'C' has no attribute 'D'");
    PyErr_Clear();
}

class getattr_throws : public with_python_interpreter {};

TEST_F(getattr_throws, simple) {
    py::owned_ref ns = RUN_PYTHON(R"(
        class A(object):
            b = 1

        expected = A.b
    )");
    ASSERT_TRUE(ns);

    py::borrowed_ref A = PyDict_GetItemString(ns.get(), "A");
    ASSERT_TRUE(A);

    py::owned_ref<> actual = py::getattr_throws(A, "b");
    ASSERT_TRUE(actual);

    py::borrowed_ref expected = PyDict_GetItemString(ns.get(), "expected");
    ASSERT_TRUE(expected);

    // compare them using object identity
    EXPECT_EQ(actual, expected);
}

TEST_F(getattr_throws, attribute_error) {
    py::owned_ref ns = RUN_PYTHON(R"(
        class A(object):
            pass
    )");
    ASSERT_TRUE(ns);

    py::borrowed_ref A = PyDict_GetItemString(ns.get(), "A");
    ASSERT_TRUE(A);

    EXPECT_THROW(py::getattr_throws(A, "b"), py::exception);
    expect_pyerr_type_and_message(PyExc_AttributeError,
                                  "type object 'A' has no attribute 'b'");
    PyErr_Clear();
}

TEST_F(getattr_throws, nested) {
    py::owned_ref ns = RUN_PYTHON(R"(
        class A(object):
            class B(object):
                class C(object):
                    d = 1

        expected = A.B.C.d
    )");
    ASSERT_TRUE(ns);

    py::borrowed_ref A = PyDict_GetItemString(ns.get(), "A");
    ASSERT_TRUE(A);

    py::owned_ref<> actual = py::nested_getattr_throws(A, "B", "C", "d");
    ASSERT_TRUE(actual);

    py::borrowed_ref expected = PyDict_GetItemString(ns.get(), "expected");
    ASSERT_TRUE(expected);

    // compare them using object identity
    EXPECT_EQ(actual, expected);
}

TEST_F(getattr_throws, nested_failure) {
    py::owned_ref ns = RUN_PYTHON(R"(
        class A(object):
            class B(object):
                class C(object):
                    pass
    )");
    ASSERT_TRUE(ns);

    py::borrowed_ref A = PyDict_GetItemString(ns.get(), "A");
    ASSERT_TRUE(A);

    // attempt to access a few fields past the end of the real attribute chain.
    EXPECT_THROW(py::nested_getattr_throws(A, "B", "C", "D", "E"), py::exception);
    expect_pyerr_type_and_message(PyExc_AttributeError,
                                  "type object 'C' has no attribute 'D'");
    PyErr_Clear();
}
}  // namespace test_getattr
