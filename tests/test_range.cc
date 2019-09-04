#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "libpy/detail/python.h"
#include "libpy/exception.h"
#include "libpy/range.h"
#include "test_utils.h"

namespace test_dict_range {
using namespace std::literals;

class range : public with_python_interpreter {};

TEST_F(range, iteration) {
    py::scoped_ref ns = RUN_PYTHON(R"(
        it_0 = []
        it_1 = [1, 2, 3]
        it_2 = (1, 2, 3)

        def gen():
            yield 1
            yield 2
            yield 3

        it_3 = gen()
    )");
    ASSERT_TRUE(ns);

    {
        PyObject* it_0 = PyDict_GetItemString(ns.get(), "it_0");
        ASSERT_TRUE(it_0);

        for ([[maybe_unused]] auto item : py::range(it_0)) {
            FAIL() << "empty sequence should not enter the loop";
        }
    }

    {
        std::vector<std::int64_t> expected = {1, 2, 3};

        for (const auto name : {"it_1", "it_2", "it_3"}) {
            PyObject* it = PyDict_GetItemString(ns.get(), name);
            ASSERT_TRUE(it);

            std::vector<std::int64_t> actual;

            for (auto value : py::range(it)) {
                actual.emplace_back(py::from_object<std::int64_t>(value));
            }

            EXPECT_EQ(actual, expected);
        }
    }
}

TEST_F(range, not_iterable) {
    EXPECT_THROW(py::range{Py_None}, py::exception);
    expect_pyerr_type_and_message(PyExc_TypeError, "'NoneType' object is not iterable");
    PyErr_Clear();
}

TEST_F(range, exception_in_next) {
    py::scoped_ref ns = RUN_PYTHON(R"(
        def gen():
            yield 1
            yield 2
            raise ValueError('die, you evil C++ program')

        it = gen()
    )");
    ASSERT_TRUE(ns);

    PyObject* it = PyDict_GetItemString(ns.get(), "it");
    ASSERT_TRUE(it);

    std::vector<std::int64_t> expected = {1, 2};
    std::vector<std::int64_t> actual;
    EXPECT_THROW(
        {
            for (auto value : py::range(it)) {
                actual.emplace_back(py::from_object<std::int64_t>(value));
            }
        },
        py::exception);
    expect_pyerr_type_and_message(PyExc_ValueError, "die, you evil C++ program");
    PyErr_Clear();

    EXPECT_EQ(actual, expected);
}
}  // namespace test_dict_range
