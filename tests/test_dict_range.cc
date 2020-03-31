#include <string>
#include <unordered_map>

#include "gtest/gtest.h"

#include "libpy/detail/python.h"
#include "libpy/dict_range.h"
#include "libpy/exception.h"
#include "test_utils.h"

namespace test_dict_range {
using namespace std::literals;

class dict_range : public with_python_interpreter {};

TEST_F(dict_range, iteration) {
    py::owned_ref ns = RUN_PYTHON(R"(
        dict_0 = {}
        dict_1 = {b'a': 1}
        dict_2 = {b'a': 1, b'b': 2, b'c': 3}
    )");
    ASSERT_TRUE(ns);

    {
        PyObject* dict_0 = PyDict_GetItemString(ns.get(), "dict_0");
        ASSERT_TRUE(dict_0);

        for ([[maybe_unused]] auto item : py::dict_range(dict_0)) {
            FAIL() << "empty dict should not enter the loop";
        }
    }

    {
        PyObject* dict_1 = PyDict_GetItemString(ns.get(), "dict_1");
        ASSERT_TRUE(dict_1);

        std::unordered_map<std::string, std::int64_t> expected = {{"a"s, 1}};
        std::unordered_map<std::string, std::int64_t> actual;

        for (auto [key, value] : py::dict_range(dict_1)) {
            actual.emplace(py::from_object<std::string>(key),
                           py::from_object<std::int64_t>(value));
        }

        EXPECT_EQ(actual, expected);
    }

    {
        PyObject* dict_2 = PyDict_GetItemString(ns.get(), "dict_2");
        ASSERT_TRUE(dict_2);

        std::unordered_map<std::string, std::int64_t> expected = {{"a"s, 1},
                                                                  {"b"s, 2},
                                                                  {"c"s, 3}};
        std::unordered_map<std::string, std::int64_t> actual;

        for (auto [key, value] : py::dict_range(dict_2)) {
            actual.emplace(py::from_object<std::string>(key),
                           py::from_object<std::int64_t>(value));
        }

        EXPECT_EQ(actual, expected);
    }
}

TEST_F(dict_range, not_a_dict) {
    py::owned_ref not_a_dict(PyList_New(0));
    ASSERT_TRUE(not_a_dict);

    EXPECT_THROW(py::dict_range::checked(not_a_dict), py::exception);
    PyErr_Clear();
}
}  // namespace test_dict_range
