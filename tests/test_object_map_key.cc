#include <unordered_map>

#include "gtest/gtest.h"

#include "libpy/exception.h"
#include "libpy/object_map_key.h"
#include "libpy/scoped_ref.h"
#include "libpy/to_object.h"
#include "test_utils.h"

namespace test_object_hash_key {
class object_map_key : public with_python_interpreter {};

TEST_F(object_map_key, eq) {
    py::object_map_key a{py::to_object(9001)};
    ASSERT_TRUE(a);

    py::object_map_key b{py::to_object(9001)};
    ASSERT_TRUE(b);

    // check that we aren't just checking pointer equality
    EXPECT_NE(a.get(), b.get());
    EXPECT_EQ(a, b);
}

TEST_F(object_map_key, eq_fails) {
    py::scoped_ref ns = RUN_PYTHON(R"(
        class C:
            def __eq__(self, other):
                raise ValueError()

        a = C()
        b = C()
    )");
    ASSERT_TRUE(ns);

    PyObject* a_ob = PyDict_GetItemString(ns.get(), "a");
    ASSERT_TRUE(a_ob);
    Py_INCREF(a_ob);
    py::object_map_key a{py::scoped_ref<>(a_ob)};

    PyObject* b_ob = PyDict_GetItemString(ns.get(), "b");
    ASSERT_TRUE(b_ob);
    Py_INCREF(b_ob);
    py::object_map_key b{py::scoped_ref<>(b_ob)};

    EXPECT_THROW(a == b, py::exception);
    PyErr_Clear();
}

TEST_F(object_map_key, hash) {
    int value = 10;
    py::object_map_key a{py::to_object(value)};
    ASSERT_TRUE(a);

    EXPECT_EQ(std::hash<py::object_map_key>{}(a), value);
}

TEST_F(object_map_key, hash_fails) {
    py::scoped_ref ns = RUN_PYTHON(R"(
        class C:
            def __hash__(self, other):
                raise ValueError()

        a = C()
    )");
    ASSERT_TRUE(ns);

    PyObject* a_ob = PyDict_GetItemString(ns.get(), "a");
    ASSERT_TRUE(a_ob);
    Py_INCREF(a_ob);
    py::object_map_key a{py::scoped_ref<>(a_ob)};

    EXPECT_THROW(std::hash<py::object_map_key>{}(a), py::exception);
    PyErr_Clear();
}

TEST_F(object_map_key, use_in_map) {
    std::unordered_map<py::object_map_key, int> map;

    int a_value = 10;
    py::object_map_key a_key{py::to_object(a_value)};
    ASSERT_TRUE(a_key);

    map[a_key] = a_value;

    EXPECT_EQ(map[a_key], a_value);

    int b_value = 20;
    // use a scoped ref to test implicit conversions
    py::scoped_ref b_key = py::to_object(b_value);
    ASSERT_TRUE(b_key);

    map[b_key] = b_value;

    EXPECT_EQ(map[b_key], b_value);
    EXPECT_EQ(map[a_key], a_value);

    int c_value = 30;
    py::scoped_ref c_key = py::to_object(c_value);
    ASSERT_TRUE(c_key);

    map.emplace(c_key, c_value);

    EXPECT_EQ(map[c_key], c_value);
    EXPECT_EQ(map[b_key], b_value);
    EXPECT_EQ(map[a_key], a_value);
}
}  // namespace test_object_hash_key
