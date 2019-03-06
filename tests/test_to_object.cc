#include <string>
#include <tuple>
#include <unordered_map>

#include "gtest/gtest.h"

#include "libpy/dense_hash_map.h"
#include "libpy/itertools.h"
#include "libpy/meta.h"
#include "libpy/to_object.h"
#include "test_utils.h"

namespace test_to_object {
class to_object : public with_python_interpreter {};

template<typename T>
std::array<T, 3> examples();

template<>
std::array<std::int64_t, 3> examples() {
    return {-200, 0, 1000};
}

template<>
std::array<std::string, 3> examples() {
    return {"foo", "", "arglebargle"};
}

template<>
std::array<std::array<char, 3>, 3> examples() {
    std::array<char, 3> foo{'f', 'o', 'o'};
    std::array<char, 3> bar{'b', 'a', 'r'};
    std::array<char, 3> baz{'b', 'a', 'z'};
    return {foo, bar, baz};
}

template<>
std::array<bool, 3> examples() {
    return {true, false, true};
}

template<>
std::array<double, 3> examples() {
    return {-1.0, -0.0, 100.0};
}

template<>
std::array<py::scoped_ref<PyObject>, 3> examples() {
    PyObject* true_ = Py_True;
    PyObject* false_ = Py_False;
    PyObject* none = Py_None;

    Py_INCREF(true_);
    Py_INCREF(false_);
    Py_INCREF(none);

    return {
        py::scoped_ref<PyObject>(true_),
        py::scoped_ref<PyObject>(false_),
        py::scoped_ref<PyObject>(none),
    };
}

template<typename M>
void test_map_to_object_impl(M m) {

    // Fill the map with some example values.
    auto it = py::zip(examples<typename M::key_type>(),
                      examples<typename M::mapped_type>());
    for (auto [key, value] : it) {
        m[key] = value;
    }

    auto check_python_map = [&](py::scoped_ref<PyObject> ob) {
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

            PyObject* result = PyDict_GetItem(ob.get(), py_key.get());
            ASSERT_TRUE(result) << "Key should have been in the map";

            bool values_equal = PyObject_RichCompareBool(py_value.get(), result, Py_EQ);
            EXPECT_EQ(values_equal, 1) << "Dict values were not equal";
        }
    };

    // Check to_object with value, const value, and rvalue reference.

    py::scoped_ref<PyObject> result = py::to_object(m);
    check_python_map(result);

    const M& const_ref = m;
    py::scoped_ref<PyObject> constref_result = py::to_object(const_ref);
    check_python_map(constref_result);

    M copy = m;  // Make a copy before moving b/c the lambda above uses ``m``.
    py::scoped_ref<PyObject> rvalueref_result = py::to_object(std::move(copy));
    check_python_map(rvalueref_result);
}

TEST_F(to_object, test_map_to_object) {
    // NOTE: This test takes a long time to compile (about a .5s per entry in this
    // tuple). This is just enough coverage to test all three of our hash table types,
    // and a few important key/value types.
    auto maps = std::make_tuple(py::dense_hash_map<std::string, py::scoped_ref<PyObject>>(
                                    std::string("missing_value")),
                                py::sparse_hash_map<std::int64_t, std::array<char, 3>>(),
                                std::unordered_map<std::string, bool>());

    // Call test_map_to_object_impl on each entry in ``maps``.
    std::apply([&](auto... map) { (test_map_to_object_impl(map), ...); }, maps);
}

template<typename V>
void test_vector_to_object_impl(V v) {
    for (auto k : examples<typename V::value_type>()) {
        v.push_back(k);
    }

    auto check_python_list = [&](py::scoped_ref<PyObject> ob) {
        ASSERT_TRUE(ob) << "to_object should not return null";
        EXPECT_EQ(PyList_Check(ob.get()), 1) << "ob should be a list";

        Py_ssize_t len = PyList_GET_SIZE(ob.get());
        EXPECT_EQ(std::size_t(len), v.size())
            << "Python list length should match C++ vector length.";

        // Values in Python list should be the result of calling to_object on each entry
        // in the C++ vector.
        for (auto [i, cxx_value] : py::enumerate(v)) {
            auto py_value = py::to_object(cxx_value);

            PyObject* result = PyList_GetItem(ob.get(), i);
            ASSERT_TRUE(result) << "Should have had a value at index " << i;

            bool values_equal = PyObject_RichCompareBool(py_value.get(), result, Py_EQ);
            EXPECT_EQ(values_equal, 1)
                << "List values at index " << i << " were not equal";
        }
    };

    // Check to_object with value, const value, and rvalue reference.

    py::scoped_ref<PyObject> result = py::to_object(v);
    check_python_list(result);

    const V& const_ref = v;
    py::scoped_ref<PyObject> constref_result = py::to_object(const_ref);
    check_python_list(constref_result);

    V copy = v;  // Make a copy before moving b/c the lambda above uses ``v``.
    py::scoped_ref<PyObject> rvalueref_result = py::to_object(std::move(copy));
    check_python_list(rvalueref_result);
}

TEST_F(to_object, test_vector_to_object) {
    auto vectors = std::make_tuple(std::vector<std::string>(),
                                   std::vector<double>(),
                                   std::vector<py::scoped_ref<PyObject>>());
    // Call test_vector_to_object_impl on each entry in ``vectors``.
    std::apply([&](auto... vec) { (test_vector_to_object_impl(vec), ...); }, vectors);
}

}  // namespace test_to_object
