#include <string>
#include <tuple>
#include <unordered_map>

#include "gtest/gtest.h"

#include "libpy/any.h"
#include "libpy/char_sequence.h"
#include "libpy/dense_hash_map.h"
#include "libpy/itertools.h"
#include "libpy/meta.h"
#include "libpy/numpy_utils.h"
#include "libpy/object_map_key.h"
#include "libpy/str_convert.h"
#include "libpy/to_object.h"
#include "test_utils.h"

namespace test_to_object {
using namespace std::literals;
using namespace py::cs::literals;

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
std::array<py::scoped_ref<>, 3> examples() {
    Py_INCREF(Py_True);
    Py_INCREF(Py_False);
    Py_INCREF(Py_None);
    return {py::scoped_ref<>(Py_True),
            py::scoped_ref<>(Py_False),
            py::scoped_ref<>(Py_None)};
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

            py::borrowed_ref result = PyDict_GetItem(ob.get(), py_key.get());
            ASSERT_TRUE(result) << "Key should have been in the map";

            bool values_equal = PyObject_RichCompareBool(py_value.get(), result.get(), Py_EQ);
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

TEST_F(to_object, map_to_object) {
    // NOTE: This test takes a long time to compile (about a .5s per entry in this
    // tuple). This is just enough coverage to test all three of our hash table types,
    // and a few important key/value types.
    auto maps = std::make_tuple(py::dense_hash_map<std::string, py::scoped_ref<PyObject>>(
                                    "missing_value"s),
                                py::sparse_hash_map<std::int64_t, std::array<char, 3>>(),
                                std::unordered_map<std::string, bool>());

    // Call test_map_to_object_impl on each entry in ``maps``.
    std::apply([&](auto... map) { (test_map_to_object_impl(map), ...); }, maps);
}

template<typename V>
void test_sequence_to_object_impl(V v) {
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

            py::borrowed_ref result = PyList_GetItem(ob.get(), i);
            ASSERT_TRUE(result) << "Should have had a value at index " << i;

            bool values_equal = PyObject_RichCompareBool(py_value.get(), result.get(), Py_EQ);
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

TEST_F(to_object, vector_to_object) {
    auto to_vec = [](const auto& arr) { return std::vector(arr.begin(), arr.end()); };
    auto vectors = std::make_tuple(to_vec(examples<std::string>()),
                                   to_vec(examples<double>()),
                                   to_vec(examples<py::scoped_ref<>>()));
    // Call test_sequence_to_object_impl on each entry in `vectors`.
    std::apply([&](auto... vec) { (test_sequence_to_object_impl(vec), ...); }, vectors);
}

TEST_F(to_object, array_to_object) {
    auto arrays = std::make_tuple(examples<std::string>(),
                                  examples<double>(),
                                  examples<py::scoped_ref<>>());
    // Call test_sequence_to_object_impl on each entry in `arrays`.
    std::apply([&](auto... arr) { (test_sequence_to_object_impl(arr), ...); }, arrays);
}

template<typename R, typename T>
auto test_any_ref_to_object(T value) {
    R ref(std::addressof(value), py::any_vtable::make<T>());

    auto ref_to_object = py::to_object(ref);
    ASSERT_TRUE(ref_to_object);
    auto value_to_object = py::to_object(value);
    ASSERT_TRUE(ref_to_object);

    int eq = PyObject_RichCompareBool(ref_to_object.get(), value_to_object.get(), Py_EQ);
    ASSERT_EQ(eq, 1);
}

TEST_F(to_object, any_ref) {
    test_any_ref_to_object<py::any_ref>(1);
    test_any_ref_to_object<py::any_ref>(1.5);
    test_any_ref_to_object<py::any_ref>("test"s);

    test_any_ref_to_object<py::any_cref>(1);
    test_any_ref_to_object<py::any_cref>(1.5);
    test_any_ref_to_object<py::any_cref>("test"s);
}

TEST_F(to_object, any_ref_of_object_refcnt) {
    PyObject* ob = Py_None;
    Py_ssize_t baseline_refcnt = Py_REFCNT(ob);
    {
        Py_INCREF(ob);
        py::scoped_ref sr(ob);
        ASSERT_EQ(Py_REFCNT(ob), baseline_refcnt + 1);

        py::any_ref ref(&sr, py::any_vtable::make<decltype(sr)>());
        // `any_ref` is *non-owning* so the reference count doesn't change
        ASSERT_EQ(Py_REFCNT(ob), baseline_refcnt + 1);
        {
            auto to_object_result = py::to_object(ref);
            ASSERT_TRUE(to_object_result);
            // `to_object` returns a new owning reference
            ASSERT_EQ(Py_REFCNT(ob), baseline_refcnt + 2);

            EXPECT_EQ(to_object_result.get(), ob);
        }

        // `to_object_result` goes out of scope, releasing its reference
        ASSERT_EQ(Py_REFCNT(ob), baseline_refcnt + 1);
    }

    // `sr` goes out of scope, releasing its reference
    ASSERT_EQ(Py_REFCNT(ob), baseline_refcnt);
}

TEST_F(to_object, any_ref_non_convertible_object) {
    // The most simple type which can be put into an `any_ref`. There is no `to_object`
    // dispatch, so we expect `to_object(S{})` would throw a runtime exception.
    struct S {
        bool operator==(const S&) const {
            return true;
        }

        bool operator!=(const S&) const {
            return false;
        }
    };

    S value;
    py::any_ref ref(&value, py::any_vtable::make<decltype(value)>());

    EXPECT_THROW(py::to_object(ref), py::exception);
    PyErr_Clear();
}

TEST_F(to_object, object_map_key) {
    py::object_map_key key{py::to_object(5)};
    ASSERT_TRUE(key);
    Py_ssize_t starting_ref_count = Py_REFCNT(key.get());

    py::scoped_ref as_ob = py::to_object(key);
    // should be the same pointer
    EXPECT_EQ(key.get(), as_ob.get());

    // now owned by both as_ob and key
    EXPECT_EQ(Py_REFCNT(key.get()), starting_ref_count + 1);
}

TEST_F(to_object, scoped_ref_nonstandard) {
    py::scoped_ref<PyArray_Descr> t = py::new_dtype<std::uint32_t>();

    py::scoped_ref ob = py::to_object(t);
    ASSERT_TRUE(ob);
    EXPECT_EQ(static_cast<PyObject*>(ob), static_cast<PyObject*>(t));
}

}  // namespace test_to_object
