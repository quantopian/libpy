#include <string>
#include <tuple>
#include <unordered_map>

#include "gtest/gtest.h"

#include "libpy/any.h"
#include "libpy/char_sequence.h"
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

TEST_F(to_object, map_to_object) {
    auto map = std::unordered_map<std::string, bool>();
    py_test::test_map_to_object_impl(map);
}

TEST_F(to_object, vector_to_object) {
    auto to_vec = [](const auto& arr) { return std::vector(arr.begin(), arr.end()); };
    auto vectors = std::make_tuple(to_vec(py_test::examples<std::string>()),
                                   to_vec(py_test::examples<double>()),
                                   to_vec(py_test::examples<py::owned_ref<>>()));
    // Call test_sequence_to_object_impl on each entry in `vectors`.
    std::apply([&](auto... vec) { (py_test::test_sequence_to_object_impl(vec), ...); }, vectors);
}

TEST_F(to_object, array_to_object) {
    auto arrays = std::make_tuple(py_test::examples<std::string>(),
                                  py_test::examples<double>(),
                                  py_test::examples<py::owned_ref<>>());
    // Call test_sequence_to_object_impl on each entry in `arrays`.
    std::apply([&](auto... arr) { (py_test::test_sequence_to_object_impl(arr), ...); }, arrays);
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
        py::owned_ref sr(ob);
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

    py::owned_ref as_ob = py::to_object(key);
    // should be the same pointer
    EXPECT_EQ(key.get(), as_ob.get());

    // now owned by both as_ob and key
    EXPECT_EQ(Py_REFCNT(key.get()), starting_ref_count + 1);
}

TEST_F(to_object, owned_ref_nonstandard) {
    py::owned_ref<PyArray_Descr> t = py::new_dtype<std::uint32_t>();

    py::owned_ref ob = py::to_object(t);
    ASSERT_TRUE(ob);
    EXPECT_EQ(static_cast<PyObject*>(ob), static_cast<PyObject*>(t));
}

TEST_F(to_object, filesystem_path) {
    std::filesystem::path test_path = "/tmp/";
    py::owned_ref ob = py::to_object(test_path);
    ASSERT_TRUE(ob);
#if PY_VERSION_HEX >= 0x03040000
    py::owned_ref ns = RUN_PYTHON(R"(
        from pathlib import Path
        py_path = Path("/tmp/")
    )");
    ASSERT_TRUE(ns);

    py::borrowed_ref py_path_ob{PyDict_GetItemString(ns.get(), "py_path")};
    ASSERT_TRUE(py_path_ob);
#else
    py::owned_ref py_path_ob = py::to_object("/tmp/");

#endif
    int eq = PyObject_RichCompareBool(ob.get(), py_path_ob.get(), Py_EQ);
    EXPECT_EQ(eq, 1);
}

}  // namespace test_to_object
