#include <type_traits>

#include "libpy/any.h"
#include "libpy/autoclass.h"
#include "libpy/demangle.h"
#include "libpy/exception.h"
#include "libpy/from_object.h"
#include "libpy/ndarray_view.h"
#include "libpy/numpy_utils.h"
#include "libpy/object_map_key.h"
#include "libpy/scoped_ref.h"
#include "libpy/util.h"

#include "test_utils.h"

namespace test_from_object {
class from_object : public with_python_interpreter {};

template<typename T>
void check_integer(const py::scoped_ref<>& ob, T expected_value) {
    auto v = py::from_object<T>(ob);
    static_assert(std::is_same_v<decltype(v), T>);
    EXPECT_EQ(v, expected_value);
}

TEST_F(from_object, test_basic_integer) {
    // test a basic set of small integers
    for (std::size_t value = 0; value < 127; ++value) {
        py::scoped_ref ob(PyLong_FromLong(value));
        ASSERT_TRUE(ob) << "ob cannot be null";

        check_integer<std::int64_t>(ob, value);
        check_integer<std::int32_t>(ob, value);
        check_integer<std::int16_t>(ob, value);
        check_integer<std::int8_t>(ob, value);

        check_integer<std::uint64_t>(ob, value);
        check_integer<std::uint32_t>(ob, value);
        check_integer<std::uint16_t>(ob, value);
        check_integer<std::uint8_t>(ob, value);
    }
}

template<typename T>
void expect_overflow(const py::scoped_ref<>& ob) {
    EXPECT_THROW(py::from_object<T>(ob), py::invalid_conversion)
        << py::util::type_name<T>();
    EXPECT_TRUE(PyErr_Occurred());
    PyErr_Clear();
}

TEST_F(from_object, test_overflow) {
    py::scoped_ref py_one(PyLong_FromLong(1));
    ASSERT_TRUE(py_one) << "py_one cannot be null";
    py::scoped_ref py_neg_one(PyLong_FromLong(-1));
    ASSERT_TRUE(py_neg_one) << "py_neg_one cannot be null";

    // the types to check, the values in the tuple don't get read
    std::tuple<signed long long,
               signed long,
               signed int,
               signed short,
               signed char,
               unsigned long long,
               unsigned long,
               unsigned int,
               unsigned short,
               unsigned char>
        typed_values;

    auto check_limit = [&](auto limit_value, const py::scoped_ref<>& to_exceed) {
        using T = decltype(limit_value);

        py::scoped_ref<> limit_ob;

        if constexpr (std::is_signed_v<T>) {
            limit_ob = py::scoped_ref(PyLong_FromLongLong(limit_value));
        }
        else {
            limit_ob = py::scoped_ref(PyLong_FromUnsignedLongLong(limit_value));
        }
        ASSERT_TRUE(limit_ob) << "limit_ob cannot be null";

        // the limit value should be convertible
        check_integer<T>(limit_ob, limit_value);

        py::scoped_ref exceeds_limit(PyNumber_Add(limit_ob.get(), to_exceed.get()));
        ASSERT_TRUE(exceeds_limit) << "exceeds_limit cannot be null";

        // after adding `to_exceed` to the limit value, the result should overflow
        expect_overflow<T>(exceeds_limit);
    };

    auto check_type = [&](auto typed_value) {
        using T = decltype(typed_value);
        check_limit(std::numeric_limits<T>::max(), py_one);
        check_limit(std::numeric_limits<T>::min(), py_neg_one);
    };

    std::apply([&](auto... typed_value) { (check_type(typed_value), ...); },
               typed_values);
}

template<typename T>
void test_typed_array_view_bad_conversion(const py::scoped_ref<>& ndarray) {
    EXPECT_THROW(py::from_object<py::array_view<T>>(ndarray), py::exception)
        << py::util::type_name<T>();
    PyErr_Clear();
}

template<typename T, typename... IncorrectTypes>
void test_typed_array_view() {
    auto ndarray = py::move_to_numpy_array(
        std::vector<std::remove_const_t<T>>{0, 1, 2, 3});
    auto view = py::from_object<py::array_view<T>>(ndarray);

    std::remove_const_t<T> expected = 0;
    for (auto v : view) {
        EXPECT_EQ(v, expected);
        ++expected;
    }

    (test_typed_array_view_bad_conversion<IncorrectTypes>(ndarray), ...);
}

TEST_F(from_object, ndarray_view) {
    test_typed_array_view<std::int64_t, std::int8_t, std::int16_t, std::int32_t>();
    test_typed_array_view<std::int64_t,
                          const std::int8_t,
                          const std::int16_t,
                          const std::int32_t>();
    test_typed_array_view<const std::int64_t,
                          const std::int8_t,
                          const std::int16_t,
                          const std::int32_t>();

    test_typed_array_view<std::int32_t, std::int8_t, std::int16_t, std::int64_t>();
    test_typed_array_view<const std::int32_t,
                          const std::int8_t,
                          const std::int16_t,
                          const std::int64_t>();
    test_typed_array_view<std::int32_t,
                          const std::int8_t,
                          const std::int16_t,
                          const std::int64_t>();

    {
        auto ndarray = py::move_to_numpy_array(std::vector<std::int64_t>{0, 1, 2, 3});
        PyArray_CLEARFLAGS(reinterpret_cast<PyArrayObject*>(ndarray.get()),
                           NPY_ARRAY_WRITEABLE);

        EXPECT_THROW(py::from_object<py::array_view<std::int64_t>>(ndarray),
                     py::exception);
        PyErr_Clear();

        auto view = py::from_object<py::array_view<const std::int64_t>>(ndarray);

        std::int64_t expected = 0;
        for (auto v : view) {
            EXPECT_EQ(v, expected);
            ++expected;
        }
    }
}

TEST_F(from_object, ndarray_view_any_ref) {
    {
        auto ndarray = py::move_to_numpy_array(std::vector<std::int32_t>{0, 1, 2, 3});
        auto view = py::from_object<py::array_view<py::any_ref>>(ndarray);

        std::int32_t expected = 0;
        for (auto v : view) {
            EXPECT_EQ(v.cast<std::int32_t>(), expected);
            ++expected;
        }
    }

    {
        auto ndarray = py::move_to_numpy_array(std::vector<std::int64_t>{0, 1, 2, 3});
        auto view = py::from_object<py::array_view<py::any_ref>>(ndarray);

        std::int64_t expected = 0;
        for (auto v : view) {
            EXPECT_EQ(v.cast<std::int64_t>(), expected);
            ++expected;
        }
    }

    {
        auto ndarray = py::move_to_numpy_array(
            std::vector<py::datetime64ns>{py::datetime64ns(0),
                                          py::datetime64ns(1),
                                          py::datetime64ns(2),
                                          py::datetime64ns(3)});
        auto view = py::from_object<py::array_view<py::any_ref>>(ndarray);

        using namespace std::literals::chrono_literals;

        py::datetime64ns expected(0);
        for (auto v : view) {
            EXPECT_EQ(v.cast<py::datetime64ns>(), expected);
            expected += 1ns;
        }
    }

    {
        std::vector objects = {py::to_object(0),
                               py::to_object(1),
                               py::to_object(2),
                               py::to_object(3)};
        for (auto ob : objects) {
            ASSERT_TRUE(ob);
        }

        std::vector<py::scoped_ref<>> objects_cp;
        auto ndarray = py::move_to_numpy_array(std::move(objects_cp));
        auto view = py::from_object<py::array_view<py::any_ref>>(ndarray);

        std::size_t ix = 0;
        for (auto v : view) {
            // compare the underlying PyObject* which compares the objects on identity
            EXPECT_EQ(v.cast<py::scoped_ref<>>().get(), objects[ix].get());
            ++ix;
        }
    }

    {
        auto ndarray = py::move_to_numpy_array(std::vector<std::int64_t>{0, 1, 2, 3});
        PyArray_CLEARFLAGS(reinterpret_cast<PyArrayObject*>(ndarray.get()),
                           NPY_ARRAY_WRITEABLE);

        EXPECT_THROW(py::from_object<py::array_view<py::any_ref>>(ndarray),
                     py::exception);
        PyErr_Clear();

        auto view = py::from_object<py::array_view<py::any_cref>>(ndarray);

        std::int64_t expected = 0;
        for (auto v : view) {
            EXPECT_EQ(v, expected);
            ++expected;
        }
    }
}

TEST_F(from_object, object_map_key) {
    PyObject* ob = Py_None;
    Py_ssize_t starting_ref_count = Py_REFCNT(ob);

    py::object_map_key key = py::from_object<py::object_map_key>(ob);
    EXPECT_EQ(key.get(), ob);
    // the key owns a new reference
    EXPECT_EQ(Py_REFCNT(ob), starting_ref_count + 1);
}

TEST_F(from_object, to_path) {
#if PY_VERSION_HEX >= 0x03060000
    py::scoped_ref ns = RUN_PYTHON(R"(
        class C(object):
            def __fspath__(self):
                return "/tmp/"

        ob = C()
        good = "/tmp/"
        goodb = b"/tmp/"
        bad = 4
    )");
    ASSERT_TRUE(ns);
    std::filesystem::path test_path = "/tmp/";

    py::borrowed_ref<> ob = PyDict_GetItemString(ns.get(), "ob");
    ASSERT_TRUE(ob);
    EXPECT_EQ(test_path, py::from_object<std::filesystem::path>(ob));

    py::borrowed_ref<> good = PyDict_GetItemString(ns.get(), "good");
    ASSERT_TRUE(good);
    EXPECT_EQ(test_path, py::from_object<std::filesystem::path>(good));

    py::borrowed_ref<> goodb = PyDict_GetItemString(ns.get(), "goodb");
    ASSERT_TRUE(goodb);
    EXPECT_EQ(test_path, py::from_object<std::filesystem::path>(goodb));

    py::borrowed_ref<> bad = PyDict_GetItemString(ns.get(), "bad");
    ASSERT_TRUE(bad);
    EXPECT_THROW(py::from_object<std::filesystem::path>(bad), py::exception);
    expect_pyerr_type_and_message(PyExc_TypeError,
                                  "expected str, bytes or os.PathLike object, not int");
    PyErr_Clear();
#endif
}


TEST_F(from_object, autoclass_const_ref) {
    struct S {
        int val;

        S(int val) : val(val) {}
    };

    auto type = py::autoclass<S>("S").type();
    auto ob = py::autoclass<S>::construct(12);
    ASSERT_TRUE(ob);

    const auto& const_ref = py::from_object<const S&>(ob);
    EXPECT_EQ(const_ref.val, 12);

    auto& unboxed = py::autoclass<S>::unbox(ob);
    EXPECT_EQ(&const_ref, &unboxed);
}

TEST_F(from_object, autoclass_const_ref_wrong_type) {
    struct S {
        int val;

        S(int val) : val(val) {}
    };

    auto type = py::autoclass<S>("S").type();
    auto ob = py::to_object(12);
    ASSERT_TRUE(ob);

    EXPECT_THROW(py::from_object<const S&>(ob), py::invalid_conversion);
    PyErr_Clear();
}

TEST_F(from_object, autoclass_mut_ref) {
    struct S {
        int val;

        S(int val) : val(val) {}
    };

    auto type = py::autoclass<S>("S").type();
    auto ob = py::autoclass<S>::construct(12);
    ASSERT_TRUE(ob);

    auto& mut_ref = py::from_object<S&>(ob);
    EXPECT_EQ(mut_ref.val, 12);

    auto& unboxed = py::autoclass<S>::unbox(ob);
    EXPECT_EQ(&mut_ref, &unboxed);
}

TEST_F(from_object, autoclass_mut_ref_wrong_type) {
    struct S {
        int val;

        S(int val) : val(val) {}
    };

    auto type = py::autoclass<S>("S").type();
    auto ob = py::to_object(12);
    ASSERT_TRUE(ob);

    EXPECT_THROW(py::from_object<S&>(ob), py::invalid_conversion);
    PyErr_Clear();
}
}  // namespace test_from_object
