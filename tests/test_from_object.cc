#include <type_traits>

#include "libpy/any.h"
#include "libpy/demangle.h"
#include "libpy/exception.h"
#include "libpy/from_object.h"
#include "libpy/numpy_utils.h"
#include "libpy/scoped_ref.h"

#include "test_utils.h"

namespace test_from_object {
class from_object : public with_python_interpreter {};

template<typename T>
void check_integer(py::scoped_ref<PyObject>& ob, T expected_value) {
    auto v = py::from_object<T>(ob);
    static_assert(std::is_same_v<decltype(v), T>);
    EXPECT_EQ(v, expected_value);
}

TEST_F(from_object, test_basic_integer) {
    // test a basic set of small integers
    for (std::size_t value = 0; value < 127; ++value) {
        auto ob = py::scoped_ref(PyLong_FromLong(value));
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
void expect_overflow(py::scoped_ref<PyObject>& ob) {
    EXPECT_THROW(py::from_object<T>(ob), py::invalid_conversion)
        << py::util::type_name<T>().get();
    EXPECT_TRUE(PyErr_Occurred());
    PyErr_Clear();
}

TEST_F(from_object, test_overflow) {
    auto py_one = py::scoped_ref(PyLong_FromLong(1));
    ASSERT_TRUE(py_one) << "py_one cannot be null";
    auto py_neg_one = py::scoped_ref(PyLong_FromLong(-1));
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

    auto check_limit = [&](auto limit_value, py::scoped_ref<PyObject>& to_exceed) {
        using T = decltype(limit_value);

        py::scoped_ref<PyObject> limit_ob;

        if constexpr (std::is_signed_v<T>) {
            limit_ob = py::scoped_ref(PyLong_FromLongLong(limit_value));
        }
        else {
            limit_ob = py::scoped_ref(PyLong_FromUnsignedLongLong(limit_value));
        }
        ASSERT_TRUE(limit_ob) << "limit_ob cannot be null";

        // the limit value should be convertible
        check_integer<T>(limit_ob, limit_value);

        auto exceeds_limit = py::scoped_ref(
            PyNumber_Add(limit_ob.get(), to_exceed.get()));
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

TEST_F(from_object, ndarray_view) {
    py::ensure_import_array _;

    {
        auto ndarray = py::move_to_numpy_array(std::vector<std::int32_t>{0, 1, 2, 3});
        auto view = py::from_object<py::array_view<std::int32_t>>(ndarray);

        std::int32_t expected = 0;
        for (auto v : view) {
            EXPECT_EQ(v, expected);
            ++expected;
        }

        EXPECT_THROW(py::from_object<py::array_view<std::int8_t>>(ndarray),
                     py::exception);
        EXPECT_THROW(py::from_object<py::array_view<std::int16_t>>(ndarray),
                     py::exception);
        EXPECT_THROW(py::from_object<py::array_view<std::int64_t>>(ndarray),
                     py::exception);

        PyErr_Clear();
    }

    {
        auto ndarray = py::move_to_numpy_array(std::vector<std::int64_t>{0, 1, 2, 3});
        auto view = py::from_object<py::array_view<std::int64_t>>(ndarray);

        std::int64_t expected = 0;
        for (auto v : view) {
            EXPECT_EQ(v, expected);
            ++expected;
        }

        EXPECT_THROW(py::from_object<py::array_view<std::int8_t>>(ndarray),
                     py::exception);
        EXPECT_THROW(py::from_object<py::array_view<std::int16_t>>(ndarray),
                     py::exception);
        EXPECT_THROW(py::from_object<py::array_view<std::int32_t>>(ndarray),
                     py::exception);

        PyErr_Clear();
    }
}

TEST_F(from_object, ndarray_view_any_ref) {
    py::ensure_import_array _;

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

        std::vector<py::scoped_ref<PyObject>> objects_cp;
        auto ndarray = py::move_to_numpy_array(std::move(objects_cp));
        auto view = py::from_object<py::array_view<py::any_ref>>(ndarray);

        std::size_t ix = 0;
        for (auto v : view) {
            // compare the underlying PyObject* which compares the objects on identity
            EXPECT_EQ(v.cast<py::scoped_ref<PyObject>>().get(), objects[ix].get());
            ++ix;
        }
    }
}
}  // namespace test_from_object
