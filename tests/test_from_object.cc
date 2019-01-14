#include <type_traits>

#include "libpy/demangle.h"
#include "libpy/from_object.h"
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
}  // namespace test_from_object
