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

template<typename T>
T max = std::numeric_limits<T>::max();

template<typename T>
T min = std::numeric_limits<T>::min();

TEST_F(from_object, test_overflow) {
    auto py_one = py::scoped_ref(PyLong_FromLong(1));
    ASSERT_TRUE(py_one) << "py_one cannot be null";
    auto py_neg_one = py::scoped_ref(PyLong_FromLong(-1));
    ASSERT_TRUE(py_neg_one) << "py_neg_one cannot be null";

    // The limit values (max and min) paired with an integer which, when added to the
    // limit value, would result in an object that overflows in `from_object`.
    auto limit_values = std::make_tuple(std::make_tuple(max<std::int64_t>, py_one),
                                        std::make_tuple(max<std::int32_t>, py_one),
                                        std::make_tuple(max<std::int16_t>, py_one),
                                        std::make_tuple(max<std::int8_t>, py_one),
                                        std::make_tuple(min<std::int64_t>, py_neg_one),
                                        std::make_tuple(min<std::int32_t>, py_neg_one),
                                        std::make_tuple(min<std::int16_t>, py_neg_one),
                                        std::make_tuple(min<std::int8_t>, py_neg_one),
                                        std::make_tuple(max<std::uint64_t>, py_one),
                                        std::make_tuple(max<std::uint32_t>, py_one),
                                        std::make_tuple(max<std::uint16_t>, py_one),
                                        std::make_tuple(max<std::uint8_t>, py_one),
                                        std::make_tuple(min<std::uint64_t>, py_neg_one),
                                        std::make_tuple(min<std::uint32_t>, py_neg_one),
                                        std::make_tuple(min<std::uint16_t>, py_neg_one),
                                        std::make_tuple(min<std::uint8_t>, py_neg_one));

    auto check_limit = [&](auto item) {
        auto [limit_value, to_exceed] = item;
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

    std::apply([&](auto... items) { (check_limit(items), ...); }, limit_values);
}
}  // namespace test_from_object
