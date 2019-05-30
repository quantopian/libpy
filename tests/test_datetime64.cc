#include <sstream>
#include <string>
#include <type_traits>

#include "Python.h"
#include "gtest/gtest.h"

#include "libpy/call_function.h"
#include "libpy/char_sequence.h"
#include "libpy/datetime64.h"
#include "libpy/exception.h"
#include "libpy/scoped_ref.h"
#include "test_utils.h"

namespace test_datetime64 {
using namespace std::string_literals;
using namespace py::cs::literals;

template<typename T>
class datetime64_all_units : public with_python_interpreter {};
TYPED_TEST_CASE_P(datetime64_all_units);

using int64_limits = std::numeric_limits<std::int64_t>;

TYPED_TEST_P(datetime64_all_units, from_int) {
    constexpr std::int64_t step_size = int64_limits::max() / (1 << 16);
    for (int step = 0; step < (1 << 16); ++step) {
        auto ticks = int64_limits::min() + step * step_size;
        py::datetime64<TypeParam> value(ticks);
        ASSERT_EQ(static_cast<std::int64_t>(value), ticks);
    }
}

TYPED_TEST_P(datetime64_all_units, epoch) {
    EXPECT_EQ(static_cast<std::int64_t>(py::datetime64<TypeParam>::epoch()), 0LL);
}

TYPED_TEST_P(datetime64_all_units, max) {
    EXPECT_EQ(static_cast<std::int64_t>(py::datetime64<TypeParam>::max()),
              int64_limits::max());
}

TYPED_TEST_P(datetime64_all_units, nat) {
    EXPECT_EQ(static_cast<std::int64_t>(py::datetime64<TypeParam>::nat()),
              int64_limits::min());

    // default construct is also nat
    EXPECT_EQ(static_cast<std::int64_t>(py::datetime64<TypeParam>{}),
              int64_limits::min());

    // unambiguously test `operator==()` and `operator!=()`:
    EXPECT_FALSE(py::datetime64<TypeParam>{} == py::datetime64<TypeParam>{});
    EXPECT_TRUE(py::datetime64<TypeParam>{} != py::datetime64<TypeParam>{});
}

TYPED_TEST_P(datetime64_all_units, min) {
    if (std::is_same_v<TypeParam, py::chrono::ns>) {
        // we use the same value as pandas.Timestamp.min for datetime64
        EXPECT_EQ(static_cast<std::int64_t>(py::datetime64<TypeParam>::min()),
                  -9223285636854775000LL);
    }
    else {
        EXPECT_EQ(static_cast<std::int64_t>(py::datetime64<TypeParam>::min()),
                  // min is reserved for nat
                  int64_limits::min() + 1);
    }
}

TYPED_TEST_P(datetime64_all_units, stream_format_nat) {
    std::stringstream stream;
    stream << py::datetime64<TypeParam>::nat();
    EXPECT_EQ(stream.str(), "NaT"s);
}

template<typename T>
constexpr void* numpy_unit_str;

template<>
constexpr auto numpy_unit_str<py::chrono::ns> = "ns"_arr;

template<>
constexpr auto numpy_unit_str<py::chrono::us> = "us"_arr;

template<>
constexpr auto numpy_unit_str<py::chrono::ms> = "ms"_arr;

template<>
constexpr auto numpy_unit_str<py::chrono::s> = "s"_arr;

template<>
constexpr auto numpy_unit_str<py::chrono::m> = "m"_arr;

template<>
constexpr auto numpy_unit_str<py::chrono::h> = "h"_arr;

template<>
constexpr auto numpy_unit_str<py::chrono::D> = "D"_arr;

TYPED_TEST_P(datetime64_all_units, stream_format) {
    auto numpy_mod = py::scoped_ref(PyImport_ImportModule("numpy"));
    if (!numpy_mod) {
        throw py::exception{};
    }
    auto numpy_datetime64 = py::scoped_ref(
        PyObject_GetAttrString(numpy_mod.get(), "datetime64"));
    if (!numpy_datetime64) {
        throw py::exception{};
    }

    using dt = py::datetime64<TypeParam>;
    constexpr std::size_t step_size = 18446657673709550807ULL / (1 << 16);
    constexpr auto min_ticks = static_cast<std::int64_t>(dt::min());
    // numpy overflows the repr code somewhere if we use really massively negative values
    for (int step = 1; step < (1 << 16); ++step) {
        std::int64_t ticks = min_ticks + step * step_size;
        dt value(ticks);

        std::stringstream stream;
        stream << value;
        auto res = py::call_function(numpy_datetime64,
                                     static_cast<std::int64_t>(value),
                                     numpy_unit_str<TypeParam>);
        if (!res) {
            throw py::exception{};
        }
        auto repr = py::scoped_ref(PyObject_Str(res.get()));
        if (!repr) {
            throw py::exception{};
        }
        auto repr_text = py::util::pystring_to_cstring(repr.get());
        if (!repr_text) {
            throw py::exception{};
        }

        ASSERT_STREQ(stream.str().c_str(), repr_text) << "ticks=" << ticks;
    }
}

REGISTER_TYPED_TEST_CASE_P(datetime64_all_units,
                           from_int,
                           epoch,
                           max,
                           nat,
                           min,
                           stream_format_nat,
                           stream_format);

using units = testing::Types<py::chrono::ns,
                             py::chrono::us,
                             py::chrono::ms,
                             py::chrono::s,
                             py::chrono::m,
                             py::chrono::h,
                             py::chrono::D>;
INSTANTIATE_TYPED_TEST_CASE_P(typed_, datetime64_all_units, units);
}  // namespace test_datetime64
