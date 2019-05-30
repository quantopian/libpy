#include <limits>
#include <stdexcept>

#include "gtest/gtest.h"

#include "libpy/datetime64.h"
#include "libpy/dense_hash_map.h"

namespace test_dense_hash_map {
TEST(dense_hash_map, invalid_empty_key) {
    using double_key = py::dense_hash_map<double, int>;
    EXPECT_THROW((double_key{std::numeric_limits<double>::quiet_NaN()}),
                 std::invalid_argument);
    EXPECT_THROW((double_key{std::numeric_limits<double>::quiet_NaN(), 10}),
                 std::invalid_argument);

    using float_key = py::dense_hash_map<float, int>;
    EXPECT_THROW((float_key{std::numeric_limits<float>::quiet_NaN()}),
                 std::invalid_argument);
    EXPECT_THROW((float_key{std::numeric_limits<float>::quiet_NaN(), 10}),
                 std::invalid_argument);

    using M8_key = py::dense_hash_map<py::datetime64ns, int>;
    EXPECT_THROW((M8_key{py::datetime64ns::nat()}), std::invalid_argument);
    EXPECT_THROW((M8_key{py::datetime64ns::nat(), 10}), std::invalid_argument);
}

TEST(dense_hash_set, invalid_empty_key) {
    using double_key = py::dense_hash_set<double>;
    EXPECT_THROW((double_key{std::numeric_limits<double>::quiet_NaN()}),
                 std::invalid_argument);
    EXPECT_THROW((double_key{std::numeric_limits<double>::quiet_NaN(), 10}),
                 std::invalid_argument);

    using float_key = py::dense_hash_set<float>;
    EXPECT_THROW((float_key{std::numeric_limits<float>::quiet_NaN()}),
                 std::invalid_argument);
    EXPECT_THROW((float_key{std::numeric_limits<float>::quiet_NaN(), 10}),
                 std::invalid_argument);

    using M8_key = py::dense_hash_set<py::datetime64ns>;
    EXPECT_THROW((M8_key{py::datetime64ns::nat()}), std::invalid_argument);
    EXPECT_THROW((M8_key{py::datetime64ns::nat(), 10}), std::invalid_argument);
}
}  // namespace test_dense_hash_map
