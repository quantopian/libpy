#include <vector>

#include "gtest/gtest.h"

#include "libpy/util.h"

namespace test_util {
TEST(searchsorted_left, all) {
    std::vector<std::int64_t> vec{1, 2, 3, 5, 6};

    // needle not in container.
    EXPECT_EQ(py::util::searchsorted_l(vec, 4), 3);

    // needle collides with value in container.
    EXPECT_EQ(py::util::searchsorted_l(vec, 3), 2);

    // needle greater than largest value in container.
    EXPECT_EQ(py::util::searchsorted_l(vec, 42), vec.size());

    // needle less than smallest value in container.
    EXPECT_EQ(py::util::searchsorted_l(vec, -1), 0);

    // needle equal to largest value in container.
    EXPECT_EQ(py::util::searchsorted_l(vec, 6), 4);

    // needle equal to smallest value in container.
    EXPECT_EQ(py::util::searchsorted_l(vec, 1), 0);
}

TEST(searchsorted_right, all) {
    std::vector<std::int64_t> vec{1, 2, 3, 5, 6};

    // needle not in container.
    EXPECT_EQ(py::util::searchsorted_r(vec, 4), 3);

    // needle collides with value in container.
    EXPECT_EQ(py::util::searchsorted_r(vec, 3), 3);

    // needle greater than largest value in container.
    EXPECT_EQ(py::util::searchsorted_r(vec, 42), vec.size());

    // needle less than smallest value in container.
    EXPECT_EQ(py::util::searchsorted_r(vec, -1), 0);

    // needle equal to largest value in container.
    EXPECT_EQ(py::util::searchsorted_r(vec, 6), vec.size());

    // needle equal to smallest value in container.
    EXPECT_EQ(py::util::searchsorted_r(vec, 1), 1);
}
}  // namespace test_util
