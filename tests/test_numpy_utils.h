#include "gtest/gtest.h"

#include "libpy/numpy_utils.h"

namespace test_numpy_utils {
TEST(py_bool, hash) {
    auto check_hash = [](bool v) {
        py::py_bool u{v};
        EXPECT_EQ(std::hash<py::py_bool>{}(u), std::hash<bool>{}(v));
    };

    check_hash(false);
    check_hash(true);
}
}  // namespace test_numpy_utils
