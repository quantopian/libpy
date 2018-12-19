#include "gtest/gtest.h"

#include "libpy/meta.h"

namespace test_meta {
struct my_type {};

TEST(element_of, true_cases) {
    bool case0 = py::meta::element_of<float, std::tuple<float, bool, int>>;
    EXPECT_EQ(case0, true);

    bool case1 = py::meta::element_of<bool, std::tuple<float, bool, int>>;
    EXPECT_EQ(case1, true);

    bool case2 = py::meta::element_of<bool, std::tuple<float, bool, int>>;
    EXPECT_EQ(case2, true);

    bool case3 = py::meta::element_of<std::tuple<>, std::tuple<std::tuple<>>>;
    EXPECT_EQ(case3, true);

    bool case4 = py::meta::element_of<my_type, std::tuple<my_type, int, float>>;
    EXPECT_EQ(case4, true);

    bool case5 = py::meta::element_of<my_type, std::tuple<my_type, my_type, my_type>>;
    EXPECT_EQ(case5, true);
}

TEST(element_of, false_cases) {
    bool case0 = py::meta::element_of<int, std::tuple<bool, float, my_type>>;
    EXPECT_EQ(case0, false);

    bool case1 = py::meta::element_of<int, std::tuple<>>;
    EXPECT_EQ(case1, false);

    bool case2 = py::meta::element_of<std::tuple<int>, std::tuple<int>>;
    EXPECT_EQ(case2, false);
}
}  // namespace test_meta
