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

using t = std::tuple<std::int64_t, std::int64_t, std::int64_t>;

class apply_to_groups_test_cases : public testing::Test {
protected:
    // vector containing tuples of input sequences and expected outputs.
    std::vector<std::tuple<std::vector<std::int64_t>, std::vector<t>>> test_cases;

    void SetUp() override {

        // case 1: empty sequence
        test_cases.emplace_back(std::vector<std::int64_t>(), std::vector<t>());

        // case 2: sequence of length 1
        test_cases.emplace_back(std::vector<std::int64_t>({1}),
                                std::vector<t>({std::make_tuple(1, 0, 1)}));

        // case 3: sequence containing single unique value
        test_cases.emplace_back(std::vector<std::int64_t>({1, 1}),
                                std::vector<t>({std::make_tuple(1, 0, 2)}));

        // case 4: sequence with multiple unique values 1
        test_cases.emplace_back(std::vector<std::int64_t>({1, 1, 2}),
                                std::vector<t>({std::make_tuple(1, 0, 2),
                                                std::make_tuple(2, 2, 3)}));

        // case 5: sequence with multiple unique values 2
        test_cases.emplace_back(std::vector<std::int64_t>({1, 1, 2, 2}),
                                std::vector<t>({std::make_tuple(1, 0, 2),
                                                std::make_tuple(2, 2, 4)}));

        // case 6: sequence with multiple unique values 3
        test_cases.emplace_back(std::vector<std::int64_t>({1, 1, 2, 2, 2, 3, 4, 4}),
                                std::vector<t>({std::make_tuple(1, 0, 2),
                                                std::make_tuple(2, 2, 5),
                                                std::make_tuple(3, 5, 6),
                                                std::make_tuple(4, 6, 8)}));

        // case 7: non-sorted sequence with multiple unique values
        test_cases.emplace_back(std::vector<std::int64_t>({1, 1, 3, 4, 4, 7, 2, 2, 1}),
                                std::vector<t>({std::make_tuple(1, 0, 2),
                                                std::make_tuple(3, 2, 3),
                                                std::make_tuple(4, 3, 5),
                                                std::make_tuple(7, 5, 6),
                                                std::make_tuple(2, 6, 8),
                                                std::make_tuple(1, 8, 9)}));
    }
};

TEST_F(apply_to_groups_test_cases, using_it_begin_end) {
    std::vector<t> test_vec;

    auto f = [&](const auto& val, std::ptrdiff_t start_ix, std::ptrdiff_t stop_ix) {
        test_vec.emplace_back(val, start_ix, stop_ix);
    };

    for (auto [input, expected_out] : test_cases) {
        py::util::apply_to_groups(input.begin(), input.end(), f);
        EXPECT_EQ(test_vec, expected_out);
        test_vec.clear();
    }
}

TEST_F(apply_to_groups_test_cases, using_it_range) {
    std::vector<t> test_vec;

    auto f = [&](const auto& val, std::ptrdiff_t start_ix, std::ptrdiff_t stop_ix) {
        test_vec.emplace_back(val, start_ix, stop_ix);
    };

    for (auto [input, expected_out] : test_cases) {
        py::util::apply_to_groups(input, f);
        EXPECT_EQ(test_vec, expected_out);
        test_vec.clear();
    }
}
}  // namespace test_util
