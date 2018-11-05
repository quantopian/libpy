#include <algorithm>
#include <vector>

#include "gtest/gtest.h"

#include "libpy/itertools.h"

TEST(zip, mismatched_sizes) {
    std::vector<int> as(10);
    std::vector<int> bs(5);

    EXPECT_THROW(py::zip(as, bs), std::invalid_argument);
}


TEST(zip, const_iterator) {
    std::size_t size = 10;

    int counter = 0;
    std::vector<int> as(size);
    std::vector<int> bs(size);
    std::vector<int> cs(size);

    auto gen = [&counter]() { return counter++; };
    std::generate(as.begin(), as.end(), gen);
    std::generate(bs.begin(), bs.end(), gen);
    std::generate(cs.begin(), cs.end(), gen);

    std::size_t ix = 0;
    for (const auto [a, b, c] : py::zip(as, bs, cs)) {
        EXPECT_EQ(a, as[ix]);
        EXPECT_EQ(b, bs[ix]);
        EXPECT_EQ(c, cs[ix]);

        ++ix;
    }

    EXPECT_EQ(ix, size);
}


TEST(zip, mutable_iterator) {
    std::size_t size = 10;

    int counter = 0;
    std::vector<int> as(size);
    std::vector<int> bs(size);
    std::vector<int> cs(size);

    auto gen = [&counter]() { return counter++; };
    std::generate(as.begin(), as.end(), gen);
    std::generate(bs.begin(), bs.end(), gen);
    std::generate(cs.begin(), cs.end(), gen);

    std::vector<int> as_original = as;
    std::vector<int> bs_original = bs;
    std::vector<int> cs_original = cs;

    std::size_t ix = 0;
    for (auto [a, b, c] : py::zip(as, bs, cs)) {
        EXPECT_EQ(a, as[ix]);
        EXPECT_EQ(b, bs[ix]);
        EXPECT_EQ(c, cs[ix]);

        a = -a;
        b = -b;
        c = -c;

        ++ix;
    }

    EXPECT_EQ(ix, size);

    for (std::size_t ix = 0; ix < size; ++ix) {
        EXPECT_EQ(as[ix], -as_original[ix]);
        EXPECT_EQ(bs[ix], -bs_original[ix]);
        EXPECT_EQ(cs[ix], -cs_original[ix]);
    }
}
