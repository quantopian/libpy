#include <random>
#include <string>

#include "gtest/gtest.h"

#include "libpy/hash.h"

namespace test_hash {
template<typename RandomEngine>
std::string random_string(RandomEngine& g) {
    std::uniform_int_distribution<unsigned char> d(0);
    unsigned char length = d(g);
    std::string out;
    for (unsigned char ix = 0; ix < length; ++ix) {
        out.push_back(d(g));
    }
    return out;
}

TEST(hash, buffer) {
#if !defined(_LIBCPP_VERSION)
    // XXX: This asserts that `py::hash_buffer` does the same thing as
    // `std::hash<std::string>`; however, it is a copy of the libstdc++
    // algorithm. libc++ uses a different hash algorithm which doesn't produce
    // the same results.
    std::mt19937 g(1868655980);

    for (std::size_t n = 0; n < 1000; ++n) {
        std::string s = random_string(g);
        EXPECT_EQ(std::hash<std::string>{}(s), py::hash_buffer(s.data(), s.size()));
    }
#endif
}
}  // namespace test_hash
