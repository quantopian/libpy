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
    std::mt19937 g(1868655980);

    for (std::size_t n = 0; n < 1000; ++n) {
        std::string s = random_string(g);
        EXPECT_EQ(std::hash<std::string>{}(s), py::hash_buffer(s.data(), s.size()));
    }
}
}  // namespace test_hash
