#include "gtest/gtest.h"

#include "libpy/scope_guard.h"

namespace test_scope_guard {
TEST(scope_guard, scope_exit) {
    bool fired = false;
    {
        py::util::scope_guard guard([&] { fired = true; });
        EXPECT_FALSE(fired);
    }

    EXPECT_TRUE(fired);
}

TEST(scope_guard, throw_) {
    bool fired = false;
    auto f = [&] {
        py::util::scope_guard guard([&] { fired = true; });
        EXPECT_FALSE(fired);
        throw std::runtime_error("boom");
    };
    EXPECT_FALSE(fired);
    EXPECT_THROW(f(), std::runtime_error);
    EXPECT_TRUE(fired);
}

TEST(scope_guard, dismiss) {
    bool fired = false;
    {
        py::util::scope_guard guard([&] { fired = true; });
        EXPECT_FALSE(fired);
        guard.dismiss();
    };
    EXPECT_FALSE(fired);
}
}  // namespace test_scope_guard
