#include "gtest/gtest.h"

#include "libpy/demangle.h"

struct test_demangle_global_type {};

namespace test_demangle {
class test_type_name_type {
public:
    class inner_type {};
};

namespace inner {
class test_type_name_type {};
}  // namespace inner

TEST(demangle, type_name) {
    {
        auto name = py::util::type_name<test_demangle_global_type>();
        EXPECT_STREQ(name.get(), "test_demangle_global_type");
    }
    {
        auto name = py::util::type_name<test_type_name_type>();
        EXPECT_STREQ(name.get(), "test_demangle::test_type_name_type");
    }

    {
        auto name = py::util::type_name<inner::test_type_name_type>();
        EXPECT_STREQ(name.get(), "test_demangle::inner::test_type_name_type");
    }

    {
        class test_type_name_type {};
        auto name = py::util::type_name<test_type_name_type>();
        EXPECT_STREQ(
            name.get(),
            "test_demangle::demangle_type_name_Test::TestBody()::test_type_name_type");
    }

    {
        auto name = py::util::type_name<test_type_name_type::inner_type>();
        EXPECT_STREQ(name.get(), "test_demangle::test_type_name_type::inner_type");
    }
}
}  // namespace test_demangle
