#include "gtest/gtest.h"

#include "libpy/demangle.h"

class test_type_name_type {
public:
    class inner_type {};
};

namespace test_demangle_namespace {
class test_type_name_type {};
}  // namespace test_demangle_namespace

TEST(demangle, type_name) {
    {
        class abc {};
        auto name = py::util::type_name<test_type_name_type>();
        EXPECT_STREQ(name.get(), "test_type_name_type");
    }

    {
        auto name = py::util::type_name<test_demangle_namespace::test_type_name_type>();
        EXPECT_STREQ(name.get(), "test_demangle_namespace::test_type_name_type");
    }

    {
        class test_type_name_type {};
        auto name = py::util::type_name<test_type_name_type>();
        EXPECT_STREQ(name.get(),
                     "demangle_type_name_Test::TestBody()::test_type_name_type");
    }

    {
        auto name = py::util::type_name<test_type_name_type::inner_type>();
        EXPECT_STREQ(name.get(), "test_type_name_type::inner_type");
    }
}
