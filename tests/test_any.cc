#include <sstream>
#include <string>

#include "gtest/gtest.h"

#include "libpy/any.h"

namespace test_any {
TEST(any_vtable, void_vtable) {
    py::any_vtable vtable = py::any_vtable::make<void>();

    void* a = nullptr;

    EXPECT_EQ(vtable.type_info(), typeid(void));
    EXPECT_EQ(vtable.size(), 0ul);
    EXPECT_EQ(vtable.align(), 0ul);
    EXPECT_FALSE(vtable.is_trivially_default_constructible());
    EXPECT_FALSE(vtable.is_trivially_destructible());
    EXPECT_FALSE(vtable.is_trivially_move_constructible());
    EXPECT_FALSE(vtable.is_trivially_copy_constructible());
    EXPECT_FALSE(vtable.is_trivially_copyable());
    EXPECT_FALSE(vtable.move_is_noexcept());

    EXPECT_THROW(vtable.copy_assign(a, a), std::runtime_error);
    EXPECT_THROW(vtable.move_assign(a, a), std::runtime_error);
    EXPECT_THROW(vtable.default_construct(a), std::runtime_error);
    EXPECT_THROW(vtable.copy_construct(a, a), std::runtime_error);
    EXPECT_THROW(vtable.move_construct(a, a), std::runtime_error);
    EXPECT_THROW(vtable.move_if_noexcept(a, a), std::runtime_error);
    EXPECT_THROW(vtable.ne(a, a), std::runtime_error);
    EXPECT_THROW(vtable.eq(a, a), std::runtime_error);
    EXPECT_THROW(vtable.to_object(a), std::runtime_error);

    EXPECT_EQ(vtable.type_name(), py::util::type_name<void>());

    EXPECT_EQ(vtable, py::any_vtable::make<void>());
    EXPECT_EQ(vtable, py::any_vtable{});  // default construct makes void table
    EXPECT_NE(vtable, py::any_vtable::make<int>());
    EXPECT_NE(vtable, py::any_vtable::make<float>());
}

TEST(any_vtable, ostream_format) {
    using namespace std::string_literals;
    {
        py::any_vtable vtable = py::any_vtable::make<int>();
        std::stringstream stream;
        int value = 99;
        vtable.ostream_format(stream, &value);
        value = -50;
        vtable.ostream_format(stream, &value);

        EXPECT_EQ(stream.str(), "99-50"s);
    }
    {
        py::any_vtable vtable = py::any_vtable::make<float>();
        std::stringstream stream;
        float value = 1.5;
        vtable.ostream_format(stream, &value);
        value = -3.5;
        vtable.ostream_format(stream, &value);

        EXPECT_EQ(stream.str(), "1.5-3.5"s);
    }
    {
        // a new type which doesn't have an overload for:
        // `operator<<(std::ostream&, const S&)`
        struct S {
            int a;

            bool operator==(const S& other) const {
                return a == other.a;
            }

            bool operator!=(const S& other) const {
                return !(*this == other);
            }
        };

        py::any_vtable vtable = py::any_vtable::make<S>();
        std::stringstream stream;
        S value{0};

        // this should throw a runtime error because this operation isn't defined
        EXPECT_THROW(vtable.ostream_format(stream, &value), py::exception);
        PyErr_Clear();
    }
}

TEST(any_vtable, map_key) {
    std::unordered_map<py::any_vtable, int> map;

    map[py::any_vtable::make<int>()] = 0;
    map[py::any_vtable::make<float>()] = 1;
    map[py::any_vtable::make<std::string>()] = 2;

    EXPECT_EQ(map[py::any_vtable::make<int>()], 0);
    EXPECT_EQ(map[py::any_vtable::make<float>()], 1);
    EXPECT_EQ(map[py::any_vtable::make<std::string>()], 2);
}

TEST(any_ref, test_construction) {
    int underlying = 1;
    py::any_ref ref = py::make_any_ref(underlying);

    EXPECT_EQ(ref.cast<int>(), underlying);
}

TEST(any_ref, test_assign) {
    int underlying = 1;
    py::any_ref ref = py::make_any_ref(underlying);

    ref = 2;

    EXPECT_EQ(ref.cast<int>(), 2);
    EXPECT_EQ(underlying, 2);

    int another_object = 3;
    py::any_ref another_ref = py::make_any_ref(another_object);

    ref = another_ref;

    EXPECT_EQ(ref.cast<int>(), another_object);
    EXPECT_EQ(underlying, another_object);

    ref.cast<int>() = 4;
    EXPECT_EQ(ref.cast<int>(), 4);
    EXPECT_EQ(underlying, 4);

    // `ref` from another `py::any_ref` should have no affect on the `rhs`.
    EXPECT_EQ(another_object, 3);

    another_object = 5;
    py::any_cref another_cref = py::make_any_cref(another_object);

    ref = another_cref;
    EXPECT_EQ(ref.cast<int>(), another_object);
    EXPECT_EQ(underlying, another_object);
}

TEST(any_ref, test_assign_type_check) {
    int underlying = 1;
    py::any_ref ref = py::make_any_ref(underlying);

    EXPECT_THROW(ref = 2.5, std::bad_any_cast);

    EXPECT_EQ(ref.cast<int>(), 1);
    EXPECT_EQ(underlying, 1);

    float another_object = 3.5;
    py::any_ref another_ref = py::make_any_ref(another_object);

    EXPECT_THROW(ref = another_ref, std::bad_any_cast);

    EXPECT_EQ(ref.cast<int>(), 1);
    EXPECT_EQ(underlying, 1);

    another_object = 4.5;
    py::any_cref another_cref = py::make_any_cref(another_object);

    EXPECT_THROW(ref = another_cref, std::bad_any_cast);
    EXPECT_EQ(ref.cast<int>(), 1);
    EXPECT_EQ(underlying, 1);
}

TEST(any_ref, test_cast) {
    int underlying = 1;
    py::any_ref ref = py::make_any_ref(underlying);

    // simple struct that has the same storage as an int, but a different type
    struct S {
        int a;

        bool operator==(const S& other) const {
            return a == other.a;
        }

        bool operator!=(const S& other) const {
            return !(*this == other);
        }
    };

    EXPECT_THROW(ref.cast<float>(), std::bad_any_cast);
    EXPECT_THROW(ref.cast<long>(), std::bad_any_cast);
    EXPECT_THROW(ref.cast<S>(), std::bad_any_cast);

    EXPECT_EQ(ref.cast<int>(), underlying);
    testing::StaticAssertTypeEq<decltype(ref.cast<int>()), int&>();

    [](const py::any_ref& const_reference_to_ref) {
        testing::StaticAssertTypeEq<decltype(const_reference_to_ref.cast<int>()),
                                    const int&>();
    }(ref);
}

TEST(any_cref, test_construction) {
    int underlying = 1;
    py::any_cref ref = py::make_any_cref(underlying);

    EXPECT_EQ(ref.cast<int>(), underlying);
}

TEST(any_cref, test_assign) {
    constexpr bool can_assign_from_any_cref_rvalue =
        std::is_assignable_v<py::any_cref, py::any_cref&&>;
    EXPECT_TRUE(can_assign_from_any_cref_rvalue);

    constexpr bool can_assign_from_int = std::is_assignable_v<py::any_cref, int>;
    EXPECT_FALSE(can_assign_from_int);

    constexpr bool can_assign_from_float = std::is_assignable_v<py::any_cref, float>;
    EXPECT_FALSE(can_assign_from_float);

    constexpr bool can_assign_from_any_ref =
        std::is_assignable_v<py::any_cref, py::any_ref>;
    EXPECT_FALSE(can_assign_from_any_ref);

    constexpr bool can_assign_from_any_cref =
        std::is_assignable_v<py::any_cref, const py::any_cref&>;
    EXPECT_FALSE(can_assign_from_any_cref);
}

TEST(any_cref, test_cast) {
    int underlying = 1;
    py::any_cref ref = py::make_any_cref(underlying);

    // simple struct that has the same storage as an int, but a different type
    struct S {
        int a;

        bool operator==(const S& other) const {
            return a == other.a;
        }

        bool operator!=(const S& other) const {
            return !(*this == other);
        }
    };

    EXPECT_THROW(ref.cast<float>(), std::bad_any_cast);
    EXPECT_THROW(ref.cast<long>(), std::bad_any_cast);
    EXPECT_THROW(ref.cast<S>(), std::bad_any_cast);

    EXPECT_EQ(ref.cast<int>(), underlying);

    testing::StaticAssertTypeEq<decltype(ref.cast<int>()), const int&>();

    [](const py::any_cref& const_reference_to_ref) {
        testing::StaticAssertTypeEq<decltype(const_reference_to_ref.cast<int>()),
                                    const int&>();
    }(ref);
}
}  // namespace test_any
