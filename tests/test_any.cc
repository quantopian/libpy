#include "gtest/gtest.h"

#include "libpy/any.h"
#include "libpy/dense_hash_map.h"

namespace test_any {
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

TEST(any_vtable, map_key) {
    py::dense_hash_map<py::any_vtable, int> map(py::any_vtable::make<void*>());

    map[py::any_vtable::make<int>()] = 0;
    map[py::any_vtable::make<float>()] = 1;
    map[py::any_vtable::make<std::string>()] = 2;

    EXPECT_EQ(map[py::any_vtable::make<int>()], 0);
    EXPECT_EQ(map[py::any_vtable::make<float>()], 1);
    EXPECT_EQ(map[py::any_vtable::make<std::string>()], 2);
}
}  // namespace test_any
