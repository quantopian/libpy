#include <cstddef>
#include <unordered_set>

#include "gtest/gtest.h"

#include <libpy/any.h>
#include <libpy/any_vector.h>

namespace test_any_vector {
TEST(any_vector, construct_from_vtable) {
    auto f = [](const py::any_vtable& vtable) {
        py::any_vector vec(vtable);

        ASSERT_EQ(vec.size(), 0ul);
        ASSERT_EQ(vec.capacity(), 0ul);
        ASSERT_EQ(vec.vtable(), vtable);
    };

    struct S {
        int data;

        bool operator==(S other) const {
            return data == other.data;
        }

        bool operator!=(S other) const {
            return data != other.data;
        }
    };

    f(py::any_vtable::make<int>());
    f(py::any_vtable::make<float>());
    f(py::any_vtable::make<S>());
}

TEST(any_vector, trivially_default_construct_elements) {
    auto f = [](const py::any_vtable& vtable, std::size_t count, auto expected_value) {
        py::any_vector vec(vtable, count);

        ASSERT_EQ(vec.size(), count);
        ASSERT_GE(vec.capacity(), count);
        ASSERT_EQ(vec.vtable(), vtable);

        for (auto v : vec) {
            EXPECT_EQ(v, expected_value);
        }
    };

    struct S {
        int data;

        bool operator==(S other) const {
            return data == other.data;
        }

        bool operator!=(S other) const {
            return data != other.data;
        }
    };

    // S has no explicit constructors and consists entirely of trivially default
    // constructible elements so it is also trivially default constructible
    ASSERT_TRUE(py::any_vtable::make<S>().is_trivially_default_constructible());

    for (std::size_t count = 0; count < 256; count += 8) {
        f(py::any_vtable::make<int>(), count, 0);
        f(py::any_vtable::make<float>(), count, 0.0f);
        f(py::any_vtable::make<S>(), count, S{0});
    }
}

TEST(any_vector, default_construct_elements) {
    auto f =
        [](const py::any_vtable& vtable, std::size_t count, const auto& expected_value) {
            py::any_vector vec(vtable, count);

            ASSERT_EQ(vec.size(), count);
            ASSERT_GE(vec.capacity(), count);
            ASSERT_EQ(vec.vtable(), vtable);

            for (std::size_t ix = 0; ix < vec.size(); ++ix) {
                EXPECT_EQ(vec[ix], expected_value);
            }
        };

    struct S {
        int data = 0;

        bool operator==(S other) const {
            return data == other.data;
        }

        bool operator!=(S other) const {
            return data != other.data;
        }
    };

    for (std::size_t count = 0; count < 256; count += 8) {
        f(py::any_vtable::make<S>(), count, S{0});
    }
}

TEST(any_vector, copy_construct_elements) {
    auto f =
        [](const py::any_vtable& vtable, std::size_t count, const auto& expected_value) {
            py::any_vector vec(vtable, count, expected_value);

            ASSERT_EQ(vec.size(), count);
            ASSERT_GE(vec.capacity(), count);
            ASSERT_EQ(vec.vtable(), vtable);

            for (std::size_t ix = 0; ix < vec.size(); ++ix) {
                EXPECT_EQ(vec[ix], expected_value);
            }
        };

    struct S {
        int data = 0;

        bool operator==(S other) const {
            return data == other.data;
        }

        bool operator!=(S other) const {
            return data != other.data;
        }
    };

    for (std::size_t count = 0; count < 256; count += 8) {
        f(py::any_vtable::make<int>(), count, 0);
        f(py::any_vtable::make<int>(), count, 1);
        f(py::any_vtable::make<int>(), count, 3);

        f(py::any_vtable::make<float>(), count, 0.0f);
        f(py::any_vtable::make<float>(), count, 1.0f);
        f(py::any_vtable::make<float>(), count, 3.0f);

        f(py::any_vtable::make<S>(), count, S{0});
        f(py::any_vtable::make<S>(), count, S{1});
        f(py::any_vtable::make<S>(), count, S{3});
    }
}

TEST(any_vector, copy_constructor_not_trivially_copyable) {
    struct S {
        std::unordered_set<int>* copy_constructions = nullptr;
        int data = 0;

        S() = default;

        S(std::unordered_set<int>& copy_constructions, int data)
            : copy_constructions(&copy_constructions), data(data) {}

        S(const S& cpfrom)
            : copy_constructions(cpfrom.copy_constructions), data(cpfrom.data) {
            copy_constructions->insert(data);
        }

        S& operator=(const S&) = default;

        S(S&&) = default;
        S& operator=(S&&) = default;

        bool operator==(const S& other) const {
            return data == other.data;
        }

        bool operator!=(const S& other) const {
            return data != other.data;
        }
    };

    auto vtable = py::any_vtable::make<S>();
    ASSERT_FALSE(vtable.is_trivially_copy_constructible());

    std::unordered_set<int> copy_constructions;
    py::any_vector vec1(vtable);

    // NOTE: These don't trigger copy construction because they should hit the rvalue
    // reference dispatch on push_back.
    vec1.push_back(S{copy_constructions, 0});
    vec1.push_back(S{copy_constructions, 1});
    vec1.push_back(S{copy_constructions, 2});
    ASSERT_EQ(copy_constructions.size(), 0ul);

    py::any_vector vec2(vec1);

    ASSERT_EQ(vec1.vtable(), vec2.vtable());
    ASSERT_EQ(vec1.size(), vec2.size());
    EXPECT_GE(vec1.capacity(), vec2.size());
    for (std::size_t ix = 0; ix < vec1.size(); ++ix) {
        EXPECT_EQ(vec1[ix], vec2[ix]);
    }
    EXPECT_EQ(copy_constructions.size(), 3ul);
    EXPECT_EQ(copy_constructions.count(0), 1ul);
    EXPECT_EQ(copy_constructions.count(1), 1ul);
    EXPECT_EQ(copy_constructions.count(2), 1ul);
}

TEST(any_vector, copy_assign_not_trivially_copyable) {
    struct rhs_element {
        std::unordered_set<int>* copy_constructions = nullptr;
        int data = 0;

        rhs_element() = default;

        rhs_element(std::unordered_set<int>& copy_constructions, int data)
            : copy_constructions(&copy_constructions), data(data) {}

        rhs_element(const rhs_element& cpfrom)
            : copy_constructions(cpfrom.copy_constructions), data(cpfrom.data) {
            copy_constructions->insert(data);
        }

        rhs_element& operator=(const rhs_element&) = default;

        rhs_element(rhs_element&&) = default;
        rhs_element& operator=(rhs_element&&) = default;

        bool operator==(const rhs_element& other) const {
            return data == other.data;
        }

        bool operator!=(const rhs_element& other) const {
            return data != other.data;
        }
    };

    struct lhs_element {
        std::unordered_set<int>* destroyed = nullptr;
        int data = 0;

        lhs_element() = default;
        lhs_element(std::unordered_set<int>& destroyed, int data)
            : destroyed(&destroyed), data(data) {}
        lhs_element(const lhs_element&) = default;
        lhs_element& operator=(const lhs_element&) = default;

        lhs_element(lhs_element&& mvfrom) noexcept
            : destroyed(mvfrom.destroyed), data(mvfrom.data) {
            mvfrom.data = -1;
        }

        lhs_element& operator=(lhs_element&& mvfrom) noexcept {
            destroyed = mvfrom.destroyed;
            data = mvfrom.data;
            mvfrom.data = -1;
            return *this;
        }

        ~lhs_element() {
            if (data >= 0) {
                destroyed->emplace(data);
            }
        }

        bool operator==(const lhs_element& other) const {
            return data == other.data;
        }

        bool operator!=(const lhs_element& other) const {
            return data != other.data;
        }
    };

    ASSERT_FALSE((std::is_assignable_v<lhs_element, rhs_element>) );

    auto rhs_vtable = py::any_vtable::make<rhs_element>();
    ASSERT_FALSE(rhs_vtable.is_trivially_copy_constructible());

    auto lhs_vtable = py::any_vtable::make<lhs_element>();
    ASSERT_FALSE(lhs_vtable.is_trivially_destructible());

    std::unordered_set<int> copy_constructions;
    std::unordered_set<int> destroyed;
    py::any_vector rhs(rhs_vtable);

    // NOTE: These don't trigger copy construction because they should hit the rvalue
    // reference dispatch on push_back.
    rhs.push_back(rhs_element{copy_constructions, 0});
    rhs.push_back(rhs_element{copy_constructions, 1});
    rhs.push_back(rhs_element{copy_constructions, 2});

    ASSERT_EQ(copy_constructions.size(), 0ul);

    // Create a new any_vector with a non-rhs_element vtable.
    py::any_vector lhs(lhs_vtable);
    lhs.push_back(lhs_element{destroyed, 4});
    lhs.push_back(lhs_element{destroyed, 5});
    lhs.push_back(lhs_element{destroyed, 6});

    ASSERT_EQ(destroyed.size(), 0ul);

    // Assigning rhs to lhs should use rhs_elements's copy constructor, and should set the
    // vtable of ``lhs`` to the vtable for ``rhs_element``.
    lhs = rhs;

    // All the lhs elements should have been destroyed.
    ASSERT_EQ(destroyed.size(), 3ul);
    EXPECT_EQ(destroyed.count(4), 1ul);
    EXPECT_EQ(destroyed.count(5), 1ul);
    EXPECT_EQ(destroyed.count(6), 1ul);

    // Vector attributes should have been copied.
    ASSERT_EQ(lhs.vtable(), rhs.vtable());
    ASSERT_EQ(lhs.size(), rhs.size());
    EXPECT_GE(lhs.capacity(), rhs.size());

    // rhs values should have been copied into lhs.
    for (std::size_t ix = 0; ix < lhs.size(); ++ix) {
        EXPECT_EQ(lhs[ix], rhs[ix]);
    }

    // We should have copy constructed all the rhs elements.
    EXPECT_EQ(copy_constructions.size(), 3ul);
    EXPECT_EQ(copy_constructions.count(0), 1ul);
    EXPECT_EQ(copy_constructions.count(1), 1ul);
    EXPECT_EQ(copy_constructions.count(2), 1ul);
}

template<typename S>
void push_back_test_body() {
    auto vtable = py::any_vtable::make<S>();
    py::any_vector vec(vtable);

    ASSERT_EQ(vec.size(), 0ul);

    vec.push_back(S{0});
    ASSERT_EQ(vec.size(), 1ul);
    ASSERT_GE(vec.capacity(), 1ul);

    EXPECT_EQ(vec.front(), S{0});
    EXPECT_EQ(vec[0], S{0});
    EXPECT_EQ(vec.back(), S{0});

    vec.push_back(S{1});
    ASSERT_EQ(vec.size(), 2ul);
    ASSERT_GE(vec.capacity(), 2ul);

    EXPECT_EQ(vec.front(), S{0});
    EXPECT_EQ(vec[0], S{0});
    EXPECT_EQ(vec[1], S{1});
    EXPECT_EQ(vec.back(), S{1});

    vec.push_back(S{2});
    ASSERT_EQ(vec.size(), 3ul);
    ASSERT_GE(vec.capacity(), 3ul);

    EXPECT_EQ(vec.front(), S{0});
    EXPECT_EQ(vec[0], S{0});
    EXPECT_EQ(vec[1], S{1});
    EXPECT_EQ(vec[2], S{2});
    EXPECT_EQ(vec.back(), S{2});

    vec.push_back(S{3});
    ASSERT_EQ(vec.size(), 4ul);
    ASSERT_GE(vec.capacity(), 4ul);

    EXPECT_EQ(vec.front(), S{0});
    EXPECT_EQ(vec[0], S{0});
    EXPECT_EQ(vec[1], S{1});
    EXPECT_EQ(vec[2], S{2});
    EXPECT_EQ(vec[3], S{3});
    EXPECT_EQ(vec.back(), S{3});
}

TEST(any_vector, trivial_copy_push_back) {
    struct S {
        int data = 0;

        bool operator==(S other) const {
            return data == other.data;
        }

        bool operator!=(S other) const {
            return data != other.data;
        }
    };
    auto vtable = py::any_vtable::make<S>();
    ASSERT_TRUE(vtable.is_trivially_copyable());

    push_back_test_body<S>();
}

TEST(any_vector, move_is_noexcept_and_trivially_destructible_push_back) {
    struct S {
        int data = 0;

        S() = default;
        S(int data) : data(data) {}
        S(const S& cpfrom) = default;
        S(S&& mvfrom) noexcept : data(mvfrom.data) {}

        S& operator=(const S& cpfrom) = default;
        S& operator=(S&& mvfrom) noexcept {
            data = mvfrom.data;
            return *this;
        }

        bool operator==(S other) const {
            return data == other.data;
        }

        bool operator!=(S other) const {
            return data != other.data;
        }
    };
    auto vtable = py::any_vtable::make<S>();
    ASSERT_FALSE(vtable.is_trivially_copyable());
    ASSERT_TRUE(vtable.move_is_noexcept());
    ASSERT_TRUE(vtable.is_trivially_destructible());

    push_back_test_body<S>();
}

TEST(any_vector, move_is_noexcept_push_back) {
    struct S {
        int data = 0;

        S() = default;
        S(int data) : data(data) {}
        S(const S& cpfrom) = default;
        S(S&& mvfrom) noexcept : data(mvfrom.data) {}

        S& operator=(const S& cpfrom) = default;
        S& operator=(S&& mvfrom) noexcept {
            data = mvfrom.data;
            return *this;
        }

        ~S() {
            // provide a user defined destructor to not be trivially destructible
        }

        bool operator==(S other) const {
            return data == other.data;
        }

        bool operator!=(S other) const {
            return data != other.data;
        }
    };
    auto vtable = py::any_vtable::make<S>();
    ASSERT_FALSE(vtable.is_trivially_copyable());
    ASSERT_TRUE(vtable.move_is_noexcept());
    ASSERT_FALSE(vtable.is_trivially_destructible());
    push_back_test_body<S>();
}

TEST(any_vector, non_noexcept_move_push_back) {
    struct S {
        int data = 0;

        S() = default;
        S(int data) : data(data) {}
        S(const S& cpfrom) = default;
        S(S&& mvfrom) : data(mvfrom.data) {}

        S& operator=(const S& cpfrom) = default;
        S& operator=(S&& mvfrom) {
            data = mvfrom.data;
            return *this;
        }

        bool operator==(S other) const {
            return data == other.data;
        }

        bool operator!=(S other) const {
            return data != other.data;
        }
    };
    auto vtable = py::any_vtable::make<S>();
    ASSERT_FALSE(vtable.is_trivially_copyable());
    ASSERT_FALSE(vtable.move_is_noexcept());
    push_back_test_body<S>();
}

TEST(any_vector, over_aligned_push_back) {
    constexpr std::size_t align = 128;
    ASSERT_GT(align, sizeof(int)) << "where are you compiling this?";
    ASSERT_GT(align, alignof(std::max_align_t));

    struct alignas(align) S {
        int data = 0;

        bool operator==(const S& other) const {
            return data == other.data;
        }

        bool operator!=(const S& other) const {
            return data != other.data;
        }
    };

    push_back_test_body<S>();
}

TEST(any_vector, iterator) {
    py::any_vector vec(py::any_vtable::make<int>());
    for (int n = 0; n < 10; ++n) {
        vec.push_back(n);
    }
    ASSERT_EQ(vec.size(), 10ul);
    const py::any_vector& cref = vec;

    int expected = 0;
    for (auto value : vec) {
        EXPECT_EQ(value, expected);
        ++expected;
    }

    expected = 0;
    for (auto cvalue : cref) {
        EXPECT_EQ(cvalue, expected);
        ++expected;
    }

    for (auto mut_value : vec) {
        mut_value = mut_value.cast<int>() + 10;
    }

    expected = 10;
    for (auto value : vec) {
        EXPECT_EQ(value, expected);
        ++expected;
    }
}

TEST(any_vector, clear) {
    struct S {
        std::unordered_set<int>* destroyed = nullptr;
        int data = 0;

        S() = default;
        S(std::unordered_set<int>& destroyed, int data)
            : destroyed(&destroyed), data(data) {}
        S(const S&) = default;
        S& operator=(const S&) = default;

        S(S&& mvfrom) noexcept : destroyed(mvfrom.destroyed), data(mvfrom.data) {
            mvfrom.data = -1;
        }

        S& operator=(S&& mvfrom) noexcept {
            destroyed = mvfrom.destroyed;
            data = mvfrom.data;
            mvfrom.data = -1;
            return *this;
        }

        ~S() {
            if (data >= 0) {
                destroyed->emplace(data);
            }
        }

        bool operator==(const S& other) const {
            return data == other.data;
        }

        bool operator!=(const S& other) const {
            return data != other.data;
        }
    };

    py::any_vector vec(py::any_vtable::make<S>());
    std::unordered_set<int> destroyed;

    vec.push_back(S{destroyed, 0});
    vec.push_back(S{destroyed, 1});
    vec.push_back(S{destroyed, 2});

    ASSERT_EQ(destroyed.size(), 0ul);
    vec.clear();
    EXPECT_EQ(destroyed.size(), 3ul);
    EXPECT_EQ(destroyed.count(0), 1ul);
    EXPECT_EQ(destroyed.count(1), 1ul);
    EXPECT_EQ(destroyed.count(2), 1ul);
}

TEST(any_vector, destruct) {
    struct S {
        std::unordered_set<int>* destroyed = nullptr;
        int data = 0;

        S() = default;
        S(std::unordered_set<int>& destroyed, int data)
            : destroyed(&destroyed), data(data) {}
        S(const S&) = default;
        S& operator=(const S&) = default;

        S(S&& mvfrom) noexcept : destroyed(mvfrom.destroyed), data(mvfrom.data) {
            mvfrom.data = -1;
        }

        S& operator=(S&& mvfrom) noexcept {
            destroyed = mvfrom.destroyed;
            data = mvfrom.data;
            mvfrom.data = -1;
            return *this;
        }

        ~S() {
            if (data >= 0) {
                destroyed->emplace(data);
            }
        }

        bool operator==(const S& other) const {
            return data == other.data;
        }

        bool operator!=(const S& other) const {
            return data != other.data;
        }
    };

    std::unordered_set<int> destroyed;

    {
        py::any_vector vec(py::any_vtable::make<S>());

        vec.push_back(S{destroyed, 0});
        vec.push_back(S{destroyed, 1});
        vec.push_back(S{destroyed, 2});

        ASSERT_EQ(destroyed.size(), 0ul);
    }

    EXPECT_EQ(destroyed.size(), 3ul);
    EXPECT_EQ(destroyed.count(0), 1ul);
    EXPECT_EQ(destroyed.count(1), 1ul);
    EXPECT_EQ(destroyed.count(2), 1ul);
}

TEST(any_vector, pop_back) {
    struct S {
        std::unordered_set<int>* destroyed = nullptr;
        int data = 0;

        S() = default;
        S(std::unordered_set<int>& destroyed, int data)
            : destroyed(&destroyed), data(data) {}
        S(const S&) = default;
        S& operator=(const S&) = default;

        S(S&& mvfrom) noexcept : destroyed(mvfrom.destroyed), data(mvfrom.data) {
            mvfrom.data = -1;
        }

        S& operator=(S&& mvfrom) noexcept {
            destroyed = mvfrom.destroyed;
            data = mvfrom.data;
            mvfrom.data = -1;
            return *this;
        }

        ~S() {
            if (data >= 0) {
                destroyed->emplace(data);
            }
        }

        bool operator==(const S& other) const {
            return data == other.data;
        }

        bool operator!=(const S& other) const {
            return data != other.data;
        }
    };

    std::unordered_set<int> destroyed;
    py::any_vector vec(py::any_vtable::make<S>());

    vec.push_back(S{destroyed, 0});
    vec.push_back(S{destroyed, 1});
    vec.push_back(S{destroyed, 2});
    ASSERT_EQ(vec.size(), 3ul);
    EXPECT_EQ(vec.back().cast<S>().data, 2);
    ASSERT_EQ(destroyed.size(), 0ul);

    vec.pop_back();

    ASSERT_EQ(vec.size(), 2ul);
    ASSERT_EQ(vec.back().cast<S>().data, 1);
    ASSERT_EQ(destroyed.size(), 1ul);
    EXPECT_EQ(destroyed.count(2), 1ul);

    vec.pop_back();

    ASSERT_EQ(vec.size(), 1ul);
    ASSERT_EQ(vec.back().cast<S>().data, 0);
    ASSERT_EQ(destroyed.size(), 2ul);
    EXPECT_EQ(destroyed.count(1), 1ul);

    vec.pop_back();

    ASSERT_EQ(vec.size(), 0ul);
    // no `.back()` on empty vector
    ASSERT_EQ(destroyed.size(), 3ul);
    EXPECT_EQ(destroyed.count(0), 1ul);
}
}  // namespace test_any_vector
