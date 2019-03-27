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
    auto f = [](const py::any_vtable& vtable, std::size_t count) {
        py::any_vector vec(vtable, count);

        ASSERT_EQ(vec.size(), count);
        ASSERT_GE(vec.capacity(), count);
        ASSERT_EQ(vec.vtable(), vtable);

        // don't check the values because it's unitialized
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
        f(py::any_vtable::make<int>(), count);
        f(py::any_vtable::make<float>(), count);
        f(py::any_vtable::make<S>(), count);
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
