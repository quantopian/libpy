#include <array>
#include <ostream>
#include <unordered_map>
#include <vector>

#include "gtest/gtest.h"

#include "libpy/devirtualize.h"
#include "libpy/hash.h"
#include "libpy/ndarray_view.h"

namespace test_devirtualize {
/** Hashable pair of `std::byte*`.
 */
struct byteptr_pair {
    std::byte* first;
    std::byte* second;

    bool operator==(const byteptr_pair& other) const {
        return first == other.first && second == other.second;
    }

    bool operator!=(const byteptr_pair& other) const {
        return !(*this == other);
    }
};
}  // namespace test_devirtualize

namespace std {
template<>
struct hash<test_devirtualize::byteptr_pair> {
    std::size_t operator()(const test_devirtualize::byteptr_pair& p) const noexcept {
        return py::hash_many(p.first, p.second);
    }
};
}  // namespace std

namespace test_devirtualize {
TEST(devirtualize, for_each) {
    std::vector<int> a = {1, 2, 3};
    std::byte* a_data = reinterpret_cast<std::byte*>(a.data());

    std::vector<float> b = {1.5, 2.5, 3.5};
    std::byte* b_data = reinterpret_cast<std::byte*>(b.data());

    std::vector<int> c = {4, 5, 6};
    std::byte* c_data = reinterpret_cast<std::byte*>(c.data());

    std::vector<py::array_view<py::any_ref>> virtualized = {a, b, c};

    py::devirtualize<int, float> devirtualized(virtualized);

    std::vector<std::byte*> int_seen;
    std::vector<std::byte*> expected_int_seen = {a_data, c_data};
    std::vector<std::byte*> float_seen;
    std::vector<std::byte*> expected_float_seen = {b_data};
    int called = 0;

    devirtualized.for_each([&](const auto& arr) {
        using T = decltype(arr);
        if constexpr (std::is_same_v<T, const py::array_view<int>&>) {
            int_seen.emplace_back(arr.buffer());
        }
        else if constexpr (std::is_same_v<T, const py::array_view<float>&>) {
            float_seen.emplace_back(arr.buffer());
        }
        else {
            static_assert(!std::is_same_v<T, T>, "unknown view type");
        }

        ++called;
    });

    EXPECT_EQ(int_seen, expected_int_seen);
    EXPECT_EQ(float_seen, expected_float_seen);
    EXPECT_EQ(called, 3);
}

TEST(devirtualize, binary_for_each) {
    std::vector<int> a0 = {1, 2, 3};
    std::byte* a0_data = reinterpret_cast<std::byte*>(a0.data());

    std::vector<float> b0 = {1.5, 2.5, 3.5};
    std::byte* b0_data = reinterpret_cast<std::byte*>(b0.data());

    std::vector<int> c0 = {4, 5, 6};
    std::byte* c0_data = reinterpret_cast<std::byte*>(c0.data());

    std::vector<int> a1 = {10, 20, 30};
    std::byte* a1_data = reinterpret_cast<std::byte*>(a1.data());

    std::vector<float> b1 = {10.5, 20.5, 30.5};
    std::byte* b1_data = reinterpret_cast<std::byte*>(b1.data());

    std::vector<int> c1 = {40, 50, 60};
    std::byte* c1_data = reinterpret_cast<std::byte*>(c1.data());

    std::vector<py::array_view<py::any_ref>> a_virt = {a0, b0, c0};
    std::vector<py::array_view<py::any_ref>> b_virt = {a1, b1, c1};

    py::nwise_devirtualize<std::tuple<int, int>, std::tuple<float, float>> devirtualized{
        a_virt, b_virt};

    std::vector<std::array<std::byte*, 2>> int_seen;
    std::vector<std::array<std::byte*, 2>> expected_int_seen = {{a0_data, a1_data},
                                                                {c0_data, c1_data}};
    std::vector<std::array<std::byte*, 2>> float_seen;
    std::vector<std::array<std::byte*, 2>> expected_float_seen = {{b0_data, b1_data}};
    int called = 0;

    devirtualized.for_each([&](const auto& arr0, const auto& arr1) {
        using T0 = decltype(arr0);
        using T1 = decltype(arr1);
        testing::StaticAssertTypeEq<T0, T1>();

        if constexpr (std::is_same_v<T0, const py::array_view<int>&>) {
            int_seen.push_back({arr0.buffer(), arr1.buffer()});
        }
        else if constexpr (std::is_same_v<T0, const py::array_view<float>&>) {
            float_seen.push_back({arr0.buffer(), arr1.buffer()});
        }
        else {
            static_assert(!std::is_same_v<T0, T0>, "unknown view type");
        }

        ++called;
    });

    EXPECT_EQ(int_seen, expected_int_seen);
    EXPECT_EQ(float_seen, expected_float_seen);
    EXPECT_EQ(called, 3);
}

TEST(devirtualize, for_each_with_ix) {
    std::vector<int> a = {1, 2, 3};
    std::byte* a_data = reinterpret_cast<std::byte*>(a.data());

    std::vector<float> b = {1.5, 2.5, 3.5};
    std::byte* b_data = reinterpret_cast<std::byte*>(b.data());

    std::vector<int> c = {4, 5, 6};
    std::byte* c_data = reinterpret_cast<std::byte*>(c.data());

    std::vector<py::array_view<py::any_ref>> virtualized = {a, b, c};

    py::devirtualize<int, float> devirtualized(virtualized);

    std::unordered_map<std::byte*, std::size_t> seen_ix;
    std::unordered_map<std::byte*, std::size_t> expected_ix = {{a_data, 0},
                                                               {b_data, 1},
                                                               {c_data, 2}};

    std::vector<std::byte*> int_seen;
    std::vector<std::byte*> expected_int_seen = {a_data, c_data};
    std::vector<std::byte*> float_seen;
    std::vector<std::byte*> expected_float_seen = {b_data};
    int called = 0;

    devirtualized.for_each_with_ix([&](std::size_t ix, const auto& arr) {
        using T = decltype(arr);
        if constexpr (std::is_same_v<T, const py::array_view<int>&>) {
            int_seen.emplace_back(arr.buffer());
        }
        else if constexpr (std::is_same_v<T, const py::array_view<float>&>) {
            float_seen.emplace_back(arr.buffer());
        }
        else {
            static_assert(!std::is_same_v<T, T>, "unknown view type");
        }

        seen_ix[arr.buffer()] = ix;
        ++called;
    });

    EXPECT_EQ(int_seen, expected_int_seen);
    EXPECT_EQ(float_seen, expected_float_seen);
    EXPECT_EQ(called, 3);
    EXPECT_EQ(seen_ix, expected_ix);
}

TEST(devirtualize, binary_for_each_with_ix) {
    std::vector<int> a0 = {1, 2, 3};
    std::byte* a0_data = reinterpret_cast<std::byte*>(a0.data());

    std::vector<float> b0 = {1.5, 2.5, 3.5};
    std::byte* b0_data = reinterpret_cast<std::byte*>(b0.data());

    std::vector<int> c0 = {4, 5, 6};
    std::byte* c0_data = reinterpret_cast<std::byte*>(c0.data());

    std::vector<int> a1 = {10, 20, 30};
    std::byte* a1_data = reinterpret_cast<std::byte*>(a1.data());

    std::vector<float> b1 = {10.5, 20.5, 30.5};
    std::byte* b1_data = reinterpret_cast<std::byte*>(b1.data());

    std::vector<int> c1 = {40, 50, 60};
    std::byte* c1_data = reinterpret_cast<std::byte*>(c1.data());

    std::vector<py::array_view<py::any_ref>> a_virt = {a0, b0, c0};
    std::vector<py::array_view<py::any_ref>> b_virt = {a1, b1, c1};

    py::nwise_devirtualize<std::tuple<int, int>, std::tuple<float, float>> devirtualized{
        a_virt, b_virt};

    std::unordered_map<byteptr_pair, std::size_t> seen_ix;
    std::unordered_map<byteptr_pair, std::size_t> expected_ix = {
        {{a0_data, a1_data}, 0},
        {{b0_data, b1_data}, 1},
        {{c0_data, c1_data}, 2},
    };

    std::vector<std::array<std::byte*, 2>> int_seen;
    std::vector<std::array<std::byte*, 2>> expected_int_seen = {{a0_data, a1_data},
                                                                {c0_data, c1_data}};
    std::vector<std::array<std::byte*, 2>> float_seen;
    std::vector<std::array<std::byte*, 2>> expected_float_seen = {{b0_data, b1_data}};
    int called = 0;

    devirtualized.for_each_with_ix(
        [&](std::size_t ix, const auto& arr0, const auto& arr1) {
            using T0 = decltype(arr0);
            using T1 = decltype(arr1);
            testing::StaticAssertTypeEq<T0, T1>();

            if constexpr (std::is_same_v<T0, const py::array_view<int>&>) {
                int_seen.push_back({arr0.buffer(), arr1.buffer()});
            }
            else if constexpr (std::is_same_v<T0, const py::array_view<float>&>) {
                float_seen.push_back({arr0.buffer(), arr1.buffer()});
            }
            else {
                static_assert(!std::is_same_v<T0, T0>, "unknown view type");
            }

            seen_ix[{arr0.buffer(), arr1.buffer()}] = ix;
            ++called;
        });

    EXPECT_EQ(int_seen, expected_int_seen);
    EXPECT_EQ(float_seen, expected_float_seen);
    EXPECT_EQ(called, 3);
    EXPECT_EQ(seen_ix, expected_ix);
}

TEST(devirtualize, unknown_type) {
    std::vector<int> a;
    std::vector<float> b;
    std::vector<int> c;

    std::vector<py::array_view<py::any_ref>> virtualized = {a, b, c};

    EXPECT_THROW(({ py::devirtualize<int, double> devirtualized{virtualized}; }),
                 std::bad_any_cast);
}

TEST(devirtualize, mismatched_size) {
    std::vector<int> a;
    std::vector<int> b;
    std::vector<int> c;

    std::vector<py::array_view<py::any_ref>> a_virt = {a};
    std::vector<py::array_view<py::any_ref>> b_virt = {b, c};

    EXPECT_THROW(
        ({
            py::nwise_devirtualize<std::tuple<int, int>> devirtualized{a_virt, b_virt};
        }),
        std::invalid_argument);
}

TEST(devirtualize, inout) {
    std::vector<int> a_in = {1};
    std::vector<int> a_out = {-1};

    std::vector<float> b_in = {1.5};
    std::vector<float> b_out = {-1.5};

    std::vector<int> c_in = {2};
    std::vector<int> c_out = {-2};

    std::vector<py::array_view<py::any_cref>> virtualized_in = {a_in, b_in, c_in};
    std::vector<py::array_view<py::any_ref>> virtualized_out = {a_out, b_out, c_out};

    py::nwise_devirtualize<std::tuple<const int, int>, std::tuple<const float, float>>
        devirtualized{virtualized_in, virtualized_out};

    devirtualized.for_each([](const auto& in, const auto& out) { out[0] = in[0]; });

    EXPECT_EQ(a_out, a_in);
    EXPECT_EQ(b_out, b_in);
    EXPECT_EQ(c_out, c_in);
}

TEST(devirtualize, inout_mixed_type) {
    std::vector<int> a_in = {1};
    std::vector<float> a_out = {-1};
    std::vector<float> a_expected_out = {1};

    std::vector<float> b_in = {1.5};
    std::vector<double> b_out = {-1.5};
    std::vector<double> b_expected_out = {1.5};

    std::vector<int> c_in = {2};
    std::vector<float> c_out = {-2};
    std::vector<float> c_expected_out = {2};

    std::vector<py::array_view<py::any_cref>> virtualized_in = {a_in, b_in, c_in};
    std::vector<py::array_view<py::any_ref>> virtualized_out = {a_out, b_out, c_out};

    py::nwise_devirtualize<std::tuple<const int, float>, std::tuple<const float, double>>
        devirtualized{virtualized_in, virtualized_out};

    devirtualized.for_each([](const auto& in, const auto& out) { out[0] = in[0]; });

    EXPECT_EQ(a_out, a_expected_out);
    EXPECT_EQ(b_out, b_expected_out);
    EXPECT_EQ(c_out, c_expected_out);
}
}  // namespace test_devirtualize
