#include <algorithm>
#include <array>
#include <ostream>
#include <vector>

#include "gtest/gtest.h"

#include "libpy/array_view.h"
#include "libpy/itertools.h"
#include "libpy/utils.h"

namespace test_array_view {
/** A non-fundamental type.
 */
class custom_object {
public:
    int a;
    float b;

    custom_object() = default;
    custom_object(int a) : a(a), b(a / 2.0){};

    bool operator==(const custom_object& other) const {
        return a == other.a and b == other.b;
    }
};

std::ostream& operator<<(std::ostream& s, const custom_object& ob) {
    return s << "<custom_object a=" << ob.a << ", b=" << ob.b << '>';
}

template<typename T>
class array_view : public testing::Test {};
TYPED_TEST_CASE_P(array_view);

template<typename Container>
void from_container() {
    Container c = {1, 2, 3, 4, 5};
    py::array_view<typename Container::value_type> view(c);
    ASSERT_EQ(view.size(), c.size());

    for (std::size_t n = 0; n < c.size(); ++n) {
        EXPECT_EQ(view[n], c[n]);
        EXPECT_EQ(view(n), c[n]);
        EXPECT_EQ(view.at(n), c[n]);
        EXPECT_EQ(view.at({n}), c[n]);
    }
};

TYPED_TEST_P(array_view, from_std_array) {
    from_container<std::array<TypeParam, 5>>();
}

TYPED_TEST_P(array_view, from_std_vector) {
    from_container<std::vector<TypeParam>>();
}

template<typename It1, typename It2>
void test_iterator(It1 arr_begin, It1 arr_end, It2 view_begin, It2 view_end) {
    auto [arr_mm, view_mm] = std::mismatch(arr_begin, arr_end, view_begin);
    EXPECT_EQ(arr_mm, arr_end) << "mismatched elements at index: "
                               << std::distance(arr_mm, arr_begin) << ": " << *arr_mm
                               << " != " << *view_mm;
    EXPECT_EQ(view_mm, view_end);
}

TYPED_TEST_P(array_view, iterator) {
    std::array<TypeParam, 5> arr = {1, 2, 3, 4, 5};
    py::array_view<TypeParam> view(arr);
    ASSERT_EQ(view.size(), arr.size());

    test_iterator(arr.begin(), arr.end(), view.begin(), view.end());
    test_iterator(arr.cbegin(), arr.cend(), view.cbegin(), view.cend());
}

TYPED_TEST_P(array_view, reverse_iterator) {
    std::array<TypeParam, 5> arr = {1, 2, 3, 4, 5};
    py::array_view<TypeParam> view(arr);
    ASSERT_EQ(view.size(), arr.size());

    test_iterator(arr.rbegin(), arr.rend(), view.rbegin(), view.rend());
    test_iterator(arr.crbegin(), arr.crend(), view.crbegin(), view.crend());
}

TYPED_TEST_P(array_view, _2d_indexing) {
    std::array<std::array<TypeParam, 3>, 4> arr;
    std::size_t value = 1;
    for (std::size_t row = 0; row < 4; ++row) {
        for (std::size_t col = 0; col < 3; ++col) {
            arr[row][col] = value++;
        }
    }
    py::ndarray_view<TypeParam, 2> view(reinterpret_cast<char*>(arr.data()),
                                        {4, 3},
                                        {sizeof(TypeParam) * 3, sizeof(TypeParam)});

    for (std::size_t row = 0; row < 4; ++row) {
        for (std::size_t col = 0; col < 3; ++col) {
            EXPECT_EQ(view(row, col), arr[row][col]);
            EXPECT_EQ(view.at(row, col), arr[row][col]);
            EXPECT_EQ(view.at({row, col}), arr[row][col]);
        }
    }
}

TYPED_TEST_P(array_view, front_back) {
    std::array<TypeParam, 5> arr = {1, 2, 3, 4, 5};
    py::array_view<TypeParam> view(arr);

    EXPECT_EQ(view.front(), arr.front());
    EXPECT_EQ(view.back(), arr.back());

    view.front() = 6;
    EXPECT_EQ(view.front(), 6);
    EXPECT_EQ(view[0], 6);
    EXPECT_EQ(arr.front(), 6);

    view.back() = 7;
    EXPECT_EQ(view.back(), 7);
    EXPECT_EQ(view[4], 7);
    EXPECT_EQ(arr.back(), 7);
}

TYPED_TEST_P(array_view, virtual_array) {
    TypeParam value(1);

    std::size_t size = 10;
    auto view = py::array_view<TypeParam>::virtual_array(value, size);
    EXPECT_EQ(view.size(), size);
    EXPECT_EQ(view.strides()[0], 0);

    std::size_t iterations = 0;

    for (const auto& element : view) {
        EXPECT_EQ(element, value);
        EXPECT_EQ(std::addressof(element), std::addressof(value));

        ++iterations;
    }

    EXPECT_EQ(iterations, size);
}

TYPED_TEST_P(array_view, negative_strides) {
    std::array<TypeParam, 5> arr = {1, 2, 3, 4, 5};
    py::array_view<TypeParam> reverse_view(reinterpret_cast<char*>(&arr.back()),
                                           {5},
                                           {-static_cast<std::int64_t>(
                                               sizeof(TypeParam))});

    EXPECT_EQ(reverse_view[0], arr[arr.size() - 1]);
    EXPECT_EQ(reverse_view[1], arr[arr.size() - 2]);
    EXPECT_EQ(reverse_view[2], arr[arr.size() - 3]);
    EXPECT_EQ(reverse_view[3], arr[arr.size() - 4]);
    EXPECT_EQ(reverse_view[4], arr[arr.size() - 5]);

    // check the iterator properly decrements the pointer
    std::size_t ix = 0;
    for (const auto& value : reverse_view) {
        EXPECT_EQ(value, arr[arr.size() - ix++ - 1]);
    }
}

REGISTER_TYPED_TEST_CASE_P(array_view,
                           from_std_array,
                           from_std_vector,
                           iterator,
                           reverse_iterator,
                           _2d_indexing,
                           front_back,
                           virtual_array,
                           negative_strides);

using array_view_types =
    testing::Types<char, unsigned char, int, float, double, custom_object>;
INSTANTIATE_TYPED_TEST_CASE_P(typed_, array_view, array_view_types);

TEST(any_ref_array_view, test_read) {
    std::array<int, 5> underlying = {0, 1, 2, 3, 4};
    py::array_view<py::any_ref> dynamic_view(underlying);

    ASSERT_EQ(dynamic_view.size(), underlying.size());

    for (auto [a, b] : py::zip(dynamic_view, underlying)) {
        EXPECT_EQ(a.cast<int>(), b);
    }

    for (std::size_t ix = 0; ix < dynamic_view.size(); ++ix) {
        auto a = dynamic_view[ix];
        auto b = underlying[ix];
        EXPECT_EQ(a.cast<int>(), b);
    }
}

TEST(any_ref_array_view, test_write) {
    std::array<int, 5> underlying = {0, 1, 2, 3, 4};
    std::array<int, 5> original_copy = underlying;

    py::array_view<py::any_ref> dynamic_view(underlying);

    ASSERT_EQ(dynamic_view.size(), underlying.size());

    for (auto [a, b] : py::zip(dynamic_view, underlying)) {
        a = b + 1;
    }

    for (auto [a, b] : py::zip(underlying, original_copy)) {
        EXPECT_EQ(a, b + 1);
    }

    for (std::size_t ix = 0; ix < dynamic_view.size(); ++ix) {
        dynamic_view[ix] = original_copy[ix] + 1;

        EXPECT_EQ(dynamic_view[ix].cast<int>(), original_copy[ix] + 1);
    }

    auto typed_view = dynamic_view.cast<int>();
    ASSERT_EQ(typed_view.size(), underlying.size());

    for (std::size_t ix = 0; ix < dynamic_view.size(); ++ix) {
        // assign through the typed view that comes from cast
        typed_view[ix] = original_copy[ix] + 2;

        EXPECT_EQ(dynamic_view[ix].cast<int>(), original_copy[ix] + 2);
    }
}

TEST(any_ref_array_view, test_cast) {
    std::array<int, 5> underlying = {0, 1, 2, 3, 4};
    py::array_view<py::any_ref> dynamic_view(underlying);

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

    EXPECT_THROW(dynamic_view.cast<float>(), std::bad_any_cast);
    EXPECT_THROW(dynamic_view.cast<long>(), std::bad_any_cast);
    EXPECT_THROW(dynamic_view.cast<S>(), std::bad_any_cast);

    auto typed_view = dynamic_view.cast<int>();
    testing::StaticAssertTypeEq<decltype(typed_view), py::array_view<int>>();

    for (auto [a, b, c] : py::zip(dynamic_view, typed_view, underlying)) {
        EXPECT_THROW(a.cast<float>(), std::bad_any_cast);
        EXPECT_THROW(a.cast<long>(), std::bad_any_cast);
        EXPECT_THROW(a.cast<S>(), std::bad_any_cast);

        EXPECT_EQ(a.cast<int>(), b);
        EXPECT_EQ(a.cast<int>(), c);
    }
}

TEST(any_ref_array_view, negative_strides) {
    std::array<int, 5> arr = {1, 2, 3, 4, 5};
    py::array_view<py::any_ref> reverse_view(reinterpret_cast<char*>(&arr.back()),
                                             {5},
                                             {-static_cast<std::int64_t>(sizeof(int))},
                                             py::any_vtable::make<int>());

    EXPECT_EQ(reverse_view[0], arr[arr.size() - 1]);
    EXPECT_EQ(reverse_view[1], arr[arr.size() - 2]);
    EXPECT_EQ(reverse_view[2], arr[arr.size() - 3]);
    EXPECT_EQ(reverse_view[3], arr[arr.size() - 4]);
    EXPECT_EQ(reverse_view[4], arr[arr.size() - 5]);

    // check the iterator properly decrements the pointer
    std::size_t ix = 0;
    for (const auto& value : reverse_view) {
        EXPECT_EQ(value, arr[arr.size() - ix++ - 1]);
    }

}
}  // namespace test_array_view
