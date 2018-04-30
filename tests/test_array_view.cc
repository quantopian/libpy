#include <algorithm>
#include <array>
#include <vector>
#include <ostream>

#include "gtest/gtest.h"

#include "libpy/array_view.h"

/** A non-fundamental type.
 */
class custom_object {
public:
    int a;
    float b;

    custom_object(int a) : a(a), b(a / 2.0) {};

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
    EXPECT_EQ(arr_mm, arr_end)
        << "mismatched elements at index: " << std::distance(arr_mm, arr_begin) << ": "
        << *arr_mm << " != " << *view_mm;
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

REGISTER_TYPED_TEST_CASE_P(array_view,
                           from_std_array,
                           from_std_vector,
                           iterator,
                           reverse_iterator);

using array_view_types =
    testing::Types<char, unsigned char, int, float, double, custom_object>;
INSTANTIATE_TYPED_TEST_CASE_P(typed_, array_view, array_view_types);
