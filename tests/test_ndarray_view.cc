#include <algorithm>
#include <array>
#include <numeric>
#include <ostream>
#include <unordered_set>
#include <vector>

#include "gtest/gtest.h"

#include "libpy/itertools.h"
#include "libpy/ndarray_view.h"
#include "test_utils.h"

namespace test_array_view {
template<typename T, std::size_t n>
std::array<T, n> arange() {
    std::array<T, n> out;
    std::iota(out.begin(), out.end(), 0);
    return out;
}

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
}

namespace std {
template<>
struct hash<test_array_view::custom_object> {
    auto operator()(const test_array_view::custom_object& ob) const {
        return std::hash<int>{}(ob.a);
    }
};
}  // namespace std

namespace test_array_view {
std::ostream& operator<<(std::ostream& s, const custom_object& ob) {
    return s << "<custom_object a=" << ob.a << ", b=" << ob.b << '>';
}

template<typename T>
class array_view : public testing::Test {};
TYPED_TEST_SUITE_P(array_view);

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
    from_container<std::array<std::remove_const_t<TypeParam>, 5>>();
}

TYPED_TEST_P(array_view, from_std_vector) {
    from_container<std::vector<std::remove_const_t<TypeParam>>>();
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
    std::array<std::remove_const_t<TypeParam>, 5> arr = {1, 2, 3, 4, 5};
    py::array_view<TypeParam> view(arr);
    ASSERT_EQ(view.size(), arr.size());

    test_iterator(arr.begin(), arr.end(), view.begin(), view.end());
    test_iterator(arr.cbegin(), arr.cend(), view.cbegin(), view.cend());
}

TYPED_TEST_P(array_view, reverse_iterator) {
    std::array<std::remove_const_t<TypeParam>, 5> arr = {1, 2, 3, 4, 5};
    py::array_view<TypeParam> view(arr);
    ASSERT_EQ(view.size(), arr.size());

    test_iterator(arr.rbegin(), arr.rend(), view.rbegin(), view.rend());
    test_iterator(arr.crbegin(), arr.crend(), view.crbegin(), view.crend());
}

TYPED_TEST_P(array_view, fill) {
    std::array<std::remove_const_t<TypeParam>, 5> arr = {1, 2, 3, 4, 5};
    py::array_view<TypeParam> view(arr);
    ASSERT_EQ(view.size(), arr.size());

    if constexpr (!std::is_const_v<TypeParam>) {
        view.fill(6);

        for (std::size_t ix = 0; ix < arr.size(); ++ix) {
            EXPECT_EQ(view[ix], 6);
            EXPECT_EQ(arr[ix], 6);
        }
    }
}

TYPED_TEST_P(array_view, _2d_indexing) {
    std::array<std::array<std::remove_const_t<TypeParam>, 3>, 4> arr;
    std::size_t value = 1;
    for (std::size_t row = 0; row < 4; ++row) {
        for (std::size_t col = 0; col < 3; ++col) {
            arr[row][col] = value++;
        }
    }
    py::ndarray_view<TypeParam, 2> view(reinterpret_cast<TypeParam*>(arr.data()),
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
    std::array<std::remove_const_t<TypeParam>, 5> arr = {1, 2, 3, 4, 5};
    py::array_view<TypeParam> view(arr);

    EXPECT_EQ(view.front(), arr.front());
    EXPECT_EQ(view.back(), arr.back());

    if constexpr (!std::is_const_v<TypeParam>) {
        view.front() = 6;
        EXPECT_EQ(view.front(), 6);
        EXPECT_EQ(view[0], 6);
        EXPECT_EQ(arr.front(), 6);

        view.back() = 7;
        EXPECT_EQ(view.back(), 7);
        EXPECT_EQ(view[4], 7);
        EXPECT_EQ(arr.back(), 7);
    }
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
    std::array<std::remove_const_t<TypeParam>, 5> arr = {1, 2, 3, 4, 5};
    py::array_view<TypeParam> reverse_view(&arr.back(),
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

TYPED_TEST_P(array_view, slice_npos_end) {
    std::array<std::remove_const_t<TypeParam>, 5> arr = {0, 1, 2, 3, 4};
    py::array_view<TypeParam> view(arr);

    auto slice = view.slice(2);
    ASSERT_EQ(slice.size(), view.size() - 2);
    for (std::size_t ix = 0; ix < slice.size(); ++ix) {
        EXPECT_EQ(slice[ix], view[ix + 2]);
    }
}

TYPED_TEST_P(array_view, slice_start_end) {
    std::array<std::remove_const_t<TypeParam>, 5> arr = {0, 1, 2, 3, 4};
    py::array_view<TypeParam> view(arr);

    auto slice = view.slice(2, 4);
    ASSERT_EQ(slice.size(), 2u);
    for (std::size_t ix = 0; ix < slice.size(); ++ix) {
        EXPECT_EQ(slice[ix], view[ix + 2]);
    }
}

TYPED_TEST_P(array_view, slice_end_greater_than_size_positive_step) {
    std::array<std::remove_const_t<TypeParam>, 5> arr = {0, 1, 2, 3, 4};
    py::array_view<TypeParam> view(arr);

    // slice clips to just the max size
    auto slice = view.slice(2, view.size() + 10);
    ASSERT_EQ(slice.size(), view.size() - 2);
    for (std::size_t ix = 0; ix < slice.size(); ++ix) {
        EXPECT_EQ(slice[ix], view[ix + 2]);
    }
}

TYPED_TEST_P(array_view, slice_start_equal_end) {
    std::array<std::remove_const_t<TypeParam>, 5> arr = {0, 1, 2, 3, 4};
    py::array_view<TypeParam> view(arr);

    // slice protects against start > end
    auto slice = view.slice(2, 2);
    ASSERT_EQ(slice.size(), 0u);
}

TYPED_TEST_P(array_view, slice_positive_step) {
    std::array<std::remove_const_t<TypeParam>, 5> arr = {0, 1, 2, 3, 4};
    py::array_view<TypeParam> view(arr);

    auto slice = view.slice(1, view.snpos, 2);
    ASSERT_EQ(slice.size(), 2u);
    for (std::size_t ix = 0; ix < slice.size(); ++ix) {
        EXPECT_EQ(slice[ix], view[(2 * ix) + 1]);
    }
}

TYPED_TEST_P(array_view, slice_negative_step) {
    std::array<std::remove_const_t<TypeParam>, 5> arr = {0, 1, 2, 3, 4};
    py::array_view<TypeParam> view(arr);

    auto slice = view.slice(view.size() - 1, -1, -1);
    ASSERT_EQ(slice.size(), view.size());
    for (std::size_t ix = 0; ix < slice.size(); ++ix) {
        EXPECT_EQ(slice[ix], view[view.size() - (ix + 1)]);
    }
}

TYPED_TEST_P(array_view, slice_0_step) {
    std::array<std::remove_const_t<TypeParam>, 5> arr = {0, 1, 2, 3, 4};
    py::array_view<TypeParam> view(arr);

    EXPECT_THROW(view.slice(0, 2, 0), std::invalid_argument);
}

TYPED_TEST_P(array_view, construction_from_const_view) {
    std::array<std::remove_const_t<TypeParam>, 5> arr = {0, 1, 2, 3, 4};
    py::array_view<std::remove_const_t<TypeParam>> view{arr};
    py::array_view<const TypeParam> const_view{view};
    ASSERT_EQ(view.size(), const_view.size());

    for (std::size_t ix = 0; ix < view.size(); ++ix) {
        EXPECT_EQ(const_view[ix], view[ix]);
    }
}

TYPED_TEST_P(array_view, for_each_unordered_1d_contig) {
    using non_const_type = std::remove_const_t<TypeParam>;
    auto arr = arange<non_const_type, 6>();
    py::array_view<TypeParam> view(arr);

    std::unordered_set<non_const_type> seen;
    py::for_each_unordered(view, [&](TypeParam& val) { seen.insert(val); });

    std::unordered_set<non_const_type> expected(arr.begin(), arr.end());
    EXPECT_EQ(seen, expected);
}

TYPED_TEST_P(array_view, for_each_unordered_1d_non_contig) {
    using non_const_type = std::remove_const_t<TypeParam>;
    auto arr = arange<non_const_type, 6>();
    py::array_view<TypeParam> view(arr.data(), {3}, {sizeof(non_const_type) * 2});

    std::unordered_set<non_const_type> seen;
    py::for_each_unordered(view, [&](TypeParam& val) { seen.insert(val); });

    std::unordered_set<non_const_type> expected;
    bool insert = true;
    for (const auto& e : arr) {
        if (insert) {
            expected.insert(e);
        }
        insert = !insert;
    }
    EXPECT_EQ(seen, expected);
}

TYPED_TEST_P(array_view, for_each_unordered_2d_c_contig) {
    using non_const_type = std::remove_const_t<TypeParam>;
    auto arr = arange<non_const_type, 2 * 3>();
    py::ndarray_view<TypeParam, 2> view(arr.data(),
                                        {2, 3},
                                        {sizeof(non_const_type) * 3,
                                         sizeof(non_const_type)});

    std::unordered_set<non_const_type> seen;
    py::for_each_unordered(view, [&](TypeParam& val) { seen.insert(val); });

    std::unordered_set<non_const_type> expected(arr.begin(), arr.end());
    EXPECT_EQ(seen, expected);
}

TYPED_TEST_P(array_view, for_each_unordered_2d_f_contig) {
    using non_const_type = std::remove_const_t<TypeParam>;
    auto arr = arange<non_const_type, 2 * 3>();
    py::ndarray_view<TypeParam, 2> view(arr.data(),
                                        {2, 3},
                                        {sizeof(non_const_type),
                                         sizeof(non_const_type) * 2});

    std::unordered_set<non_const_type> seen;
    py::for_each_unordered(view, [&](TypeParam& val) { seen.insert(val); });

    std::unordered_set<non_const_type> expected(arr.begin(), arr.end());
    EXPECT_EQ(seen, expected);
}

TYPED_TEST_P(array_view, for_each_unordered_2d_non_contig) {
    using non_const_type = std::remove_const_t<TypeParam>;
    auto arr = arange<non_const_type, 4 * 6>();
    py::ndarray_view<TypeParam, 2> view(arr.data(),
                                        {2, 3},
                                        {sizeof(non_const_type) * 2,
                                         sizeof(non_const_type) * 2 * 2});

    std::unordered_set<non_const_type> seen;
    py::for_each_unordered(view, [&](TypeParam& val) { seen.insert(val); });

    std::unordered_set<non_const_type> expected;
    for (std::size_t row = 0; row < 2; ++row) {
        for (std::size_t col = 0; col < 3; ++col) {
            expected.insert(view(row, col));
        }
    }
    EXPECT_EQ(seen, expected);
}

REGISTER_TYPED_TEST_SUITE_P(array_view,
                           from_std_array,
                           from_std_vector,
                           iterator,
                           reverse_iterator,
                           fill,
                           _2d_indexing,
                           front_back,
                           virtual_array,
                           negative_strides,
                           slice_npos_end,
                           slice_start_end,
                           slice_end_greater_than_size_positive_step,
                           slice_start_equal_end,
                           slice_positive_step,
                           slice_negative_step,
                           slice_0_step,
                           construction_from_const_view,
                           for_each_unordered_1d_contig,
                           for_each_unordered_1d_non_contig,
                           for_each_unordered_2d_c_contig,
                           for_each_unordered_2d_f_contig,
                           for_each_unordered_2d_non_contig);

template<typename T>
struct tuple_to_types;

template<typename... Ts>
struct tuple_to_types<std::tuple<Ts...>> {
    using type = testing::Types<Ts...>;
};

template<typename... Ts>
struct add_const;

template<>
struct add_const<> {
    using type = std::tuple<>;
};

template<typename T, typename... Ts>
struct add_const<T, Ts...> {
    using type =
        py::meta::type_cat<std::tuple<T, const T>, typename add_const<Ts...>::type>;
};

using array_view_test_types =
    typename add_const<char, unsigned char, int, float, double, custom_object>::type;

INSTANTIATE_TYPED_TEST_SUITE_P(typed_,
                              array_view,
                              typename tuple_to_types<array_view_test_types>::type);

class array_view_extra : public with_python_interpreter {};

TEST_F(array_view_extra, from_buffer_protocol) {
    py::scoped_ref<> ns = RUN_PYTHON(R"(
        import numpy as np
        array = np.array([1.5, 2.5, 3.5], dtype='f8')
        view = memoryview(array)
    )");
    ASSERT_TRUE(ns);

    py::borrowed_ref view = PyDict_GetItemString(ns.get(), "view");
    ASSERT_TRUE(view);

    auto [array_view, buf] = py::array_view<double>::from_buffer_protocol(view);
    ::testing::StaticAssertTypeEq<decltype(array_view), py::array_view<double>>();

    ASSERT_EQ(array_view.size(), 3u);
    EXPECT_EQ(array_view[0], 1.5);
    EXPECT_EQ(array_view[1], 2.5);
    EXPECT_EQ(array_view[2], 3.5);

    EXPECT_THROW(({ py::array_view<float>::from_buffer_protocol(view); }), py::exception);
    PyErr_Clear();

    EXPECT_THROW(({ py::ndarray_view<double, 2>::from_buffer_protocol(view); }),
                 py::exception);
    PyErr_Clear();
}

class any_ref_array_view : public with_python_interpreter {};

TEST_F(any_ref_array_view, test_read) {
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

TEST_F(any_ref_array_view, test_write) {
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

TEST_F(any_ref_array_view, test_cast) {
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

TEST_F(any_ref_array_view, negative_strides) {
    std::array<int, 5> arr = {1, 2, 3, 4, 5};
    py::array_view<py::any_ref> reverse_view(&arr.back(),
                                             {5},
                                             {-static_cast<std::int64_t>(sizeof(int))});

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

TEST_F(any_ref_array_view, test_vtable) {
    auto test = [](auto data) {
        py::scoped_ref ndarray = py::move_to_numpy_array(std::vector{data});
        ASSERT_TRUE(ndarray);
        auto view = py::from_object<py::array_view<py::any_cref>>(ndarray);
        EXPECT_EQ(view.vtable(), py::any_vtable::make<decltype(data)>());
        ASSERT_EQ(view.size(), 1);
        EXPECT_EQ(view[0], data);
    };

    std::tuple dtypes = {py::py_bool{true},
                         std::int8_t{1},
                         std::int16_t{1},
                         std::int32_t{1},
                         std::int64_t{1},
                         std::uint8_t{1},
                         std::uint16_t{1},
                         std::uint32_t{1},
                         std::uint64_t{1},
                         float{1},
                         double{1},
                         py::none,
                         std::array<char, 10>{1},
                         std::array<char, 63>{1},
                         py::datetime64<py::chrono::ns>{1},
                         py::datetime64<py::chrono::us>{1},
                         py::datetime64<py::chrono::ms>{1},
                         py::datetime64<py::chrono::s>{1},
                         py::datetime64<py::chrono::m>{1},
                         py::datetime64<py::chrono::h>{1},
                         py::datetime64<py::chrono::D>{1}};

    std::apply([&](auto... args) { (test(args), ...); }, dtypes);

    EXPECT_THROW(test(std::array<char, 64>{1}), py::exception);
    expect_pyerr_type_and_message(
        PyExc_TypeError,
        "cannot create vtable for fixed width strings with size greater than 63");
    PyErr_Clear();
}
}  // namespace test_array_view
