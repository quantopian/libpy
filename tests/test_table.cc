#include <algorithm>
#include <array>
#include <ostream>
#include <vector>

#include "gtest/gtest.h"

#include "libpy/char_sequence.h"
#include "libpy/table.h"

namespace test_table {
using namespace py::cs::literals;

/** A non-fundamental type.
 */
class custom_object {
public:
    int a;
    float b;

    custom_object() = delete;
    explicit custom_object(int a) : a(a), b(a / 2.0){};

    bool operator==(const custom_object& other) const {
        return a == other.a && b == other.b;
    }

    custom_object& operator++() {
        ++a;
        ++b;
        return *this;
    }
};

std::ostream& operator<<(std::ostream& s, const custom_object& ob) {
    return s << "<custom_object a=" << ob.a << ", b=" << ob.b << '>';
}

TEST(row_view, assign) {
    using R = py::row_view<py::C<std::int64_t>("a"_cs),
                           py::C<double>("b"_cs),
                           py::C<custom_object>("c"_cs)>;

    std::int64_t a = 1;
    double b = 2.5;
    custom_object c(3);
    R row_view(&a, &b, &c);

    EXPECT_EQ(row_view.get("a"_cs), 1L);
    EXPECT_EQ(row_view.get("b"_cs), 2.5);
    EXPECT_EQ(row_view.get("c"_cs), custom_object(3));

    // assign through the view
    row_view = std::make_tuple(2, 3.5, custom_object(4));

    EXPECT_EQ(row_view.get("a"_cs), 2L);
    EXPECT_EQ(a, 2l);
    EXPECT_EQ(row_view.get("b"_cs), 3.5);
    EXPECT_EQ(b, 3.5);
    EXPECT_EQ(row_view.get("c"_cs), custom_object(4));
    EXPECT_EQ(c, custom_object(4));

    // assign to the underlying
    a = 3;
    b = 4.5;
    c = custom_object(5);

    EXPECT_EQ(row_view.get("a"_cs), 3L);
    EXPECT_EQ(row_view.get("b"_cs), 4.5);
    EXPECT_EQ(row_view.get("c"_cs), custom_object(5));

    // assign from a row view: this should dereference assign to the underlying
    std::int64_t new_a = 4;
    double new_b = 5.5;
    custom_object new_c(6);
    R new_row_view(&new_a, &new_b, &new_c);

    row_view = new_row_view;

    EXPECT_EQ(row_view, new_row_view);
    EXPECT_EQ(row_view.get("a"_cs), 4L);
    EXPECT_EQ(a, 4L);
    EXPECT_EQ(row_view.get("b"_cs), 5.5);
    EXPECT_EQ(b, 5.5);
    EXPECT_EQ(row_view.get("c"_cs), custom_object(6));
    EXPECT_EQ(c, custom_object(6));
}

TEST(row_view, structured_binding) {
    using R = py::row_view<py::C<std::int64_t>("a"_cs),
                           py::C<double>("b"_cs),
                           py::C<custom_object>("c"_cs)>;

    std::int64_t a = 1;
    double b = 2.5;
    custom_object c(3);
    R row_view(&a, &b, &c);

    auto [bound_a, bound_b, bound_c] = row_view;
    EXPECT_EQ(bound_a, a);
    EXPECT_EQ(bound_b, b);
    EXPECT_EQ(bound_c, c);

    auto& [ref_a, ref_b, ref_c] = row_view;
    EXPECT_EQ(ref_a, a);
    EXPECT_EQ(ref_b, b);
    EXPECT_EQ(ref_c, c);

    ref_a = 2;
    ref_b = 3.5;
    ref_c = custom_object(4);

    EXPECT_EQ(a, 2);
    EXPECT_EQ(b, 3.5);
    EXPECT_EQ(c, custom_object(4));
}

TEST(row_view, subset) {
    using R = py::row_view<py::C<std::int64_t>("a"_cs),
                           py::C<double>("b"_cs),
                           py::C<custom_object>("c"_cs)>;

    std::int64_t a = 1;
    double b = 2.5;
    custom_object c(3);
    R row_view(&a, &b, &c);

    {
        // drop the `c` column
        auto subset = row_view.subset("a"_cs, "b"_cs);
        EXPECT_EQ(subset, std::make_tuple(a, b));
    }

    {
        // transpose columns
        auto subset = row_view.subset("b"_cs, "a"_cs, "c"_cs);
        EXPECT_EQ(subset, std::make_tuple(b, a, c));
    }

    {
        // mutate through subset
        auto subset = row_view.subset("a"_cs, "b"_cs);
        subset = std::make_tuple(2, 3.5);
        EXPECT_EQ(subset, std::make_tuple(2, 3.5));
        EXPECT_EQ(row_view, std::make_tuple(2, 3.5, custom_object(3)));
    }
}

TEST(row_view, drop) {
    using R = py::row_view<py::C<std::int64_t>("a"_cs),
                           py::C<double>("b"_cs),
                           py::C<custom_object>("c"_cs)>;

    std::int64_t a = 1;
    double b = 2.5;
    custom_object c(3);
    R row_view(&a, &b, &c);

    {
        // drop the `c` column
        auto subset = row_view.drop("c"_cs);
        EXPECT_EQ(subset, std::make_tuple(a, b));
    }

    {
        // drop 2 columns
        auto subset = row_view.drop("a"_cs, "c"_cs);
        EXPECT_EQ(subset, std::make_tuple(b));
    }

    {
        // mutate through subset
        auto subset = row_view.drop("b"_cs);
        subset = std::make_tuple(2, custom_object(4));
        EXPECT_EQ(subset, std::make_tuple(2, custom_object(4)));
        EXPECT_EQ(row_view, std::make_tuple(2, 2.5, custom_object(4)));
    }
}

TEST(row_view, relabel) {
    using R = py::row_view<py::C<std::int64_t>("a"_cs),
                           py::C<double>("b"_cs),
                           py::C<custom_object>("c"_cs)>;

    std::int64_t a = 1;
    double b = 2.5;
    custom_object c(3);
    R row_view(&a, &b, &c);

    auto relabeled = row_view.relabel(std::make_pair("a"_cs, "a-new"_cs),
                                      std::make_pair("c"_cs, "c-new"_cs));

    EXPECT_EQ(&relabeled.get("a-new"_cs), &row_view.get("a"_cs));
    EXPECT_EQ(&relabeled.get("b"_cs), &row_view.get("b"_cs));
    EXPECT_EQ(&relabeled.get("c-new"_cs), &row_view.get("c"_cs));
}

TEST(row, assign) {
    using R = py::row<py::C<std::int64_t>("a"_cs),
                      py::C<double>("b"_cs),
                      py::C<custom_object>("c"_cs)>;

    std::int64_t a = 1;
    double b = 2.5;
    custom_object c(3);
    R row(a, b, c);

    auto expect_original_unchanged = [&] {
        EXPECT_EQ(a, 1L);
        EXPECT_EQ(b, 2.5);
        EXPECT_EQ(c, custom_object(3));
    };

    EXPECT_EQ(row.get("a"_cs), 1L);
    EXPECT_EQ(row.get("b"_cs), 2.5);
    EXPECT_EQ(row.get("c"_cs), custom_object(3));

    // assign with a tuple
    row = std::make_tuple(2, 3.5, custom_object(4));

    EXPECT_EQ(row.get("a"_cs), 2L);
    EXPECT_EQ(row.get("b"_cs), 3.5);
    EXPECT_EQ(row.get("c"_cs), custom_object(4));
    expect_original_unchanged();

    // assign with another row
    R new_row(3, 4.5, custom_object(5));
    row = new_row;
    EXPECT_EQ(row, new_row);
    EXPECT_EQ(row.get("a"_cs), 3L);
    EXPECT_EQ(row.get("b"_cs), 4.5);
    EXPECT_EQ(row.get("c"_cs), custom_object(5));

    // assign with a view
    a = 4;
    b = 5.5;
    c = custom_object(6);

    using RV = py::row_view<py::C<std::int64_t>("a"_cs),
                            py::C<double>("b"_cs),
                            py::C<custom_object>("c"_cs)>;

    RV row_view(&a, &b, &c);

    row = row_view;
    EXPECT_EQ(row, row_view);
    EXPECT_EQ(row.get("a"_cs), 4L);
    EXPECT_EQ(row.get("b"_cs), 5.5);
    EXPECT_EQ(row.get("c"_cs), custom_object(6));

    // assign to the underlying objects of the view;
    a = 5;
    b = 6.5;
    c = custom_object(7);

    row = row_view;
    EXPECT_EQ(row, row_view);
    EXPECT_EQ(row.get("a"_cs), 5L);
    EXPECT_EQ(row.get("b"_cs), 6.5);
    EXPECT_EQ(row.get("c"_cs), custom_object(7));
}

TEST(row, structured_binding) {
    using R = py::row<py::C<std::int64_t>("a"_cs),
                      py::C<double>("b"_cs),
                      py::C<custom_object>("c"_cs)>;

    std::int64_t a = 1;
    double b = 2.5;
    custom_object c(3);
    R row(a, b, c);

    auto [bound_a, bound_b, bound_c] = row;
    EXPECT_EQ(bound_a, a);
    EXPECT_EQ(bound_b, b);
    EXPECT_EQ(bound_c, c);

    auto& [ref_a, ref_b, ref_c] = row;
    EXPECT_EQ(ref_a, a);
    EXPECT_EQ(ref_b, b);
    EXPECT_EQ(ref_c, c);

    // these references are into the row, not the original variables
    ref_a = 2;
    ref_b = 3.5;
    ref_c = custom_object(4);

    EXPECT_EQ(row.get("a"_cs), 2L);
    EXPECT_EQ(row.get("b"_cs), 3.5);
    EXPECT_EQ(row.get("c"_cs), custom_object(4));

    // the original values are unchanged
    EXPECT_EQ(a, 1);
    EXPECT_EQ(b, 2.5);
    EXPECT_EQ(c, custom_object(3));
}

TEST(row, cat) {
    using A =
        py::row<py::C<std::int64_t>("a_first"_cs), py::C<std::int32_t>("a_second"_cs)>;
    using B = py::row<py::C<double>("b_first"_cs), py::C<float>("b_second"_cs)>;
    using C = py::row<py::C<std::string_view>("c_first"_cs),
                      py::C<std::string>("c_second"_cs),
                      py::C<std::string_view>("c_third"_cs),
                      py::C<std::string_view>("c_fourth"_cs)>;

    A a(1, 2);
    B b(3.5, 4.5);
    C c("l", "m", "a", "o");

    auto actual_first_cat = py::row_cat(a, b);

    using first_cat_type = py::row<py::C<std::int64_t>("a_first"_cs),
                                   py::C<std::int32_t>("a_second"_cs),
                                   py::C<double>("b_first"_cs),
                                   py::C<float>("b_second"_cs)>;
    EXPECT_TRUE((std::is_same_v<decltype(actual_first_cat), first_cat_type>) );

    first_cat_type expected_first_cat(1, 2, 3.5, 4.5);
    EXPECT_EQ(actual_first_cat, expected_first_cat);

    auto actual_second_cat = py::row_cat(a, b, c);

    using second_cat_type = py::row<py::C<std::int64_t>("a_first"_cs),
                                    py::C<std::int32_t>("a_second"_cs),
                                    py::C<double>("b_first"_cs),
                                    py::C<float>("b_second"_cs),
                                    py::C<std::string_view>("c_first"_cs),
                                    py::C<std::string>("c_second"_cs),
                                    py::C<std::string_view>("c_third"_cs),
                                    py::C<std::string_view>("c_fourth"_cs)>;
    EXPECT_TRUE((std::is_same_v<decltype(actual_second_cat), second_cat_type>) );

    second_cat_type expected_second_cat(1, 2, 3.5, 4.5, "l", "m", "a", "o");
    EXPECT_EQ(actual_second_cat, expected_second_cat);
}

TEST(table, emplace_back) {
    using T = py::table<py::C<std::int64_t>("a"_cs),
                        py::C<double>("b"_cs),
                        py::C<custom_object>("c"_cs)>;
    T table;
    ASSERT_EQ(table.size(), 0ul);

    // insert a tuple
    auto expected_row_0 = std::make_tuple(1, 2.5, custom_object(3));
    table.emplace_back(expected_row_0);
    ASSERT_EQ(table.size(), 1ul);

    auto test_row_0 = [&] {
        auto row = table[0];
        EXPECT_EQ(row, row);
        EXPECT_EQ(row, expected_row_0);
        EXPECT_EQ(row, row.copy());
    };

    test_row_0();

    // insert a row
    auto expected_row_1 = T::row_type(2, 3.5, custom_object(4));
    table.emplace_back(expected_row_1);
    ASSERT_EQ(table.size(), 2ul);

    auto test_row_1 = [&] {
        auto row = table[1];
        EXPECT_EQ(row, row);
        EXPECT_EQ(row, expected_row_1);
        EXPECT_EQ(row, row.copy());
    };

    test_row_0();
    test_row_1();

    // insert a row_view
    std::int64_t row_2_a = 3;
    double row_2_b = 4.5;
    custom_object row_2_c{5};
    auto expected_row_2 = T::row_view_type(&row_2_a, &row_2_b, &row_2_c);
    table.emplace_back(expected_row_2);
    ASSERT_EQ(table.size(), 3ul);

    auto test_row_2 = [&] {
        auto row = table[2];
        EXPECT_EQ(row, row);
        EXPECT_EQ(row, expected_row_2);
        EXPECT_EQ(row, row.copy());
    };

    test_row_0();
    test_row_1();
    test_row_2();

    // use forwarding constructor
    // note: 6 is being directly forwarded to `custom_object(int)`
    table.emplace_back(4, 5.5, 6);
    std::int64_t row_3_a = 4;
    double row_3_b = 5.5;
    custom_object row_3_c{6};
    auto expected_row_3 = T::row_view_type(&row_3_a, &row_3_b, &row_3_c);

    auto test_row_3 = [&] {
        auto row = table[3];
        EXPECT_EQ(row, row);
        EXPECT_EQ(row, expected_row_3);
        EXPECT_EQ(row, row.copy());
    };

    test_row_0();
    test_row_1();
    test_row_2();
    test_row_3();
}

template<typename T>
void test_row_iter(T& table) {
    std::int64_t expected_a = 0;
    double expected_b = 1.5;
    custom_object expected_c(2);

    for (auto row : table) {
        auto& a = row.get("a"_cs);
        auto& b = row.get("b"_cs);
        auto& c = row.get("c"_cs);

        if constexpr (std::is_const_v<T>) {
            testing::StaticAssertTypeEq<decltype(a), const std::int64_t&>();
            testing::StaticAssertTypeEq<decltype(b), const double&>();
            testing::StaticAssertTypeEq<decltype(c), const custom_object&>();
        }
        else {
            testing::StaticAssertTypeEq<decltype(a), std::int64_t&>();
            testing::StaticAssertTypeEq<decltype(b), double&>();
            testing::StaticAssertTypeEq<decltype(c), custom_object&>();
        }

        EXPECT_EQ(a, ++expected_a);
        EXPECT_EQ(b, ++expected_b);
        EXPECT_EQ(c, ++expected_c);
    }
}

TEST(table, row_iter) {
    using T = py::table<py::C<std::int64_t>("a"_cs),
                        py::C<double>("b"_cs),
                        py::C<custom_object>("c"_cs)>;

    T table;

    std::int64_t a = 0;
    double b = 1.5;
    custom_object c(2);
    for (std::size_t ix = 0; ix < 64; ++ix) {
        table.emplace_back(std::make_tuple(++a, ++b, ++c));
    }
    ASSERT_EQ(table.size(), 64ul);

    test_row_iter(table);
    test_row_iter<const T>(table);
}

TEST(table, insert) {
    using T = py::table<py::C<std::int64_t>("a"_cs),
                        py::C<double>("b"_cs),
                        py::C<custom_object>("c"_cs)>;

    T table;

    T intermediate;
    std::int64_t a = 0;
    double b = 1.5;
    custom_object c(2);
    for (std::size_t ix = 0; ix < 64; ++ix) {
        intermediate.emplace_back(std::make_tuple(++a, ++b, ++c));
    }
    ASSERT_EQ(intermediate.size(), 64ul);

    table.insert(table.end(), intermediate.begin(), intermediate.end());
    ASSERT_EQ(table.size(), intermediate.size());

    for (std::size_t ix = 0; ix < 64; ++ix) {
        EXPECT_EQ(table[ix], intermediate[ix]);
    }

    // insert into the middle a a subset of the intermedate
    table.insert(table.begin() + 32, intermediate.begin(), intermediate.begin() + 8);
    ASSERT_EQ(table.size(), intermediate.size() + 8);

    for (std::size_t ix = 0; ix < 32; ++ix) {
        // the first 32 elements are the same
        EXPECT_EQ(table[ix], intermediate[ix]);
    }
    for (std::size_t ix = 0; ix < 8; ++ix) {
        // the next 8 elements are repeats of the first 8 from `intermediate`
        EXPECT_EQ(table[ix + 32], intermediate[ix]);
    }
    for (std::size_t ix = 32; ix < 64; ++ix) {
        // the last 32 elements are the last 32 elements of `intermediate`
        EXPECT_EQ(table[ix + 8], intermediate[ix]);
    }

    class row_iter {
    private:
        const T& m_table;
        std::size_t m_ix;

    public:
        row_iter(const T& table, std::size_t ix) : m_table(table), m_ix(ix) {}

        void operator++() {
            ++m_ix;
        }

        std::int64_t operator-(const row_iter& other) const {
            return m_ix - other.m_ix;
        }

        bool operator!=(const row_iter& other) const {
            return m_ix != other.m_ix;
        }

        T::row_type operator*() const {
            return m_table[m_ix];
        }
    };

    row_iter begin{intermediate, 0};
    row_iter end{intermediate, 8};

    // insert from an iterator pair that doesn't come from a like-shaped table
    table.insert(table.end(), begin, end);
    ASSERT_EQ(table.size(), intermediate.size() + 8 + 8);

    for (std::size_t ix = 0; ix < 32; ++ix) {
        // the first 32 elements are the same
        EXPECT_EQ(table[ix], intermediate[ix]);
    }
    for (std::size_t ix = 0; ix < 8; ++ix) {
        // the next 8 elements are repeats of the first 8 from `intermediate`
        EXPECT_EQ(table[ix + 32], intermediate[ix]);
    }
    for (std::size_t ix = 32; ix < 64; ++ix) {
        // the next 32 elements are the last 32 elements of `intermediate`
        EXPECT_EQ(table[ix + 8], intermediate[ix]);
    }
    for (std::size_t ix = 0; ix < 8; ++ix) {
        // the last 8 elements are a repeat of the first 8 elements of `intermediate`
        EXPECT_EQ(table[ix + 64 + 8], intermediate[ix]);
    }
}

TEST(table, reserve) {
    using T = py::table<py::C<std::int64_t>("a"_cs),
                        py::C<double>("b"_cs),
                        py::C<custom_object>("c"_cs)>;

    T table;
    EXPECT_EQ(table.capacity(), 0ul);

    table.reserve(5);
    EXPECT_EQ(table.capacity(), 5ul);

    table.reserve(10);
    EXPECT_EQ(table.capacity(), 10ul);

    // capacity doesn't go down if reserving a smaller size
    table.reserve(5);
    EXPECT_EQ(table.capacity(), 10ul);
}

TEST(table_view, relabel) {
    using Table = py::table<py::C<std::int64_t>("a"_cs),
                            py::C<double>("b"_cs),
                            py::C<custom_object>("c"_cs)>;
    using View = typename Table::view_type;

    Table table;

    std::int64_t a = 0;
    double b = 1.5;
    custom_object c(2);
    for (std::size_t ix = 0; ix < 64; ++ix) {
        table.emplace_back(std::make_tuple(++a, ++b, ++c));
    }

    View view(table);

    auto relabeled = view.relabel(std::make_pair("a"_cs, "a-new"_cs),
                                  std::make_pair("c"_cs, "c-new"_cs));

    ASSERT_EQ(relabeled.size(), view.size());
    ASSERT_EQ(relabeled.size(), 64ul);
    for (std::size_t ix = 0; ix < relabeled.size(); ++ix) {
        auto base_row = view[ix];
        auto relabeled_row = relabeled[ix];

        EXPECT_EQ(&relabeled_row.get("a-new"_cs), &base_row.get("a"_cs));
        EXPECT_EQ(&relabeled_row.get("b"_cs), &base_row.get("b"_cs));
        EXPECT_EQ(&relabeled_row.get("c-new"_cs), &base_row.get("c"_cs));
    }
}
}  // namespace test_table
