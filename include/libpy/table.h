#pragma once
/* table.h

 This file defines four types:

 - `py::table`
 - `py::table_view`
 - `py::row`
 - `py::row_view`

 All four of these types are variadic templates designed to be templated on sentinel
 values created with `py::C`. Each sentinel encodes a (name, type) pair that describes a
 column (in the case of table/table_view) or a field (in the case of row/row_view). For
 details on how this encoding works, see the docs for `py::detail::column` in
 table_details.h.

 ### `table` and `table_view`

 `table` and `table_view` are data structures that model column-oriented tables, in which
 each column has a name and an associated type. Columns (represented as arrays) can be
 looked up by name using `table.get("column_name"_cs)`.

 The main difference between `table` and `table_view` is that `table` owns its own memory
 and can be resized, whereas `table_view` is a view into memory owned by another object
 (often a collection of numpy arrays created in Python code), and cannot be resized.

 `table` is generally used for constructing new tables in C++.
 `table_view` is generally used for receiving tables from Python.

 ### `row` and `row_view`

 `row` and `row_view` are data structures that represent "rows" of `table` and
 `table_view`.

 The main difference between `row` and `row_view` is that a `row` stores its own values
 (often copies of values from a table) whereas a `row_view` holds **pointers** to values
 owned by another object (often a `table_view`) Assignment through a `row_view`
 transitively assigns through these pointers to the underlying storage.

*/
#include <tuple>
#include <utility>
#include <vector>

#include "libpy/char_sequence.h"
#include "libpy/from_object.h"
#include "libpy/itertools.h"
#include "libpy/meta.h"
#include "libpy/ndarray_view.h"
#include "libpy/numpy_utils.h"
#include "libpy/str_convert.h"
#include "libpy/table_details.h"
#include "libpy/to_object.h"
#include "libpy/util.h"

namespace py {

/** Build a specification for a column to pass to `table` or `table_view`.

    ### Usage

    \code
    using my_table = py::table<py::C<int>("some_name"_cs),
                               py::C<double>("some_other_name"_cs)>;
    my_table t;
    \endcode

    @tparam Value The scalar dtype of the column.
    @param Key A `std::integer_sequence` of chars containing the column name.
               `py::cs::operator""_cs` can make this structure as a UDL.
    @return A sentinel value suitable for passing as a template parameter to `table` or
            `table_view`.
 */
template<typename Value, typename Key>
constexpr detail::column<Key, Value>* C(Key) {
    return &detail::column_singleton<detail::column<Key, Value>>;
}

/** Helper for getting the name associated with a column singleton.
 */
template<auto ColumnSingleton>
using column_name = typename detail::unwrap_column<ColumnSingleton>::key;

/** Helper for getting the data type associated with a column singleton.
 */
template<auto ColumnSingleton>
using column_type = typename detail::unwrap_column<ColumnSingleton>::value;

/** Helper for converting a parameter pack of column singletons into a tuple whose field
 *  types encode the names of the input columns.
 */
template<auto... ColumnSingletons>
using column_names = std::tuple<column_name<ColumnSingletons>...>;

/** Helper for converting a parameter pack of column singletons into a tuple whose field
    types correspond to the types of the input columns.
 */
template<auto... ColumnSingletons>
using column_types = std::tuple<column_type<ColumnSingletons>...>;

/** Convert a column with `T` values to a one with `const T` values.
 */
template<auto p, typename C = typename detail::unwrap_column<p>::const_column>
constexpr C* const_column = &detail::column_singleton<C>;

/** Convert a column with `const T` values to a one with `T` values.
 */
template<auto p, typename C = typename detail::unwrap_column<p>::remove_const_column>
constexpr C* remove_const_column = &detail::column_singleton<C>;

namespace detail {

/** Helper for `row::get` and `row_view::get`.
 */
template<std::size_t ix, typename T, auto... columns>
struct get_helper {
    static const std::tuple_element_t<ix, T>& f(const T& ob) {
        return std::get<ix>(ob);
    }
};

template<std::size_t ix,
         template<auto...>
         typename T,
         auto... other_columns,
         auto... columns>
struct get_helper<ix, T<other_columns...>, columns...> {
    static_assert(sizeof...(columns) == sizeof...(other_columns),
                  "input columns do not match");

    static const auto& f(const T<other_columns...>& ob) {
        return ob.get(std::get<ix>(std::make_tuple(column_name<columns>{}...)));
    }
};

}  // namespace detail

template<auto... columns>
class row {
protected:
    using tuple_type = typename py::column_types<columns...>;
    using keys_type = typename py::column_names<columns...>;

    template<std::size_t, typename>
    friend struct std::tuple_element;

    template<typename>
    friend struct std::tuple_size;

    template<std::size_t... ix, typename O>
    void assign(std::index_sequence<ix...>, const O& values) {
        static_assert(sizeof...(ix) == std::tuple_size_v<O>,
                      "cannot assign row from differently sized tuple-like object");
        ((std::get<ix>(m_data) = detail::get_helper<ix, O, columns...>::f(values)), ...);
    }

    template<typename F, typename Agg, typename O, std::size_t... ix>
    auto cmp(F&& f, Agg&& agg, const O& values, std::index_sequence<ix...>) const {
        return agg(f(get<ix>(), detail::get_helper<ix, O, columns...>::f(values))...);
    }

    template<typename F, typename Agg, typename O>
    auto cmp(F&& f, Agg&& agg, const O& values) const {
        return cmp(std::forward<F>(f),
                   std::forward<Agg>(agg),
                   values,
                   std::make_index_sequence<sizeof...(columns)>{});
    }

    tuple_type m_data;

public:
    template<typename... Ts>
    row(Ts&&... cs) : m_data(std::forward<Ts>(cs)...) {}

    row(const row&) = default;
    row(row&&) = default;
    row& operator=(const row&) = default;
    row& operator=(row&&) = default;

    template<typename O>
    row& operator=(const O& values) {
        assign(std::make_index_sequence<sizeof...(columns)>{}, values);
        return *this;
    }

    operator tuple_type() const {
        return m_data;
    }

    /** Retrieve a value by index.

        @tparam The index to look up.
        @return A reference to the value.
    */
    template<std::size_t ix>
    constexpr const auto& get() const {
        return std::get<ix>(m_data);
    }

    /** Retrieve a value by index.

        @tparam The index to look up.
        @return A reference to the value.
    */
    template<std::size_t ix>
    auto& get() {
        return std::get<ix>(m_data);
    }

    /** Retrieve a value by name.

        @param ColumnName `std::integer_sequence` of chars containing the column name.
        @return A reference to the value.
     */
    template<typename ColumnName>
    constexpr const auto& get(ColumnName) const {
        return std::get<py::meta::search_tuple<ColumnName, keys_type>>(m_data);
    }

    /** Retrieve a value by name.

        @param ColumnName `std::integer_sequence` of chars containing the column name.
        @return A reference to the view over the column.
     */
    template<typename ColumnName>
    auto& get(ColumnName) {
        return std::get<py::meta::search_tuple<ColumnName, keys_type>>(m_data);
    }

    template<typename O>
    bool operator==(const O& other) const {
        return cmp([](const auto& a, const auto& b) { return a == b; },
                   [](const auto&... vs) { return (vs && ...); },
                   other);
    }

    template<typename O>
    bool operator!=(const O& other) const {
        return cmp([](const auto& a, const auto& b) { return a != b; },
                   [](const auto&... vs) { return (vs || ...); },
                   other);
    }
};

inline py::row<> row_cat() {
    return {};
}

template<auto... columns>
row<columns...> row_cat(const row<columns...>& r) {
    return r;
}

template<auto... first_columns, auto... second_columns>
row<first_columns..., second_columns...> row_cat(const row<first_columns...>& a,
                                                 const row<second_columns...>& b) {
    return row<first_columns..., second_columns...>(
        a.get(column_name<first_columns>{})..., b.get(column_name<second_columns>{})...);
}

template<typename A, typename B, typename... Tail>
auto row_cat(const A& a, const B& b, const Tail&... tail) {
    return row_cat(row_cat(a, b), row_cat(tail...));
}
}  // namespace py

namespace std {
template<auto... columns>
struct tuple_size<py::row<columns...>> {
    static constexpr std::size_t value =
        std::tuple_size<typename py::row<columns...>::tuple_type>::value;
};

template<std::size_t ix, auto... columns>
struct tuple_element<ix, py::row<columns...>> {
    using type = tuple_element_t<ix, typename py::row<columns...>::tuple_type>;
};
}  // namespace std

namespace py {

template<auto... columns>
class row_view {
protected:
    // A row_view is a tuple of pointers to values owned by another object.
    using tuple_type = std::tuple<column_type<columns>*...>;
    using keys_type = column_names<columns...>;

    template<std::size_t, typename>
    friend struct std::tuple_element;

    template<typename>
    friend struct std::tuple_size;

    template<std::size_t... ix, typename O>
    void assign(std::index_sequence<ix...>, const O& values) const {
        static_assert(sizeof...(ix) == std::tuple_size_v<O>,
                      "cannot assign row from differently sized tuple-like object");
        ((*std::get<ix>(m_data) = detail::get_helper<ix, O, columns...>::f(values)), ...);
    }

    template<typename F, typename Agg, typename O, std::size_t... ix>
    auto cmp(F&& f, Agg&& agg, const O& values, std::index_sequence<ix...>) const {
        return agg(f(get<ix>(), detail::get_helper<ix, O, columns...>::f(values))...);
    }

    template<typename F, typename Agg, typename O>
    auto cmp(F&& f, Agg&& agg, const O& values) const {
        return cmp(std::forward<F>(f),
                   std::forward<Agg>(agg),
                   values,
                   std::make_index_sequence<sizeof...(columns)>{});
    }

    tuple_type m_data;

    template<typename... Ts>
    auto subset_tuple(std::tuple<Ts...>) const {
        return subset(Ts{}...);
    }

protected:
    template<auto...>
    friend class row_view;

    explicit row_view(const tuple_type& data) : m_data(data) {}

public:
    template<typename ColumnName>
    using get_column_type =
        std::tuple_element_t<py::meta::search_tuple<ColumnName, keys_type>,
                             column_types<columns...>>;

    row_view(column_type<columns>*... cs) : m_data(cs...) {}
    row_view(const row_view&) = default;

    row_view& operator=(const row_view& values) {
        assign(std::make_index_sequence<sizeof...(columns)>{}, values);
        return *this;
    }

    template<typename O>
    row_view& operator=(const O& values) {
        assign(std::make_index_sequence<sizeof...(columns)>{}, values);
        return *this;
    }

    operator std::tuple<column_type<remove_const_column<columns>>...>() const {
        return std::apply([](auto... data) { return std::make_tuple(*data...); }, m_data);
    }

    /** Copy a view into an owning row.
     */
    row<remove_const_column<columns>...> copy() const {
        return std::apply(
            [](auto... data) { return row<remove_const_column<columns>...>(*data...); },
            m_data);
    }

    /** Retrieve a value by index.

        @tparam The index to look up.
        @return A reference to the value.
    */
    template<std::size_t ix>
    constexpr auto& get() const {
        return *std::get<ix>(m_data);
    }

    /** Retrieve a column by name.

        @param ColumnName `std::integer_sequence` of chars containing the column name.
        @return A reference to the view over the column.
     */
    template<typename ColumnName>
    constexpr auto& get(ColumnName) const {
        return *std::get<py::meta::search_tuple<ColumnName, keys_type>>(m_data);
    }

    /** Return a subset of the columns.

        @parameters ColumnNames The names of the columns to take.
        @return A view over a subset of the columns.
    */
    template<typename... ColumnNames>
    row_view<C<get_column_type<ColumnNames>>(ColumnNames{})...> subset(ColumnNames...) {
        return {&get(ColumnNames{})...};
    }

    /** Return a subset of the columns.

        @parameters ColumnNames The names of the columns to take.
        @return A view over a subset of the columns.
    */
    template<typename... ColumnNames>
    row_view<C<get_column_type<ColumnNames>>(ColumnNames{})...>
    subset(ColumnNames...) const {
        return {&get(ColumnNames{})...};
    }

    /** Remove some columns from the row view.
     */
    template<typename... ColumnNames>
    auto drop(ColumnNames...) const {
        return subset_tuple(py::meta::set_diff<keys_type, std::tuple<ColumnNames...>>{});
    }

    /** Relabel columns in a row view.

        @param Pairs of `{old_name, new_name}` to replace in the table. Columns not
               specified will be unchanged.
        @return The relabeled row viewing the same memory.
     */
    template<typename... From, typename... To>
    auto relabel(std::pair<From, To>...) const {
        using Mappings = std::tuple<std::pair<From, To>...>;

        using OutType = row_view<C<get_column_type<column_name<columns>>>(
            detail::relabeled_column_name<column_name<columns>, Mappings>{})...>;

        return OutType(m_data);
    }

    template<typename O>
    bool operator==(const O& other) const {
        return cmp([](const auto& a, const auto& b) { return a == b; },
                   [](const auto&... vs) { return (vs && ...); },
                   other);
    }

    template<typename O>
    bool operator!=(const O& other) const {
        return cmp([](const auto& a, const auto& b) { return a != b; },
                   [](const auto&... vs) { return (vs || ...); },
                   other);
    }
};

template<auto... columns>
void swap(row_view<columns...> a, row_view<columns...> b) {
    auto a_tmp = a.copy();
    a = b;
    b = a_tmp;
}
}  // namespace py

namespace std {
template<auto... columns>
struct tuple_size<py::row_view<columns...>> {
    static constexpr std::size_t value =
        std::tuple_size<typename py::row_view<columns...>::tuple_type>::value;
};

template<std::size_t ix, auto... columns>
struct tuple_element<ix, py::row_view<columns...>> {
    using type = std::remove_pointer_t<
        tuple_element_t<ix, typename py::row_view<columns...>::tuple_type>>;
};
}  // namespace std

namespace py {
template<auto...>
class table;

template<auto...>
class table_view;

namespace detail::table_iter {
template<auto... columns>
class iterator {
protected:
    using tuple_type = std::tuple<py::array_view<column_type<columns>>...>;

public:
    tuple_type m_columns;
    std::intptr_t m_ix;

protected:
    template<auto...>
    friend class rows;

    template<auto...>
    friend class py::table;

    template<auto...>
    friend class py::table_view;

    friend class iterator<remove_const_column<columns>...>;

    iterator(const tuple_type& cs, std::intptr_t ix) : m_columns(cs), m_ix(ix) {}

    const tuple_type& column_tuple() const {
        return m_columns;
    }

    std::intptr_t ix() const {
        return m_ix;
    }

public:
    using difference_type = std::ptrdiff_t;
    using value_type = row<remove_const_column<columns>...>;
    using const_value_type = const value_type;
    using reference = row_view<columns...>;
    using const_reference = row<const_column<columns>...>;
    using pointer = value_type*;
    using iterator_category = std::random_access_iterator_tag;

    // allow casts from `iterator` to `const_iterator` on `table` and `table_view`
    operator iterator<const_column<columns>...>() const {
        return std::apply(
            [this](const auto&... cs) {
                return iterator<const_column<columns>...>{std::make_tuple(cs.freeze()...),
                                                          m_ix};
            },
            m_columns);
    }

    reference operator*() const {
        return std::apply([this](auto&... cs) { return reference{&cs[m_ix]...}; },
                          m_columns);
    }

    reference operator[](difference_type ix) const {
        return *(*this + ix);
    }

    iterator& operator++() {
        m_ix += 1;
        return *this;
    }

    iterator operator++(int) {
        iterator out = *this;
        m_ix += 1;
        return out;
    }

    iterator& operator+=(difference_type n) {
        m_ix += n;
        return *this;
    }

    iterator operator+(difference_type n) const {
        return iterator(m_columns, m_ix + n);
    }

    iterator& operator--() {
        m_ix -= 1;
        return *this;
    }

    iterator operator--(int) {
        iterator out = *this;
        m_ix -= 1;
        return out;
    }

    iterator& operator-=(difference_type n) {
        m_ix -= n;
        return *this;
    }

    difference_type operator-(const iterator& other) const {
        return m_ix - other.m_ix;
    }

    iterator operator-(difference_type n) const {
        return iterator(m_columns, m_ix - n);
    }

    bool operator!=(const iterator& other) const {
        return !(m_ix == other.m_ix && m_columns == other.m_columns);
    }

    bool operator==(const iterator& other) const {
        return m_ix == other.m_ix && m_columns == other.m_columns;
    }

    bool operator<(const iterator& other) const {
        return m_ix < other.m_ix;
    }

    bool operator<=(const iterator& other) const {
        return m_ix <= other.m_ix;
    }

    bool operator>(const iterator& other) const {
        return m_ix > other.m_ix;
    }

    bool operator>=(const iterator& other) const {
        return m_ix >= other.m_ix;
    }
};

template<auto... columns>
class rows {
private:
    using tuple_type = std::tuple<py::array_view<column_type<columns>>...>;
    tuple_type m_columns;

    using iterator = table_iter::iterator<columns...>;

public:
    rows(const tuple_type& cs) : m_columns(cs) {}

    std::size_t size() const {
        static_assert(sizeof...(columns) > 0, "a table with no columns has no size");
        return std::get<0>(m_columns).size();
    }

    /** The number of rows in the table.
     */
    std::ptrdiff_t ssize() const {
        return static_cast<std::ptrdiff_t>(size());
    }

    iterator begin() const {
        return {m_columns, 0};
    }

    iterator end() const {
        return {m_columns, ssize()};
    }

    auto operator[](std::size_t ix) const {
        return begin()[ix];
    }
};
}  // namespace detail::table_iter

/** A collection of named `std::vector` objects.

    @tparam columns A set of column specifications created by `py::C`.
 */
template<auto... columns>
class table {
public:
    using view_type = table_view<columns...>;
    using row_type = row<columns...>;
    using row_view_type = row_view<columns...>;

private:
    using keys_type = column_names<columns...>;
    using tuple_type = std::tuple<std::vector<column_type<columns>>...>;

    tuple_type m_columns;

    /** Retrieve a column by name.

        Returns an ndarray_view so that the column length cannot be changed, but the
        values may be mutated.

        @param ColumnName `std::integer_sequence` of chars containing the column name.
        @return A reference to the vector owning the column.
     */
    template<typename ColumnName>
    constexpr auto& get_mutable(ColumnName) {
        return std::get<py::meta::search_tuple<ColumnName, keys_type>>(m_columns);
    }

    template<std::size_t ix>
    auto move_to_objects(py::str_type key_type) {
        // Convert compile-time column name into Python value.
        auto column_name = py::to_stringlike(std::tuple_element_t<ix, keys_type>{},
                                             key_type);

        if (!column_name) {
            throw py::exception();
        }

        auto array = py::move_to_numpy_array(std::move(std::get<ix>(m_columns)));
        if (!array) {
            throw py::exception();
        }
        return std::make_tuple(column_name, array);
    }

    template<std::size_t... ix>
    void move_into_dict(std::index_sequence<ix...>,
                        const py::owned_ref<>& out,
                        py::str_type key_type) {
        std::array<std::tuple<py::owned_ref<PyObject>, py::owned_ref<>>,
                   sizeof...(columns)>
            obs = {this->move_to_objects<ix>(key_type)...};

        for (auto& [key, value] : obs) {
            if (PyDict_SetItem(out.get(), key.get(), value.get())) {
                throw py::exception();
            }
        }
    }

public:
    table() = default;

    /** Create an owning copy of a compatible table view. The tables will be aligned by
        column name.

        @param cpfrom The view to copy from.
     */
    template<auto... other_columns>
    explicit table(const table_view<other_columns...>& cpfrom)
        : m_columns({cpfrom.get(column_name<columns>{}).begin(),
                     cpfrom.get(column_name<columns>{}).end()}...) {
        static_assert(sizeof...(columns) == sizeof...(other_columns),
                      "input columns do not match");
    }

    /** The number of rows in the table.
     */
    std::size_t size() const {
        static_assert(sizeof...(columns) > 0, "a table with no columns has no size");
        return std::get<0>(m_columns).size();
    }

    /** The number of rows in the table.
     */
    std::ptrdiff_t ssize() const {
        return static_cast<std::ptrdiff_t>(size());
    }

    /** The number of rows this table can store before resizing and invalidating iterators
        and references.
     */
    std::size_t capacity() const {
        static_assert(sizeof...(columns) > 0, "a table with no columns has no capacity");
        // This assumes that all of our vectors will grow the same way. At least with
        // libstdc++ this is true.
        return std::get<0>(m_columns).capacity();
    }

    using iterator = detail::table_iter::iterator<columns...>;
    using const_iterator = detail::table_iter::iterator<const_column<columns>...>;

private:
    /** Create a tuple of views over the columns.

        @return rows view
     */
    auto column_views() {
        return std::apply(
            [](auto&... cs) {
                return std::make_tuple(
                    py::array_view<
                        typename std::remove_reference_t<decltype(cs)>::value_type>(
                        cs)...);
            },
            m_columns);
    }

    /** Create a tuple of views over the columns.

        @return rows view
     */
    auto column_views() const {
        return std::apply(
            [](const auto&... cs) {
                return std::make_tuple(
                    py::array_view<
                        const typename std::remove_reference_t<decltype(cs)>::value_type>(
                        cs)...);
            },
            m_columns);
    }

public:
    iterator begin() {
        return {column_views(), 0};
    }

    const_iterator begin() const {
        return {column_views(), 0};
    }

    iterator end() {
        return {column_views(), ssize()};
    }

    const_iterator end() const {
        return {column_views(), ssize()};
    }

    auto operator[](std::size_t ix) {
        return begin()[ix];
    }

    auto operator[](std::size_t ix) const {
        return begin()[ix];
    }

    /** Retrieve a column by name.

        @param ColumnName `std::integer_sequence` of chars containing the column name.
        @return A reference to the column.
     */
    template<typename ColumnName>
    constexpr const auto& get(ColumnName) const {
        constexpr std::size_t ix = py::meta::search_tuple<ColumnName, keys_type>;
        return std::get<ix>(m_columns);
    }

    /** Retrieve a column by name.

        Returns an ndarray_view so that the column length cannot be changed, but the
        values may be mutated.

        @param ColumnName `std::integer_sequence` of chars containing the column name.
        @return A reference to the view over the column.
     */
    template<typename ColumnName>
    auto get(ColumnName) {
        constexpr std::size_t ix = py::meta::search_tuple<ColumnName, keys_type>;
        auto& column = std::get<ix>(m_columns);
        return py::array_view<
            typename std::remove_reference_t<decltype(column)>::value_type>(column);
    }

    /** Create a row-wise view over the table.

        @return rows view
     */
    [[deprecated("rows() is now redundant, access rows from the table directly")]] auto
    rows() {
        return detail::table_iter::rows<columns...>(column_views());
    }

    /** Create a row-wise view over the table.

        @return rows view
     */
    [[deprecated("rows() is now redundant, access rows from the table directly")]] auto
    rows() const {
        return detail::table_iter::rows<const_column<columns>...>(column_views());
    }

private:
    template<std::size_t... ix>
    void reserve(std::index_sequence<ix...>, std::size_t new_cap) {
        (std::get<ix>(m_columns).reserve(new_cap), ...);
    }

public:
    /** Reserve space for up to `new_cap` rows. If `new_cap` is less than the current
        `size()` of the table, nothing happens.

        `reserve()` does not change the size of the table.

        If `new_cap` is greater than `capacity()`, all iterators, including the
        past-the-end iterator, and all references to the elements are
        invalidated. Otherwise, no iterators or references are invalidated.
     */
    void reserve(std::size_t new_cap) {
        reserve(std::make_index_sequence<sizeof...(columns)>{}, new_cap);
    }

private:
    template<std::size_t... ix>
    void emplace_back(std::index_sequence<ix...>, row_type&& row) {
        (std::get<ix>(m_columns).emplace_back(std::move(row.template get<ix>())), ...);
    }

public:
    /** Append a row of data to the table.

        @param args Arguments to forward to the `row_type` constructor.
     */
    template<typename... Args>
    void emplace_back(Args&&... args) {
        emplace_back(std::make_index_sequence<sizeof...(columns)>{},
                     row_type{std::forward<Args>(args)...});
    }

private:
    /** Optimization for the case where we know the iterator is over a like-shaped table
        or table view.

        In this case, we call `insert` on each of the constituent vectors individually
        instead of `insert`ing one row at a time.
     */
    template<std::size_t... ix>
    iterator insert(std::index_sequence<ix...>,
                    const_iterator pos,
                    const_iterator first,
                    const_iterator last) {
        auto insert_ix = pos - begin();
        (std::get<ix>(m_columns).insert(std::get<ix>(m_columns).begin() + insert_ix,
                                        std::get<ix>(first.column_tuple()).begin() +
                                            first.ix(),
                                        std::get<ix>(first.column_tuple()).begin() +
                                            last.ix()),
         ...);
        return begin() + insert_ix;
    }

    /** Like-shaped table iterators, but non-const versions. Just cast them to const and
        forward to insert. This is needed because we have a template in this overload
        set, otherwise the implicit conversion would kick in.
     */
    template<std::size_t... ix>
    iterator insert(std::index_sequence<ix...> is,
                    const_iterator pos,
                    iterator first,
                    iterator last) {
        return insert(is,
                      pos,
                      static_cast<const_iterator>(first),
                      static_cast<const_iterator>(last));
    }

    /** If we don't know what kind of iterator this is, just accumulate the result into
        an intermediate table and then use the optimized insert.

        Calling `insert()` for scalars in a loop is horrific because it will keep shifting
        the elements to the right of it each time. Insert with iterators just shifts by
        the size of the newly inserted slice once and then copies everything over.

        More work can be done to optimize this case.
     */
    template<std::size_t... ix, typename I>
    iterator insert(std::index_sequence<ix...> is, const_iterator pos, I first, I last) {
        table intermediate;
        intermediate.reserve(last - first);
        for (; first != last; ++first) {
            intermediate.emplace_back(*first);
        }
        return insert(is, pos, intermediate.begin(), intermediate.end());
    }

public:
    /** Inserts elements from range `[first, last)` before `pos`.

        @param pos The iterator to the position before which elements will be inserted.
        @param begin The beginning of the range to insert.
        @param end The end of the range to insert.
     */
    template<typename I>
    iterator insert(const_iterator pos, I first, I last) {
        return insert(std::make_index_sequence<sizeof...(columns)>{}, pos, first, last);
    }

    /** Move the structure into a Python dict of numpy arrays.

        @return A Python dict of numpy arrays.
     */
    py::owned_ref<> to_python_dict(py::str_type key_type = py::str_type::bytes) && {
        py::owned_ref out(PyDict_New());
        if (!out) {
            return nullptr;
        }
        try {
            move_into_dict(std::make_index_sequence<sizeof...(columns)>{}, out, key_type);
        }
        catch (py::exception&) {
            return nullptr;
        }

        return out;
    }
};

/** A collection of named 1D `py::ndarray_view` objects.

    @tparam columns A set of column specifications created by `py::C`.
 */
template<auto... columns>
class table_view {
private:
    using keys_type = column_names<columns...>;
    using tuple_type = std::tuple<py::array_view<column_type<columns>>...>;
    tuple_type m_columns;

    template<typename... Ts>
    auto subset_tuple(std::tuple<Ts...>) {
        return subset(Ts{}...);
    }

    template<typename... Ts>
    auto subset_tuple(std::tuple<Ts...>) const {
        return subset(Ts{}...);
    }

protected:
    template<auto...>
    friend class table_view;

    /** Create a table view from constituent column views. This does not check the length
        of the columns so it should only be used internally.

        @param cs The columns of the table.
     */
    explicit table_view(const tuple_type& cs) : m_columns(cs) {}

public:
    using row_type = row<columns...>;
    using row_view_type = row_view<columns...>;

    // sentinel for slice
    static constexpr std::size_t npos = -1;

    template<typename ColumnName>
    using get_column_type =
        std::tuple_element_t<py::meta::search_tuple<ColumnName, keys_type>,
                             column_types<columns...>>;

    /** Construct a table view over a compatible table. The tables will be
       aligned by column name.

        @param table The table to view.
     */
    template<auto... other_columns>
    table_view(table<other_columns...>& table)
        : m_columns(table.get(column_name<columns>{})...) {
        static_assert(sizeof...(columns) == sizeof...(other_columns),
                      "input columns do not match");
    }

    /** Create a table view from constituent column views.

        @param cs The columns of the table.
     */
    template<typename... Columns>
    table_view(const Columns&... cs) : m_columns(cs...) {
        if (!py::util::all_equal(cs.size()...)) {
            throw std::invalid_argument("columns must be same length");
        }
    }

    table_view(const table_view& cpfrom) = default;
    table_view& operator=(const table_view& cpfrom) = default;

    /** The number of rows in the table.
     */
    std::size_t size() const {
        static_assert(sizeof...(columns) > 0, "a table with no columns has no size");
        return std::get<0>(m_columns).size();
    }

    /** The number of rows in the table.
     */
    std::ptrdiff_t ssize() const {
        return static_cast<std::ptrdiff_t>(size());
    }

    using iterator = detail::table_iter::iterator<columns...>;

    iterator begin() const {
        return {m_columns, 0};
    }

    iterator end() const {
        return {m_columns, ssize()};
    }

    auto operator[](std::size_t ix) const {
        return begin()[ix];
    }

    /** Retrieve a column by name.

        @param ColumnName `std::integer_sequence` of chars containing the column name.
        @return A reference to the view over the column.
     */
    template<typename ColumnName>
    constexpr const auto& get(ColumnName) const {
        constexpr std::size_t ix = py::meta::search_tuple<ColumnName, keys_type>;
        return std::get<ix>(m_columns);
    }

    /** Create a row-wise view over the table.

        @return rows view
     */
    [[deprecated("rows() is now redundant, access rows from the table directly")]] auto
    rows() const {
        return detail::table_iter::rows<columns...>(m_columns);
    }

    /** Create a slice of the table by slicing each column.

        @param start The start index of the slice.
        @param stop The stop index of the slice, exclusive.
        @param step The value to increment each index by.
        @return A view over a subset of the rows.
    */
    table_view
    slice(std::size_t start, std::size_t stop = npos, std::size_t step = 1) const {
        return {get(column_name<columns>{}).slice(start, stop, step)...};
    }

    /** Return a subset of the columns.

        @parameters ColumnNames The names of the columns to take.
        @return A view over a subset of the columns.
    */
    template<typename... ColumnNames>
    table_view<C<get_column_type<ColumnNames>>(ColumnNames{})...>
    subset(ColumnNames...) const {
        return {get(ColumnNames{})...};
    }

    /** Remove some columns from the table view.
     */
    template<typename... ColumnNames>
    auto drop(ColumnNames...) const {
        return subset_tuple(py::meta::set_diff<keys_type, std::tuple<ColumnNames...>>{});
    }

    /** Relabel columns in a table.

        @param Pairs of `{old_name, new_name}` to replace in the table. Columns not
               specified will be unchanged.
        @return The relabeled table viewing the same memory.
     */
    template<typename... From, typename... To>
    auto relabel(std::pair<From, To>...) const {
        using Mappings = std::tuple<std::pair<From, To>...>;

        using OutType = table_view<C<get_column_type<column_name<columns>>>(
            detail::relabeled_column_name<column_name<columns>, Mappings>{})...>;

        return OutType(m_columns);
    }

    /** Create a new immutable view over the same memory.

        @return The frozen view.
     */
    table_view<const_column<columns>...> freeze() const {
        return {get(column_name<columns>{}).freeze()...};
    }
};

namespace dispatch {
template<auto... columns>
struct from_object<py::table_view<columns...>> {
private:
    using type = py::table_view<columns...>;

    template<typename Column>
    static auto pop_column(py::borrowed_ref<> t) {
        auto text = py::cs::to_array(typename Column::key{});
        auto column_name = py::to_object(
            *reinterpret_cast<std::array<char, text.size() - 1>*>(text.data()));
        if (!column_name) {
            throw py::exception();
        }
        PyObject* column_ob = PyDict_GetItem(t.get(), column_name.get());
        if (!column_ob) {
            throw py::exception(PyExc_ValueError, "missing column: ", column_name);
        }
        if (PyDict_DelItem(t.get(), column_name.get())) {
            // pop the item to track which columns we used
            throw py::exception();
        }
        return py::from_object<py::array_view<typename Column::value>>(column_ob);
    }

public:
    static type f(py::borrowed_ref<> t) {
        if (!PyDict_Check(t.get())) {
            throw py::exception(
                PyExc_TypeError,
                "from_object<table_view<...>> input must be a Python dictionary, got: ",
                Py_TYPE(t.get())->tp_name);
        }

        py::owned_ref copy(PyDict_Copy(t.get()));
        if (!copy) {
            throw py::exception();
        }

        type out(pop_column<py::detail::unwrap_column<columns>>(copy)...);
        if (PyDict_Size(copy.get())) {
            py::owned_ref keys(PyDict_Keys(copy.get()));
            if (!keys) {
                throw py::exception();
            }
            throw py::exception(PyExc_ValueError, "extra columns provided: ", keys);
        }
        return out;
    }
};

template<auto... columns>
struct from_object<py::row<columns...>> {
private:
    using type = py::row<columns...>;

    template<typename Column>
    static auto pop_column(py::borrowed_ref<> t) {
        auto text = py::cs::to_array(typename Column::key{});
        auto column_name = py::to_object(
            *reinterpret_cast<std::array<char, text.size() - 1>*>(text.data()));
        if (!column_name) {
            throw py::exception();
        }
        PyObject* column_ob = PyDict_GetItem(t.get(), column_name.get());
        if (!column_ob) {
            throw py::exception(PyExc_ValueError, "missing column: ", column_name);
        }
        if (PyDict_DelItem(t.get(), column_name.get())) {
            // pop the item to track which columns we used
            throw py::exception();
        }
        return py::from_object<typename Column::value>(column_ob);
    }

public:
    static type f(py::borrowed_ref<> t) {
        if (!PyDict_Check(t.get())) {
            throw py::exception(
                PyExc_TypeError,
                "from_object<row<...>> input must be a Python dictionary, got: ",
                Py_TYPE(t.get())->tp_name);
        }

        py::owned_ref copy(PyDict_Copy(t.get()));
        if (!copy) {
            throw py::exception();
        }

        type out(pop_column<py::detail::unwrap_column<columns>>(copy)...);
        if (PyDict_Size(copy.get())) {
            py::owned_ref keys(PyDict_Keys(copy.get()));
            if (!keys) {
                throw py::exception();
            }
            throw py::exception(PyExc_ValueError, "extra columns provided: ", keys);
        }
        return out;
    }
};
}  // namespace dispatch
}  // namespace py
