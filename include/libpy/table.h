#pragma once

#include <tuple>
#include <utility>
#include <vector>

#include "libpy/array_view.h"
#include "libpy/char_sequence.h"
#include "libpy/from_object.h"
#include "libpy/itertools.h"
#include "libpy/meta.h"
#include "libpy/numpy_utils.h"
#include "libpy/utils.h"

namespace py {
namespace detail {
template<std::size_t ix, typename Needle, typename... Haystack>
struct search;

template<std::size_t ix, typename Needle, typename... Tail>
struct search<ix, Needle, Needle, Tail...> {
    constexpr static std::size_t value = ix;
};

template<std::size_t ix, typename Needle, typename Head, typename... Tail>
struct search<ix, Needle, Head, Tail...> {
    constexpr static std::size_t value = search<ix + 1, Needle, Tail...>::value;
};

/** A container to hold the key and value types for a column.

    @tparam Key `std::integer_sequence` of chars containing the column name.
    @tparam Value The scalar type of the column.
 */
template<typename Key, typename Value>
struct column {
    using key = Key;
    using value = Value;

    using const_column = column<Key, std::add_const_t<Value>>;
    using remove_const_column = column<Key, std::remove_const_t<Value>>;
};

template<typename T>
T column_singleton;

template<auto p, typename C = typename std::remove_pointer_t<decltype(p)>::const_column>
constexpr C* const_column = &column_singleton<C>;

template<auto p,
         typename C = typename std::remove_pointer_t<decltype(p)>::remove_const_column>
constexpr C* remove_const_column = &column_singleton<C>;
}  // namespace detail

/** Create a specification for a column to pass to `table` or `table_view`.

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

/** A helper for unwrapping the result of `C` to get the underlying column type.
 */
template<auto p>
using unwrap_column = std::remove_pointer_t<decltype(p)>;

namespace detail {
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
        return ob.get(
            std::get<ix>(std::make_tuple(typename unwrap_column<columns>::key{}...)));
    }
};
}  // namespace detail

template<auto... columns>
class row {
protected:
    using tuple_type = std::tuple<typename unwrap_column<columns>::value...>;

    template<std::size_t, typename>
    friend struct std::tuple_element;

    template<typename>
    friend struct std::tuple_size;

    template<std::size_t... ix, typename O>
    void assign(std::index_sequence<ix...>, const O& values) {
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
        return std::get<detail::search<0,
                                       ColumnName,
                                       typename unwrap_column<columns>::key...>::value>(
            m_data);
    }

    /** Retrieve a value by name.

        @param ColumnName `std::integer_sequence` of chars containing the column name.
        @return A reference to the view over the column.
     */
    template<typename ColumnName>
    auto& get(ColumnName) {
        return std::get<detail::search<0,
                                       ColumnName,
                                       typename unwrap_column<columns>::key...>::value>(
            m_data);
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
        a.get(typename unwrap_column<first_columns>::key{})...,
        b.get(typename unwrap_column<second_columns>::key{})...);
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
namespace detail {}  // namespace detail

template<auto... columns>
class row_view {
protected:
    using tuple_type = std::tuple<typename unwrap_column<columns>::value*...>;

    template<std::size_t, typename>
    friend struct std::tuple_element;

    template<typename>
    friend struct std::tuple_size;

    template<std::size_t... ix, typename O>
    void assign(std::index_sequence<ix...>, const O& values) {
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

public:
    row_view(typename unwrap_column<columns>::value*... cs) : m_data(cs...) {}
    row_view(const row_view&) = default;
    row_view& operator=(const row_view&) = default;

    template<typename O>
    row_view& operator=(const O& values) {
        assign(std::make_index_sequence<sizeof...(columns)>{}, values);
        return *this;
    }

    /** Copy a view into an owning row.
     */
    row<detail::remove_const_column<columns>...> copy() const {
        return std::apply(
            [](auto... data) {
                return row<detail::remove_const_column<columns>...>(*data...);
            },
            m_data);
    }

    /** Retrieve a value by index.

        @tparam The index to look up.
        @return A reference to the value.
    */
    template<std::size_t ix>
    constexpr const auto& get() const {
        return *std::get<ix>(m_data);
    }

    /** Retrieve a value by index.

        @tparam The index to look up.
        @return A reference to the value.
    */
    template<std::size_t ix>
    auto& get() {
        return *std::get<ix>(m_data);
    }

    /** Retrieve a column by name.

        @param ColumnName `std::integer_sequence` of chars containing the column name.
        @return A reference to the view over the column.
     */
    template<typename ColumnName>
    constexpr const auto& get(ColumnName) const {
        return *std::get<detail::search<0,
                                        ColumnName,
                                        typename unwrap_column<columns>::key...>::value>(
            m_data);
    }

    /** Retrieve a column by name.

        @param ColumnName `std::integer_sequence` of chars containing the column name.
        @return A reference to the view over the column.
     */
    template<typename ColumnName>
    constexpr auto& get(ColumnName) {
        return *std::get<detail::search<0,
                                        ColumnName,
                                        typename unwrap_column<columns>::key...>::value>(
            m_data);
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
}

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
}

namespace py {
namespace detail::table_iter {
template<auto... columns>
class rows {
private:
    std::tuple<py::array_view<typename unwrap_column<columns>::value>...> m_columns;

    template<auto... inner_columns>
    class generic_iterator {
    private:
        std::tuple<py::array_view<typename unwrap_column<inner_columns>::value>...>
            m_columns;
        std::size_t m_ix;

    protected:
        friend class rows;

        generic_iterator(
            const std::tuple<
                py::array_view<typename unwrap_column<inner_columns>::value>...>& cs,
            std::size_t ix)
            : m_columns(cs), m_ix(ix) {}

    public:
        using difference_type = std::size_t;
        using value_type = row_view<inner_columns...>;
        using const_value_type = row_view<detail::const_column<inner_columns>...>;
        using iterator_category = std::random_access_iterator_tag;

        value_type operator*() {
            return std::apply([this](auto&... cs) { return value_type{&cs[m_ix]...}; },
                              m_columns);
        }

        const_value_type operator*() const {
            return std::apply(
                [this](const auto&... cs) { return const_value_type{cs[m_ix]...}; },
                m_columns);
        }

        value_type operator[](difference_type ix) {
            return *(*this + ix);
        }

        const_value_type operator[](difference_type ix) const {
            return *(*this + ix);
        }

        generic_iterator& operator++() {
            m_ix += 1;
            return *this;
        }

        generic_iterator operator++(int) {
            generic_iterator out = *this;
            m_ix += 1;
            return out;
        }

        generic_iterator& operator+=(difference_type n) {
            m_ix += n;
            return *this;
        }

        generic_iterator operator+(difference_type n) const {
            return generic_iterator(m_columns, m_ix + n);
        }

        generic_iterator& operator--() {
            m_ix -= 1;
            return *this;
        }

        generic_iterator operator--(int) {
            generic_iterator out = *this;
            m_ix -= 1;
            return out;
        }

        generic_iterator& operator-=(difference_type n) {
            m_ix -= n;
            return *this;
        }

        difference_type operator-(const generic_iterator& other) const {
            return m_ix - other.m_ix;
        }

        bool operator!=(const generic_iterator& other) const {
            return !(m_ix == other.m_ix && m_columns == other.m_columns);
        }

        bool operator==(const generic_iterator& other) const {
            return m_ix == other.m_ix && m_columns == other.m_columns;
        }

        bool operator<(const generic_iterator& other) const {
            return m_ix < other.m_ix;
        }

        bool operator<=(const generic_iterator& other) const {
            return m_ix <= other.m_ix;
        }

        bool operator>(const generic_iterator& other) const {
            return m_ix > other.m_ix;
        }

        bool operator>=(const generic_iterator& other) const {
            return m_ix >= other.m_ix;
        }
    };

public:
    using iterator = generic_iterator<columns...>;
    using const_iterator = generic_iterator<const_column<columns>...>;

    rows(const std::tuple<py::array_view<typename unwrap_column<columns>::value>...>& cs)
        : m_columns(cs) {}

    std::size_t size() const {
        static_assert(sizeof...(columns) > 0, "a table with no columns has no size");
        return std::get<0>(m_columns).size();
    }

    iterator begin() {
        return iterator(m_columns, 0);
    }

    iterator end() {
        return iterator(m_columns, size());
    }

    const_iterator begin() const {
        return const_iterator(m_columns, 0);
    }

    const_iterator end() const {
        return const_iterator(m_columns, size());
    }

    auto operator[](std::size_t ix) {
        return begin()[ix];
    }

    auto operator[](std::size_t ix) const {
        return begin()[ix];
    }
};
}  // namespace detail

template<auto... columns>
class table_view;

/** A collection of named `std::vector` objects.

    @tparam columns A set of column specifications created by `py::C`.
 */
template<auto... columns>
class table {
private:
    std::tuple<std::vector<typename unwrap_column<columns>::value>...> m_columns;

    template<std::size_t... ix, typename... Row>
    void emplace_back(std::index_sequence<ix...>, Row&&... row) {
        (std::get<ix>(m_columns).emplace_back(std::forward<Row>(row)), ...);
    }

    /** Retrieve a column by name.

        Returns an ndarray_view so that the column length cannot be changed, but the
        values may be mutated.

        @param ColumnName `std::integer_sequence` of chars containing the column name.
        @return A reference to the vector owning the column.
     */
    template<typename ColumnName>
    constexpr auto& get_mutable(ColumnName) {
        return std::get<detail::search<0,
                                       ColumnName,
                                       typename unwrap_column<columns>::key...>::value>(
            m_columns);
    }

    template<std::size_t ix>
    auto move_to_objects() {
        auto text = std::get<ix>(this->column_names());
        auto column_name = py::to_object(
            *reinterpret_cast<std::array<char, text.size() - 1>*>(text.data()));
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
    void move_into_dict(std::index_sequence<ix...>, py::scoped_ref<PyObject>& out) {
        std::array<std::tuple<py::scoped_ref<PyObject>, py::scoped_ref<PyObject>>,
                   sizeof...(columns)>
            obs = {this->move_to_objects<ix>()...};

        for (auto& [key, value] : obs) {
            if (PyDict_SetItem(out.get(), key.get(), value.get())) {
                throw py::exception();
            }
        }
    }

public:
    using row_type = row<columns...>;
    using row_view_type = row_view<columns...>;

    table() = default;

    /** Create an owning copy of a compatible table view. The tables will be aligned by
        column name.

        @param cpfrom The view to copy from.
     */
    template<auto... other_columns>
    explicit table(const table_view<other_columns...>& cpfrom)
        : m_columns({cpfrom.get(typename unwrap_column<columns>::key{}).begin(),
                     cpfrom.get(typename unwrap_column<columns>::key{}).end()}...) {
        static_assert(sizeof...(columns) == sizeof...(other_columns),
                      "input columns do not match");
    }

    /** The number of rows in the table.
     */
    std::size_t size() const {
        static_assert(sizeof...(columns) > 0, "a table with no columns has no size");
        return std::get<0>(m_columns).size();
    }

    /** A constexpr sequence of column names as null terminated `std::array<char>`.
     */
    constexpr static auto column_names() {
        return std::make_tuple(
            py::cs::to_array(typename unwrap_column<columns>::key{})...);
    }

    /** Retrieve a column by name.

        @param ColumnName `std::integer_sequence` of chars containing the column name.
        @return A reference to the column.
     */
    template<typename ColumnName>
    constexpr const auto& get(ColumnName) const {
        return std::get<detail::search<0,
                                       ColumnName,
                                       typename unwrap_column<columns>::key...>::value>(
            m_columns);
    }

    /** Retrieve a column by name.

        Returns an ndarray_view so that the column length cannot be changed, but the
        values may be mutated.

        @param ColumnName `std::integer_sequence` of chars containing the column name.
        @return A reference to the view over the column.
     */
    template<typename ColumnName>
    constexpr auto& get(ColumnName) {
        auto& column = std::get<
            detail::search<0, ColumnName, typename unwrap_column<columns>::key...>::
                value>(m_columns);
        return py::array_view<
            typename std::remove_reference_t<decltype(column)>::value_type>(column);
    }

    /** Create a row-wise view over the table.

        @return rows view
     */
    auto rows() {
        return detail::table_iter::rows<columns...>(std::apply(
            [](auto&... cs) {
                return std::make_tuple(
                    py::array_view<
                        typename std::remove_reference_t<decltype(cs)>::value_type>(
                        cs)...);
            },
            m_columns));
    }

    /** Create a row-wise view over the table.

        @return rows view
     */
    auto rows() const {
        return detail::table_iter::rows<detail::const_column<columns>...>(std::apply(
            [](const auto&... cs) {
                return std::make_tuple(
                    py::array_view<
                        typename std::remove_reference_t<decltype(cs)>::value_type>(cs)
                        .freeze()...);
            },
            m_columns));
    }

    /** Append a row of data to the table.

        @param row A tuple of scalar elements to add for each column. The parameter order
                   matches the column order in the type definition.
     */
    template<typename... Row>
    void emplace_back(const std::tuple<Row...>& row) {
        std::apply(
            [this](const auto&... row) {
                this->emplace_back(std::make_index_sequence<sizeof...(columns)>{},
                                   row...);
            },
            row);
    }

    /** Append a row of data to the table.

        @param row A tuple of scalar elements to add for each column. The parameter order
                   matches the column order in the type definition.
     */
    template<typename... Row>
    void emplace_back(std::tuple<Row...>&& row) {
        std::apply(
            [this](auto&&... row) {
                this->emplace_back(std::make_index_sequence<sizeof...(columns)>{},
                                   std::move(row)...);
            },
            row);
    }

    /** Append a row of data to the table.

        @param row A row_view of scalar elements to add for each column. The row will be
                   aligned by column name.
     */
    template<auto... other_columns>
    void emplace_back(const py::row_view<other_columns...>& row) {
        static_assert(sizeof...(columns) == sizeof...(other_columns),
                      "input columns do not match");
        (get_mutable(typename unwrap_column<columns>::key{})
             .emplace_back(row.get(typename unwrap_column<columns>::key{})),
         ...);
    }

    /** Append a row of data to the table.

        @param row A row of scalar elements to add for each column. The row will be
                   aligned by column name.
     */
    template<auto... other_columns>
    void emplace_back(const py::row<other_columns...>& row) {
        static_assert(sizeof...(columns) == sizeof...(other_columns),
                      "input columns do not match");
        (get_mutable(typename unwrap_column<columns>::key{})
             .emplace_back(row.get(typename unwrap_column<columns>::key{})),
         ...);
    }

    /** Move the structure into a Python dict of numpy arrays.

        @return A Python dict of numpy arrays.
     */
    py::scoped_ref<PyObject> to_python_dict() && {
        auto out = py::scoped_ref(PyDict_New());
        if (!out) {
            return nullptr;
        }
        try {
            move_into_dict(std::make_index_sequence<sizeof...(columns)>{}, out);
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
    std::tuple<py::array_view<typename unwrap_column<columns>::value>...> m_columns;

public:
    using row_type = row<columns...>;
    using row_view_type = row_view<columns...>;

    // sentinel for slice
    static constexpr std::size_t npos = -1;

    template<typename ColumnName>
    using column_type = std::tuple_element_t<
        detail::search<0, ColumnName, typename unwrap_column<columns>::key...>::value,
        std::tuple<typename unwrap_column<columns>::value...>>;

    /** Construct a table view over a compatible table. The tables will be aligned by
        column name.

        @param table The table to view.
     */
    template<auto... other_columns>
    table_view(table<other_columns...>& table)
        : m_columns(table.get(typename unwrap_column<columns>::key{})...) {
        static_assert(sizeof...(columns) == sizeof...(other_columns),
                      "input columns do not match");
    }

    /** Create a table view from constituent column views.

        @param cs The columns of the table.
     */
    template<typename... Columns>
    table_view(const Columns&... cs) : m_columns(cs...) {
        if (!utils::all_equal(cs.size()...)) {
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

    /** A constexpr sequence of column names as null terminated `std::array<char>`.
     */
    constexpr static auto column_names() {
        return std::make_tuple(
            py::cs::to_array(typename unwrap_column<columns>::key{})...);
    }

    /** Retrieve a column by name.

        @param ColumnName `std::integer_sequence` of chars containing the column name.
        @return A reference to the view over the column.
     */
    template<typename ColumnName>
    constexpr const auto& get(ColumnName) const {
        return std::get<detail::search<0,
                                       ColumnName,
                                       typename unwrap_column<columns>::key...>::value>(
            m_columns);
    }

    /** Retrieve a column by name.

        @param ColumnName `std::integer_sequence` of chars containing the column name.
        @return A reference to the view over the column.
     */
    template<typename ColumnName>
    auto& get(ColumnName) {
        return std::get<detail::search<0,
                                       ColumnName,
                                       typename unwrap_column<columns>::key...>::value>(
            m_columns);
    }

    /** Create a row-wise view over the table.

        @return rows view
     */
    auto rows() {
        return detail::table_iter::rows<columns...>(m_columns);
    }

    /** Create a row-wise view over the table.

        @return rows view
     */
    auto rows() const {
        return detail::table_iter::rows<detail::const_column<columns>...>(
            std::apply([](const auto&... cs) { return std::make_tuple(cs.freeze()...); },
                       m_columns));
    }

    /** Create a slice of the table by slicing each column.

        @param start The start index of the slice.
        @param stop The stop index of the slice, exclusive.
        @param step The value to increment each index by.
        @return A view over a subset of the rows.
    */
    table_view slice(std::size_t start, std::size_t stop = npos, std::size_t step = 1) {
        return {get(typename unwrap_column<columns>::key{}).slice(start, stop, step)...};
    }

    /** Create a slice of the table by slicing each column.

        @param start The start index of the slice.
        @param stop The stop index of the slice, exclusive.
        @param step The value to increment each index by.
        @return A view over a subset of the rows.
     */
    table_view<detail::const_column<columns>...>
    slice(std::size_t start, std::size_t stop = npos, std::size_t step = 1) const {
        return {get(typename unwrap_column<columns>::key{}).slice(start, stop, step)...};
    }

    /** Return a subset of the columns.

        @parameters ColumnNames The names of the columns to take.
        @return A view over a subset of the columns.
    */
    template<typename... ColumnNames>
    table_view<C<column_type<ColumnNames>>(ColumnNames{})...> subset(ColumnNames...) {
        return {get(ColumnNames{})...};
    }

    /** Return a subset of the columns.

        @parameters ColumnNames The names of the columns to take.
        @return A view over a subset of the columns.
    */
    template<typename... ColumnNames>
    table_view<C<const column_type<ColumnNames>>(ColumnNames{})...>
    subset(ColumnNames...) const {
        return {get(ColumnNames{})...};
    }

    /** Create a new immutable view over the same memory.

        @return The frozen view.
     */
    table_view<detail::const_column<columns>...> freeze() const {
        return {get(typename unwrap_column<columns>::key{}).freeze()...};
    }
};

namespace dispatch {
template<auto... columns>
struct from_object<py::table_view<columns...>> {
private:
    using type = py::table_view<columns...>;

    template<typename Column>
    static auto get_column(PyObject* t) {
        auto text = py::cs::to_array(typename Column::key{});
        auto column_name = py::to_object(
            *reinterpret_cast<std::array<char, text.size() - 1>*>(text.data()));
        if (!column_name) {
            throw py::exception();
        }
        PyObject* column_ob = PyDict_GetItem(t, column_name.get());
        if (!column_ob) {
            throw py::exception(PyExc_ValueError, "missing column: ", column_name);
        }
        if (PyDict_DelItem(t, column_name.get())) {
            // pop the item to track which columns we used
            throw py::exception();
        }
        return py::from_object<py::array_view<typename Column::value>>(column_ob);
    }

public:
    static type f(PyObject* t) {
        if (!PyDict_Check(t)) {
            throw py::exception(
                PyExc_TypeError,
                "from_object<table_view<...>> input must be a Python dictionary, got: ",
                Py_TYPE(t)->tp_name);
        }

        auto copy = py::scoped_ref(PyDict_Copy(t));
        if (!copy) {
            throw py::exception();
        }

        type out(get_column<unwrap_column<columns>>(copy.get())...);
        if (PyDict_Size(copy.get())) {
            auto keys = py::scoped_ref(PyDict_Keys(copy.get()));
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
