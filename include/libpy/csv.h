#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_set>
#include <variant>
#include <vector>

#include "libpy/datetime64ns.h"
#include "libpy/exception.h"
#include "libpy/itertools.h"
#include "libpy/numpy_utils.h"
#include "libpy/scoped_ref.h"
#include "libpy/to_object.h"
#include "libpy/valgrind.h"

namespace py::csv {
struct fast_float {};
}  // namespace py::csv

namespace py::dispatch {
template<>
struct new_dtype<py::csv::fast_float> : public new_dtype<double> {};
}  // namespace py::dispatch

namespace py::csv {
constexpr std::size_t min_group_size = 4096;

namespace detail {
template<typename... Ts>
std::string format_string(Ts&&... msg) {
    std::stringstream s;
    (s << ... << msg);
    return s.str();
}

template<typename... Ts>
std::runtime_error formatted_error(Ts&&... msg) {
    return std::runtime_error(format_string(std::forward<Ts>(msg)...));
}

template<typename... Ts>
std::runtime_error
position_formatted_error(std::size_t row, std::size_t col, Ts&&... msg) {
    return formatted_error("line ", row, " column ", col, ": ", std::forward<Ts>(msg)...);
}

/** The state of the row parser, either quoted or not quoted.
 */
enum class quote_state {
    quoted,
    not_quoted,
};

/** Error thrown when processing quoted cells fails.
 */
class quote_error : public std::runtime_error {
public:
    template<typename... Ts>
    quote_error(Ts&&... msg)
        : std::runtime_error(format_string(std::forward<Ts>(msg)...)) {}
};

/** Isolate the current cell in the row. This does not do any escaping so it shouldn't
    be used with strings.

    @param row The row to parse.
    @param offset The offset into the row of the start of the cell.
    @param delim The cell delimiter.
    @return A tuple of the isolated cell, the number of bytes consumed, and a boolean
            indicating that more cells are expected.
 */
std::tuple<std::string_view, std::size_t, bool>
isolate_unquoted_cell(const std::string_view& row, std::size_t offset, char delim) {
    auto subrow = row.substr(offset);
    const void* loc = std::memchr(subrow.data(), delim, subrow.size());
    std::size_t size;
    std::size_t consumed;
    bool more;
    if (loc) {
        size = reinterpret_cast<const char*>(loc) - subrow.data();
        consumed = size + 1;
        more = true;
    }
    else {
        size = consumed = subrow.size();
        more = false;
    }

    return {subrow.substr(0, size), consumed, more};
}

}  // namespace detail

/** A cell parser is an object that represents a row in a CSV.

    Subclasses are required to implement `chomp``.
 */
class cell_parser {
protected:
    char m_delim;

public:
    cell_parser(char delim) : m_delim(delim) {}

    /** Set the line count. This should pre-allocate space for `num_lines` values to be
        parsed.
     */
    virtual void set_num_lines(std::size_t) {}

    virtual ~cell_parser() = default;

    /** "chomp" text from a row and parse the given cell.

        @param row_ix The row number (0-indexed) being parsed.
        @param row The entire row being parsed.
        @param offset The offset into `row` where the cell starts.
        @return A tuple of the number of characters consumed from the row and a boolean
                which is true if we expect there to be more columns to parse in this row.
     */
    virtual std::tuple<std::size_t, bool>
    chomp(std::size_t row_ix, const std::string_view& row, std::size_t offset) = 0;

    /** Move the state of this column to a Python tuple of numpy arrays.

        If this parser doesn't produce values, it just returns a new reference to
        `Py_None`.
     */
    virtual py::scoped_ref<PyObject> move_to_python_tuple() && {
        Py_INCREF(Py_None);
        return py::scoped_ref(Py_None);
    }
};

/** Base class for cell parsers that produce statically typed vectors of values.
 */
template<typename T>
class typed_cell_parser_base : public cell_parser {
protected:
    std::vector<T> m_parsed;
    std::vector<py::py_bool> m_mask;

public:
    using type = T;

    typed_cell_parser_base(char delim) : cell_parser(delim) {}

    const std::vector<T>& parsed() const {
        return m_parsed;
    }

    virtual void set_num_lines(std::size_t nrows) {
        m_parsed.resize(nrows);
        m_mask.resize(nrows);
    }

    py::scoped_ref<PyObject> move_to_python_tuple() && {
        auto values = py::move_to_numpy_array(std::move(m_parsed));
        if (!values) {
            return nullptr;
        }

        auto mask_array = py::move_to_numpy_array(std::move(m_mask));
        if (!mask_array) {
            return nullptr;
        }

        return py::scoped_ref(PyTuple_Pack(2, values.get(), mask_array.get()));
    }

    virtual std::tuple<std::vector<T>, std::vector<py::py_bool>> move_to_cxx_tuple() && {
        return {std::move(m_parsed), std::move(m_mask)};
    }
};

template<typename T>
class typed_cell_parser;

namespace detail {
template<typename F>
std::tuple<std::size_t, bool>
chomp_quoted_string(F&& f, char delim, const std::string_view& row, std::size_t offset) {
    quote_state st = quote_state::not_quoted;
    std::size_t started_quote;

    auto cell = row.substr(offset);

    std::size_t ix;
    for (ix = 0; ix < cell.size(); ++ix) {
        char c = cell[ix];

        if (c == '\\') {
            if (++ix == cell.size()) {
                throw formatted_error("line ",
                                      cell,
                                      ": row ends with escape character: ",
                                      row);
            }

            f(cell[ix]);
            continue;
        }

        switch (st) {
        case quote_state::not_quoted:
            if (c == '"') {
                st = quote_state::quoted;
                started_quote = ix;
            }
            else if (c == delim) {
                return {ix + 1, true};
            }
            else {
                f(cell[ix]);
            }
            break;
        case quote_state::quoted:
            if (c == '"') {
                st = quote_state::not_quoted;
            }
            else {
                f(cell[ix]);
            }
        }
    }

    if (st == quote_state::quoted) {
        started_quote += offset;
        std::string underline(started_quote + 2, ' ');
        underline[0] = '\n';
        underline[underline.size() - 1] = '^';
        throw formatted_error("row ends while quoted, quote begins at index ",
                              started_quote,
                              ":\n",
                              row,
                              underline);
    }

    return {ix, false};
}
}  // namespace detail

template<std::size_t n>
class typed_cell_parser<std::array<char, n>>
    : public typed_cell_parser_base<std::array<char, n>> {
public:
    typed_cell_parser(char delim) : typed_cell_parser_base<std::array<char, n>>(delim) {}

    virtual std::tuple<std::size_t, bool>
    chomp(std::size_t ix, const std::string_view& row, std::size_t offset) {
        auto& cell = this->m_parsed[ix];
        std::size_t cell_ix = 0;
        auto ret = detail::chomp_quoted_string(
            [&](char c) {
                if (cell_ix < cell.size()) {
                    cell[cell_ix++] = c;
                }
            },
            this->m_delim,
            row,
            offset);
        this->m_mask[ix] = cell_ix > 0;
        return ret;
    }
};

namespace detail {
template<auto P>
class double_parser : public typed_cell_parser_base<double> {
public:
    double_parser(char delim) : typed_cell_parser_base<double>(delim) {}

    virtual std::tuple<std::size_t, bool>
    chomp(std::size_t ix, const std::string_view& row, std::size_t offset) {
        const char* first = &row.data()[offset];
        const char* last;
        this->m_parsed[ix] = P(first, &last);

        std::size_t size = last - first;
        if (*last != this->m_delim && size != row.size() - offset) {
            // error if we are not at the end of a cell nor at the end of a row
            std::string_view cell;
            auto end = std::memchr(first, this->m_delim, row.size() - offset);
            if (end) {
                // This branch happens when we have input like `1.5garbage|` where `|` is
                // the delimiter. `last` doesn't point to the delimiter nor the end end of
                // the row. We want to isolate the string `1.5garbage` to report in the
                // error.
                cell = row.substr(offset, reinterpret_cast<const char*>(end) - first);
            }
            else {
                // This branch happens when we have input like `1.5garbage` assuming none
                // of "garbage" is the delimiter. `last` doesn't point to the delimiter
                // nor the end end of the row. We want to isolate the string `1.5garbage`
                // to report in the error.
                cell = row.substr(offset);
            }
            throw detail::formatted_error("invalid digit in double: ", cell);
        }

        this->m_mask[ix] = size > 0;

        bool more = *last == this->m_delim;
        return {size + more, more};
    }
};

double regular_strtod(const char* ptr, const char** last) {
    return std::strtod(ptr, const_cast<char**>(last));
}

double fast_strtod(const char* ptr, const char** last) {
    double result;
    double whole_part = 0;
    double fractional_part = 0;
    double fractional_denom = 1;

    bool negate = *ptr == '-';
    if (negate) {
        ++ptr;
    }
    while (true) {
        char c = *ptr;

        if (c == 'e' || c == 'E') {
            ++ptr;
            goto begin_exp;
        }
        if (c == '.') {
            ++ptr;
            break;
        }

        int value = c - '0';
        if (value < 0 || value > 9) {
            *last = ptr;

            if (negate) {
                whole_part = -whole_part;
            }
            return whole_part;
        }

        whole_part *= 10;
        whole_part += value;
        ++ptr;
    }

    while (true) {
        char c = *ptr;
        if (c == 'e' || c == 'E') {
            ++ptr;
            break;
        }

        int value = c - '0';
        if (value < 0 || value > 9) {
            *last = ptr;

            result = whole_part + fractional_part / fractional_denom;
            if (negate) {
                result = -result;
            }
            return result;
        }

        fractional_denom *= 10;
        fractional_part *= 10;
        fractional_part += value;
        ++ptr;
    }

begin_exp:
    result = whole_part + fractional_part / fractional_denom;
    if (negate) {
        result = -result;
    }

    long exp = 0;
    bool exp_negate = *ptr == '-';
    if (exp_negate || *ptr == '+') {
        ++ptr;
    }
    while (true) {
        int value = *ptr - '0';
        if (value < 0 || value > 9) {
            *last = ptr;

            if (exp_negate) {
                exp = -exp;
            }
            result *= std::pow(10, exp);
            return result;
        }

        exp *= 10;
        exp += value;
        ++ptr;
    }
}

}  // namespace detail

template<>
class typed_cell_parser<double> : public detail::double_parser<detail::regular_strtod> {
public:
    typed_cell_parser(char delim)
        : detail::double_parser<detail::regular_strtod>(delim) {}
};

template<>
class typed_cell_parser<fast_float> : public detail::double_parser<detail::fast_strtod> {
public:
    typed_cell_parser(char delim) : detail::double_parser<detail::fast_strtod>(delim) {}
};

template<>
class typed_cell_parser<py::datetime64ns>
    : public typed_cell_parser_base<py::datetime64ns> {
private:
    /** The number of days in each month for non-leap years.
     */
    constexpr static std::array<uint8_t, 12> days_in_month = {31,   // jan
                                                              28,   // feb
                                                              31,   // mar
                                                              30,   // apr
                                                              31,   // may
                                                              30,   // jun
                                                              31,   // jul
                                                              31,   // aug
                                                              30,   // sep
                                                              31,   // oct
                                                              30,   // nov
                                                              31};  // dec

    static bool is_leapyear(long year) {
        return !(year % 4) && (!(year % 100) || year % 400);
    }

    static std::time_t to_unix_time(int year, int month, int day) {
        std::int64_t seconds = 0;
        if (year < 1970) {
            for (std::uint16_t n = 1969; n >= year; --n) {
                seconds -= (365 + is_leapyear(n)) * 24 * 60 * 60;
            }
        }
        else {
            for (std::uint16_t n = 1970; n < year; ++n) {
                seconds += (365 + is_leapyear(n)) * 24 * 60 * 60;
            }
        }
        for (std::uint8_t n = 0; n < month; ++n) {
            std::uint8_t days = days_in_month[n] + (n == 1 && is_leapyear(year));
            seconds += days * 24 * 60 * 60;
        }
        return seconds + day * 24 * 60 * 60;
    }

    template<std::size_t ndigits>
    static int parse_int(const std::string_view& cs) {
        static_assert(ndigits > 0, "parse_int must be at least 1 char wide");

        int result = 0;
        for (std::size_t n = 0; n < ndigits; ++n) {
            int c = cs[n] - '0';

            if (c < 0 || c > 9) {
                throw detail::formatted_error("invalid digit in int: ",
                                              cs.substr(ndigits));
            }

            result *= 10;
            result += c;
        }
        return result;
    }

    static std::tuple<int, int, int> parse_year_month_day(const std::string_view& raw) {
        if (raw.size() != 10) {
            throw detail::formatted_error("date string is not exactly 10 characters: ",
                                          raw);
        }

        int year = parse_int<4>(raw);

        if (raw[4] != '-') {
            throw detail::formatted_error("expected hyphen at index 4: ", raw);
        }

        // we subtract one to zero index month
        int month = parse_int<2>(raw.substr(5)) - 1;
        if (month < 0 || month > 11) {
            throw detail::formatted_error("month not in range [1, 12]: ", raw);
        }

        if (raw[7] != '-') {
            throw detail::formatted_error("expected hyphen at index 7: ", raw);
        }

        int leap_day = month == 1 && is_leapyear(year);

        // we subtract one to zero index day
        int day = parse_int<2>(raw.substr(8)) - 1;
        int max_day = days_in_month[month] + leap_day - 1;
        if (day < 0 || day > max_day) {
            throw detail::formatted_error(
                "day out of bounds for month (max=", max_day + 1, "): ", raw);
        }

        return {year, month, day};
    }

public:
    typed_cell_parser(char delim) : typed_cell_parser_base<py::datetime64ns>(delim) {}

    virtual std::tuple<std::size_t, bool>
    chomp(std::size_t ix, const std::string_view& row, std::size_t offset) {
        auto [raw, consumed, more] =
            detail::isolate_unquoted_cell(row, offset, this->m_delim);
        if (!raw.size()) {
            return {consumed, more};
        }

        time_t seconds = std::apply(to_unix_time, parse_year_month_day(raw));

        // adapt to a `std::chrono::time_point` to properly handle time unit
        // conversions and time since epoch
        this->m_parsed[ix] = std::chrono::system_clock::from_time_t(seconds);
        this->m_mask[ix] = true;
        return {consumed, more};
    }
};

template<>
class typed_cell_parser<py::py_bool> : public typed_cell_parser_base<py::py_bool> {
public:
    typed_cell_parser(char delim) : typed_cell_parser_base<py::py_bool>(delim) {}

    virtual std::tuple<std::size_t, bool>
    chomp(std::size_t ix, const std::string_view& row, std::size_t offset) {
        auto [raw, consumed, more] =
            detail::isolate_unquoted_cell(row, offset, this->m_delim);
        if (raw.size() == 0) {
            return {consumed, more};
        }
        if (raw.size() != 1) {
            throw detail::formatted_error("bool is not 0 or 1: ", raw);
        }

        bool value;
        if (raw[0] == '0') {
            value = false;
        }
        else if (raw[0] == '1') {
            value = true;
        }
        else {
            throw detail::formatted_error("bool is not 0 or 1: ", raw[0]);
        }

        this->m_parsed[ix] = value;
        this->m_mask[ix] = true;
        return {consumed, more};
    }
};

class header_parser : public cell_parser {
protected:
    std::vector<std::string> m_parsed;

public:
    header_parser(char delim) : cell_parser(delim) {}

    virtual std::tuple<std::size_t, bool>
    chomp(std::size_t, const std::string_view& row, std::size_t offset) {
        this->m_parsed.emplace_back();
        auto& cell = this->m_parsed.back();
        return detail::chomp_quoted_string([&](char c) { cell.push_back(c); },
                                           this->m_delim,
                                           row,
                                           offset);
    }

    const std::vector<std::string>& parsed() const {
        return m_parsed;
    }
};

class skip_parser : public cell_parser {
public:
    skip_parser(char delim) : cell_parser(delim) {}

    virtual std::tuple<std::size_t, bool>
    chomp(std::size_t, const std::string_view& row, std::size_t offset) {
        return detail::chomp_quoted_string([](char) {}, this->m_delim, row, offset);
    }
};

namespace detail {
/** An alias for creating a vector of cell parsers which uses a "template template
    parameter" to be parametric over the type of smart pointer used. The two standard
    options for `ptr_type` are `std::shared_ptr` and `std::unique_ptr`.

    @tparam ptr_type The type of smart pointer to `cell_parser` to use.
 */
template<template<typename...> typename ptr_type>
using parser_types = std::vector<ptr_type<cell_parser>>;

/** Parse a single row and store the values in the vectors.

    @param row The row index.
    @param data The view over the row to parse.
    @param parsers The cell parsers.
 */
template<template<typename...> typename ptr_type>
void parse_row(std::size_t row,
               const std::string_view& data,
               parser_types<ptr_type>& parsers) {

    std::size_t col = 0;
    std::size_t consumed = 0;
    bool more = true;

    for (auto& parser : parsers) {
        if (!more) {
            throw formatted_error("line ",
                                  row + 2,
                                  ": less columns than expected, got ",
                                  col,
                                  " but expected ",
                                  parsers.size());
        }
        try {
            auto [new_consumed, new_more] = parser->chomp(row, data, consumed);
            consumed += new_consumed;
            more = new_more;
        }
        catch (const std::exception& e) {
            throw position_formatted_error(row + 2, col, e.what());
        }

        ++col;
    }

    if (consumed != data.size()) {
        throw formatted_error("line ",
                              row + 2,
                              ": more columns than expected, expected ",
                              parsers.size());
    }
}

template<template<typename...> typename ptr_type>
void parse_lines(std::vector<std::string_view>::iterator begin,
                 std::vector<std::string_view>::iterator end,
                 std::size_t offset,
                 parser_types<ptr_type>& parsers) {
    for (std::size_t ix = offset; begin != end; ++begin, ++ix) {
        parse_row(ix, *begin, parsers);
    }
}

template<template<typename...> typename ptr_type>
void parse_lines_worker(std::mutex* exception_mutex,
                        std::vector<std::exception_ptr>* exceptions,
                        std::vector<std::string_view>::iterator begin,
                        std::vector<std::string_view>::iterator end,
                        std::size_t offset,
                        parser_types<ptr_type>* parsers) {
    try {
        parse_lines(begin, end, offset, *parsers);
    }
    catch (const std::exception&) {
        std::lock_guard<std::mutex> guard(*exception_mutex);
        exceptions->emplace_back(std::current_exception());
    }
}

std::vector<std::string_view> split_into_lines(const std::string_view& data,
                                               const std::string_view& line_ending,
                                               std::size_t skip_rows) {
    // The current position into the input.
    std::string_view::size_type pos = 0;
    // The index of the next newline.
    std::string_view::size_type end;

    std::vector<std::string_view> lines;
    lines.reserve(min_group_size);

    // optionally skip some rows
    for (std::size_t n = 0; n < skip_rows; ++n) {
        if ((end = data.find(line_ending, pos)) == std::string_view::npos) {
            break;
        }
        // advance past line ending
        pos = end + line_ending.size();
    }

    while ((end = data.find(line_ending, pos)) != std::string_view::npos) {
        lines.emplace_back(data.substr(pos, end - pos));

        // advance past line ending
        pos = end + line_ending.size();
    }

    if (pos != data.size()) {
        // add any data after the last newline if there is anything to add
        lines.emplace_back(data.substr(pos));
    }

    return lines;
}

/** Parse a full CSV given the header and the types of the columns.

    @param data The CSV to parse.
    @param parsers The cell parsers.
    @param line_ending The string to split lines on.
 */
template<template<typename...> typename ptr_type>
void parse_from_header(const std::string_view& data,
                       parser_types<ptr_type>& parsers,
                       const std::string_view& line_ending,
                       std::size_t num_threads,
                       std::size_t skip_rows = 0) {
    std::vector<std::string_view> lines = split_into_lines(data, line_ending, skip_rows);

    for (auto& parser : parsers) {
        parser->set_num_lines(lines.size());
    }

    std::size_t group_size;
    if (num_threads <= 1 || (group_size = lines.size() / num_threads) < min_group_size) {
        parse_lines(lines.begin(), lines.end(), 0, parsers);
    }
    else {
        std::mutex exception_mutex;
        std::vector<std::exception_ptr> exceptions;

        std::vector<std::thread> threads;
        std::size_t n;
        for (n = 0; n < num_threads; ++n) {
            std::size_t start = n * group_size;
            threads.emplace_back(
                std::thread(parse_lines_worker<ptr_type>,
                            &exception_mutex,
                            &exceptions,
                            lines.begin() + start,
                            lines.begin() + std::max(start + group_size, lines.size()),
                            start,
                            &parsers));
        }

        for (auto& thread : threads) {
            thread.join();
        }

        for (auto& e : exceptions) {
            std::rethrow_exception(e);
        }
    }
}

template<typename T>
struct dtype_option {
    using type = T;

    static bool matches(PyObject* dtype) {
        auto candidate = py::new_dtype<T>();
        if (!candidate) {
            throw py::exception();
        }

        int result =
            PyObject_RichCompareBool(dtype, static_cast<PyObject*>(candidate), Py_EQ);
        if (result < 0) {
            throw py::exception();
        }
        return result;
    }

    static std::unique_ptr<cell_parser> create_parser(char delimiter) {
        return std::make_unique<typed_cell_parser<T>>(delimiter);
    }
};

/** Create a cell parser with the correct static type given the runtime numpy `dtype`.
    `create_parser<...>::f` expands to the equivalent of:

    ```
    void f(PyObject* dtype, char delimiter, parser_types<std::unique_ptr>& parsers) {
        if (dtype == Ts[0]) {
            parsers.emplace_back(std::make_unique<typed_cell_parser<Ts[0]>>(delimiter));
        }
        else if (dtype = Ts[1]) {
            parsers.emplace_back(std::make_unique<typed_cell_parser<Ts[1]>>(delimiter));
        }
        // ...
        else if (dtype = Ts[-1]) {
            parsers.emplace_back(std::make_unique<typed_cell_parser<Ts[-1]>>(delimiter));
        }
        else {
            throw py::exception(PyExc_TypeError);
        }
    }
    ```
 */
template<typename...>
struct create_parser;

template<typename T, typename... Ts>
struct create_parser<T, Ts...> {
    static std::unique_ptr<cell_parser> f(PyObject* dtype, char delimiter) {
        using option = dtype_option<T>;
        if (!option::matches(dtype)) {
            return create_parser<Ts...>::f(dtype, delimiter);
        }

        return option::create_parser(delimiter);
    }
};

template<>
struct create_parser<> {
    [[noreturn]] static std::unique_ptr<cell_parser> f(PyObject* dtype, char) {
        throw py::exception(PyExc_TypeError, "unknown dtype: ", dtype);
    }
};

template<template<typename...> typename ptr_type, typename GetParser>
std::tuple<std::string_view, parser_types<ptr_type>>
parse_header(const std::string_view& data,
             char delimiter,
             const std::string_view& line_ending,
             GetParser&& get_parser) {
    constexpr bool is_shared = std::is_same_v<ptr_type<void>, std::shared_ptr<void>>;
    constexpr bool is_unique = std::is_same_v<ptr_type<void>, std::unique_ptr<void>>;
    static_assert(is_shared || is_unique,
                  "ptr_type must be std::shared_ptr or std::unique_ptr");

    auto line_end = data.find(line_ending, 0);
    auto line = data.substr(0, line_end);

    header_parser header_parser(delimiter);
    for (auto [consumed, more] = std::make_tuple(0, true); more;) {
        auto [new_consumed, new_more] = header_parser.chomp(0, line, consumed);
        consumed += new_consumed;
        more = new_more;
    }

    std::unordered_set<std::string> column_names;
    parser_types<ptr_type> parsers;
    for (const auto& cell : header_parser.parsed()) {
        if (column_names.count(cell)) {
            throw detail::formatted_error("column name duplicated: ", cell);
        }
        column_names.emplace(cell);

        auto search = get_parser(cell);
        if (!search) {
            if constexpr (is_shared) {
                parsers.emplace_back(std::make_shared<skip_parser>(delimiter));
            }
            else {
                parsers.emplace_back(std::make_unique<skip_parser>(delimiter));
            }
        }
        else {
            parsers.emplace_back(std::move(*search));
        }
    };

    auto start = std::min(line.size() + line_ending.size(), data.size());
    return {data.substr(start), std::move(parsers)};
}

/** Verify that the `dtypes` dict keys are a subset of the actual header in the file.

    This function will throw a C++ exception and raise a Python exception on failure.

    @param dtypes The user-provided dtypes.
    @param header The actual header found in the CSV.
 */
void verify_dtypes_dict(PyObject* dtypes, std::vector<py::scoped_ref<PyObject>>& header) {
    auto expected_keys = py::scoped_ref(PySet_New(dtypes));
    if (!expected_keys) {
        throw py::exception();
    }
    auto actual_keys = py::to_object(header);
    if (!actual_keys) {
        throw py::exception();
    }

    auto actual_keys_set = py::scoped_ref(PySet_New(actual_keys.get()));
    if (!actual_keys_set) {
        throw py::exception();
    }

    auto diff = py::scoped_ref(
        PyNumber_Subtract(expected_keys.get(), actual_keys_set.get()));
    if (!diff) {
        throw py::exception();
    }
    if (PySet_GET_SIZE(diff.get())) {
        auto as_list = py::scoped_ref(PySequence_List(diff.get()));
        if (PyList_Sort(as_list.get()) < 0) {
            throw py::exception();
        }
        throw py::exception(PyExc_ValueError,
                            "dtype keys not present in header: ",
                            as_list,
                            "\nheader: ",
                            actual_keys);
    }
}
}  // namespace detail

/** CSV parsing function.

    @param data The string data to parse as a CSV.
    @param types A mapping from column name to a cell parser for the given column. The
                 parsers will be updated in place with the parsed data.
    @param delimiter The delimiter between cells.
    @param line_ending The separator between line.
 */
void parse(const std::string_view& data,
           const std::unordered_map<std::string, std::shared_ptr<cell_parser>>& types,
           char delimiter,
           const std::string_view& line_ending,
           std::size_t num_threads) {

    auto get_parser =
        [&](const auto& cell) -> std::optional<std::shared_ptr<cell_parser>> {
        auto search = types.find(cell);
        if (search == types.end()) {
            return {};
        }
        return search->second;
    };
    auto [to_parse, parsers] =
        detail::parse_header<std::shared_ptr>(data, delimiter, line_ending, get_parser);

    detail::parse_from_header(to_parse, parsers, line_ending, num_threads);
}

/** Python CSV parsing function.

    @param data The string data to parse as a CSV.
    @param dtypes A Python dictionary from column name to dtype. Columns not
    present are ignored.
    @param delimiter The delimiter between cells.
    @param line_ending The separator between line.
    @return A Python dictionary from column name to a tuple of (value, mask)
   arrays.
 */
template<typename... possible_types>
PyObject* py_parse(PyObject*,
                   const std::string_view& data,
                   PyObject* dtypes,
                   char delimiter,
                   const std::string_view& line_ending,
                   std::size_t num_threads) {
    if (PyDict_Size(dtypes) < 0) {
        // use `PyDict_Size` to ensure this is a dict with a reasonable error message
        return nullptr;
    }

    std::vector<py::scoped_ref<PyObject>> header;

    auto get_parser =
        [&](const auto& cell) -> std::optional<std::unique_ptr<cell_parser>> {
        header.emplace_back(py::to_object(cell));

        PyObject* dtype = PyDict_GetItem(dtypes, header.back().get());
        if (!dtype) {
            return {};
        }

        return detail::create_parser<possible_types...>::f(dtype, delimiter);
    };

    auto [to_parse, parsers] =
        detail::parse_header<std::unique_ptr>(data, delimiter, line_ending, get_parser);

    detail::verify_dtypes_dict(dtypes, header);
    detail::parse_from_header(to_parse, parsers, line_ending, num_threads);

    auto out = py::scoped_ref(PyDict_New());
    if (!out) {
        return nullptr;
    }

    for (std::size_t ix = 0; ix < header.size(); ++ix) {
        auto& name = header[ix];
        auto& parser = parsers[ix];

        auto value = std::move(*parser).move_to_python_tuple();
        if (!value) {
            return nullptr;
        }
        if (value.get() == Py_None) {
            continue;
        }

        if (PyDict_SetItem(out.get(), name.get(), value.get())) {
            return nullptr;
        }
    }

    return std::move(out).escape();
}
}  // namespace py::csv
