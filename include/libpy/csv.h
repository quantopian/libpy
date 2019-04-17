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

#include "libpy/datetime64.h"
#include "libpy/exception.h"
#include "libpy/itertools.h"
#include "libpy/numpy_utils.h"
#include "libpy/scoped_ref.h"
#include "libpy/to_object.h"
#include "libpy/valgrind.h"

namespace py::csv {
/** Tag type for marking that CSV parsing should use use `fast_strtod` which is much
 * faster but loses precision. This is primarily used to speed up development cycles, and
 * it should not be used in production unless you are absolutely certain it is okay.
 */
struct fast_float32 {};

/** Tag type for marking that CSV parsing should use use `fast_strtod` which is much
 * faster but loses precision. This is primarily used to speed up development cycles, and
 * it should not be used in production unless you are absolutely certain it is okay.
 */
struct fast_float64 {};
}  // namespace py::csv

namespace py::dispatch {
template<>
struct new_dtype<py::csv::fast_float32> : public new_dtype<float> {};

template<>
struct new_dtype<py::csv::fast_float64> : public new_dtype<double> {};
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
    char m_delim = '\0';

public:
    /** Set the line count. This should pre-allocate space for `num_lines` values to be
        parsed.
     */
    virtual void set_num_lines(std::size_t) {}

    /** Set the delimiter.
     */
    void set_delim(char delim) {
        m_delim = delim;
    }

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
    virtual py::scoped_ref<> move_to_python_tuple() && {
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

    const std::vector<T>& parsed() const {
        return m_parsed;
    }

    virtual void set_num_lines(std::size_t nrows) {
        m_parsed.resize(nrows);
        m_mask.resize(nrows);
    }

    py::scoped_ref<> move_to_python_tuple() && {
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

    virtual std::tuple<std::vector<T>, std::vector<py::py_bool>> move_to_tuple() && {
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
template<const auto& scalar_parse>
class fundamental_parser : public typed_cell_parser_base<decltype(
                               scalar_parse((const char*){}, (const char**){}))> {
public:
    virtual std::tuple<std::size_t, bool>
    chomp(std::size_t ix, const std::string_view& row, std::size_t offset) {
        const char* first = &row.data()[offset];
        const char* last;
        this->m_parsed[ix] = scalar_parse(first, &last);

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
            throw detail::formatted_error(
                "invalid digit in ",
                py::util::type_name<typename fundamental_parser::type>().get(),
                ": ",
                cell);
        }

        this->m_mask[ix] = size > 0;

        bool more = *last == this->m_delim;
        return {size + more, more};
    }
};

template<typename T>
T fast_unsigned_strtod(const char* ptr, const char** last) {
    T result;
    T whole_part = 0;
    T fractional_part = 0;
    T fractional_denom = 1;

    while (true) {
        char c = *ptr;

        if (c == 'e' || c == 'E') {
            ++ptr;
            goto begin_exponent;
        }
        if (c == '.') {
            ++ptr;
            goto after_decimal;
        }

        int value = c - '0';
        if (value < 0 || value > 9) {
            *last = ptr;
            return whole_part;
        }

        whole_part *= 10;
        whole_part += value;
        ++ptr;
    }

after_decimal:
    while (true) {
        char c = *ptr;
        if (c == 'e' || c == 'E') {
            ++ptr;
            goto begin_exponent;
        }

        int value = c - '0';
        if (value < 0 || value > 9) {
            *last = ptr;

            result = whole_part + fractional_part / fractional_denom;
            return result;
        }

        fractional_denom *= 10;
        fractional_part *= 10;
        fractional_part += value;
        ++ptr;
    }

begin_exponent:
    result = whole_part + fractional_part / fractional_denom;

    long exponent = 0;
    bool exponent_negate = *ptr == '-';
    if (exponent_negate || *ptr == '+') {
        ++ptr;
    }
    while (true) {
        int value = *ptr - '0';
        if (value < 0 || value > 9) {
            *last = ptr;

            if (exponent_negate) {
                exponent = -exponent_negate;
            }
            result *= std::pow(10, exponent_negate);
            return result;
        }

        exponent *= 10;
        exponent += value;
        ++ptr;
    }
}

template<typename T>
T fast_unsigned_strtol(const char* ptr, const char** last) {
    T result = 0;
    while (true) {
        char c = *ptr - '0';
        if (c < 0 || c > 9) {
            *last = ptr;
            return result;
        }

        if (__builtin_mul_overflow(result, 10, &result) ||
            __builtin_add_overflow(result, c, &result)) {
            throw std::overflow_error("integer would overflow");
        }
        ++ptr;
    }
}

template<auto F>
auto signed_adapter(const char* ptr, const char** last) -> decltype(F(ptr, last)) {
    bool negate = *ptr == '-';
    if (negate) {
        ++ptr;
        return -F(ptr, last);
    }
    return F(ptr, last);
}

}  // namespace detail

/** A wrapper around `std::strtod` to give it the same interface as `fast_strtod`.

    @tparam T Either `float` or `double` to switch between `std::strtof` and
              `std::strtod`.
    @param ptr The beginning of the string to parse.
    @param last An output argument to take a pointer to the first character not parsed.
    @return As much of `ptr` parsed as a double as possible.
 */
template<typename T>
T regular_strtod(const char* ptr, const char** last) {
    static_assert(std::is_same_v<T, double> || std::is_same_v<T, float>,
                  "regular_strtod<T>: T must be `double` or `float`");

    // strtod and strtof skip leading whitespace before parsing, which causes problems
    // when there's a missing value at the end of a line. Therefore, the first character
    // of a cell is whitespace, treat it as a failed parse.
    if (std::isspace(*ptr)) {
        *last = ptr;
        return {};
    }

    if constexpr (std::is_same_v<T, double>) {
        return std::strtod(ptr, const_cast<char**>(last));
    }
    else {
        return std::strtof(ptr, const_cast<char**>(last));
    }
}

/** A faster, but lower precision, implementation of `strtod`.

    @tparam T The precision of the value to parse, either `float` or `double`.
    @param ptr The beginning of the string to parse.
    @param last An output argument to take a pointer to the first character not parsed.
    @return As much of `ptr` parsed as a `T` as possible.
 */
template<typename T>
auto fast_strtod = detail::signed_adapter<detail::fast_unsigned_strtod<T>>;

/** A faster, but less accepting, implementation of `strtol`.

    @tparam T The type of integer to parse as. This should be a signed integral type.
    @param ptr The beginning of the string to parse.
    @param last An output argument to take a pointer to the first character not parsed.
    @return As much of `ptr` parsed as a `T` as possible.
 */
template<typename T,
         typename = std::enable_if_t<std::is_integral_v<T> && std::is_signed_v<T>>>
auto fast_strtol = detail::signed_adapter<detail::fast_unsigned_strtol<T>>;

template<>
class typed_cell_parser<float>
    : public detail::fundamental_parser<regular_strtod<float>> {};

template<>
class typed_cell_parser<double>
    : public detail::fundamental_parser<regular_strtod<double>> {};

template<>
class typed_cell_parser<fast_float32>
    : public detail::fundamental_parser<fast_strtod<float>> {};

template<>
class typed_cell_parser<fast_float64>
    : public detail::fundamental_parser<fast_strtod<double>> {};

template<>
class typed_cell_parser<std::int8_t>
    : public detail::fundamental_parser<fast_strtol<std::int8_t>> {};

template<>
class typed_cell_parser<std::int16_t>
    : public detail::fundamental_parser<fast_strtol<std::int16_t>> {};

template<>
class typed_cell_parser<std::int32_t>
    : public detail::fundamental_parser<fast_strtol<std::int32_t>> {};

template<>
class typed_cell_parser<std::int64_t>
    : public detail::fundamental_parser<fast_strtol<std::int64_t>> {};

/* Regardless of the input type, we always convert datetimes to datetime64<ns> for the
   output buffer.
*/
template<typename Unit>
class typed_cell_parser<py::datetime64<Unit>>
    : public typed_cell_parser_base<py::datetime64ns> {
private:
    static_assert(std::is_same_v<Unit, py::chrono::s> ||
                      std::is_same_v<Unit, py::chrono::D>,
                  "can only parse datetime64 at second or daily frequency");

    /** The number of days in each month for non-leap years.
     */
    static constexpr std::array<uint8_t, 12> days_in_month = {31,   // jan
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

    static constexpr std::array<uint16_t, 12> build_days_before_month() {
        std::array<uint16_t, 12> out = {0};
        for (std::size_t n = 1; n < 12; ++n) {
            out[n] = days_in_month[n - 1] + out[n - 1];
        }
        return out;
    }

    /** The number of days that occur before the first of the month in a non-leap year.
     */
    static constexpr std::array<uint16_t, 12> days_before_month =
        build_days_before_month();

    static constexpr bool is_leapyear(int year) {
        return !(year % 4) && (!(year % 100) || year % 400);
    }

    static constexpr int leap_years_before(int year) {
        --year;
        return (year / 4) - (year / 100) + (year / 400);
    }

    static std::chrono::seconds time_since_epoch(int year, int month, int day) {
        using days = std::chrono::duration<std::int64_t, std::ratio<86400>>;
        // The number of seconds in 365 days. This doesn't account for leap years, we will
        // manually add those days.
        using years = std::chrono::duration<std::int64_t, std::ratio<31536000>>;

        std::chrono::seconds out = years(year - 1970);
        out += days(leap_years_before(year) - leap_years_before(1970));
        out += days(days_before_month[month - 1]);
        out += days(month > 2 && is_leapyear(year));
        out += days(day - 1);

        return out;
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

    static void expect_char(const std::string_view& raw, std::size_t ix, char c) {
        if (raw[ix] != c) {
            throw detail::formatted_error("expected '", c, "' at index ", ix, ": ", raw);
        }
    }

    static std::tuple<int, int, int> parse_year_month_day(const std::string_view& raw,
                                                          bool expect_time) {
        if (expect_time) {
            if (raw.size() <= 10) {
                throw detail::formatted_error(
                    "date string is not at least 10 characters: ", raw);
            }
        }
        else if (raw.size() != 10) {
            throw detail::formatted_error("date string is not exactly 10 characters: ",
                                          raw);
        }

        int year = parse_int<4>(raw);
        expect_char(raw, 4, '-');
        int month = parse_int<2>(raw.substr(5));
        if (month < 1 || month > 12) {
            throw detail::formatted_error("month not in range [1, 12]: ", raw);
        }

        expect_char(raw, 7, '-');
        int leap_day = month == 2 && is_leapyear(year);

        int day = parse_int<2>(raw.substr(8));
        int max_day = days_in_month[month - 1] + leap_day;
        if (day < 1 || day > max_day) {
            throw detail::formatted_error(
                "day out of bounds for month (max=", max_day, "): ", raw);
        }

        return {year, month, day};
    }

    static std::tuple<int, int, int>
    parse_hours_minutes_seconds(const std::string_view& raw) {
        if (raw.size() != 19) {
            throw detail::formatted_error("date string is not exactly 19 characters: ",
                                          raw);
        }

        expect_char(raw, 10, ' ');
        int hours = parse_int<2>(raw.substr(11));
        if (hours < 0 || hours > 23) {
            throw detail::formatted_error("hour not in range [0, 24): ", raw);
        }

        expect_char(raw, 13, ':');
        int minutes = parse_int<2>(raw.substr(14));

        expect_char(raw, 16, ':');
        int seconds = parse_int<2>(raw.substr(17));

        return {hours, minutes, seconds};
    }

public:
    virtual std::tuple<std::size_t, bool>
    chomp(std::size_t ix, const std::string_view& row, std::size_t offset) {
        auto [raw, consumed, more] =
            detail::isolate_unquoted_cell(row, offset, this->m_delim);
        if (!raw.size()) {
            return {consumed, more};
        }

        bool expect_time = std::is_same_v<Unit, py::chrono::s>;
        auto value = std::apply(time_since_epoch, parse_year_month_day(raw, expect_time));
        if (expect_time) {
            auto [hours, minutes, seconds] = parse_hours_minutes_seconds(raw);
            value += std::chrono::hours(hours);
            value += std::chrono::minutes(minutes);
            value += std::chrono::seconds(seconds);
        }

        this->m_parsed[ix] = py::datetime64ns(value);
        this->m_mask[ix] = true;
        return {consumed, more};
    }
};

template<>
class typed_cell_parser<py::py_bool> : public typed_cell_parser_base<py::py_bool> {
public:
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
                       char delimiter,
                       const std::string_view& line_ending,
                       std::size_t num_threads,
                       std::size_t skip_rows = 0) {
    std::vector<std::string_view> lines = split_into_lines(data, line_ending, skip_rows);

    for (auto& parser : parsers) {
        parser->set_num_lines(lines.size());
        parser->set_delim(delimiter);
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

    static std::unique_ptr<cell_parser> create_parser() {
        return std::make_unique<typed_cell_parser<T>>();
    }
};

template<typename...>
struct create_parser;

template<typename T, typename... Ts>
struct create_parser<T, Ts...> {
    static std::unique_ptr<cell_parser> f(PyObject* dtype) {
        using option = dtype_option<T>;
        if (!option::matches(dtype)) {
            return create_parser<Ts...>::f(dtype);
        }

        return option::create_parser();
    }
};

template<>
struct create_parser<> {
    [[noreturn]] static std::unique_ptr<cell_parser> f(PyObject* dtype) {
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

    header_parser header_parser;
    header_parser.set_delim(delimiter);
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
                parsers.emplace_back(std::make_shared<skip_parser>());
            }
            else {
                parsers.emplace_back(std::make_unique<skip_parser>());
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
void verify_dtypes_dict(PyObject* dtypes, std::vector<py::scoped_ref<>>& header) {
    py::scoped_ref expected_keys(PySet_New(dtypes));
    if (!expected_keys) {
        throw py::exception();
    }
    py::scoped_ref actual_keys = py::to_object(header);
    if (!actual_keys) {
        throw py::exception();
    }

    py::scoped_ref actual_keys_set(PySet_New(actual_keys.get()));
    if (!actual_keys_set) {
        throw py::exception();
    }

    py::scoped_ref diff(PyNumber_Subtract(expected_keys.get(), actual_keys_set.get()));
    if (!diff) {
        throw py::exception();
    }
    if (PySet_GET_SIZE(diff.get())) {
        py::scoped_ref as_list(PySequence_List(diff.get()));
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

    detail::parse_from_header(to_parse, parsers, delimiter, line_ending, num_threads);
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

    std::vector<py::scoped_ref<>> header;

    auto get_parser =
        [&](const auto& cell) -> std::optional<std::unique_ptr<cell_parser>> {
        header.emplace_back(py::to_object(cell));

        PyObject* dtype = PyDict_GetItem(dtypes, header.back().get());
        if (!dtype) {
            return {};
        }

        return detail::create_parser<possible_types...>::f(dtype);
    };

    auto [to_parse, parsers] =
        detail::parse_header<std::unique_ptr>(data, delimiter, line_ending, get_parser);

    detail::verify_dtypes_dict(dtypes, header);
    detail::parse_from_header(to_parse, parsers, delimiter, line_ending, num_threads);

    py::scoped_ref out(PyDict_New());
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
