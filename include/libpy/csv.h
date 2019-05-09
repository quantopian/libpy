#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <exception>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
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
#include "libpy/scope_guard.h"
#include "libpy/scoped_ref.h"
#include "libpy/stream.h"
#include "libpy/to_object.h"
#include "libpy/valgrind.h"

#if LIBPY_NO_CSV_PREFETCH
#define __builtin_prefetch(...)
#endif

namespace py::csv {
namespace detail {
// Allow us to configure the parser for a different l1dcache line size at compile time.
#ifndef LIBPY_L1DCACHE_LINE_SIZE
#define LIBPY_L1DCACHE_LINE_SIZE 64
#endif

int l1dcache_line_size = LIBPY_L1DCACHE_LINE_SIZE;
}  // namespace detail

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
constexpr std::size_t min_split_lines_bytes_size = 16384;

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

template<typename T>
struct destruct_but_not_free {
    void operator()(T* value) {
        if (value) {
            value->~T();
        }
    }
};

/** A unique pointer which only destructs, but doesn't free the storage. It is designed to
    be used to provide unique ownership and lifetime management of a dynamic value
    allocated in a `std::aligned_storage_t` or `std::aligned_union_t` on the stack.
*/
template<typename T>
using stack_allocated_unique_ptr = std::unique_ptr<T, destruct_but_not_free<T>>;
}  // namespace detail

/** A cell parser is an object that represents a row in a CSV.

    Subclasses are required to implement `chomp``.
 */
class cell_parser {
public:
    /** Set the line count. This should pre-allocate space for `num_lines` values to be
        parsed.
     */
    virtual void set_num_lines(std::size_t) {}

    virtual ~cell_parser() = default;

    /** "chomp" text from a row and parse the given cell.

        @param delim The delimiter.
        @param row_ix The row number (0-indexed) being parsed.
        @param row The entire row being parsed.
        @param offset The offset into `row` where the cell starts.
        @return A tuple of the number of characters consumed from the row and a boolean
                which is true if we expect there to be more columns to parse in this row.
     */
    virtual std::tuple<std::size_t, bool> chomp(char delim,
                                                std::size_t row_ix,
                                                const std::string_view& row,
                                                std::size_t offset) = 0;

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

    virtual void set_num_lines(std::size_t nrows) override {
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

    std::tuple<std::vector<T>, std::vector<py::py_bool>> move_to_tuple() && {
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
    std::tuple<std::size_t, bool> chomp(char delim,
                                        std::size_t ix,
                                        const std::string_view& row,
                                        std::size_t offset) override {
        auto& cell = this->m_parsed[ix];
        std::size_t cell_ix = 0;
        auto ret = detail::chomp_quoted_string(
            [&](char c) {
                if (cell_ix < cell.size()) {
                    cell[cell_ix++] = c;
                }
            },
            delim,
            row,
            offset);
        this->m_mask[ix] = cell_ix > 0;
        return ret;
    }
};

namespace detail {

/**Get the result type of a function suitable for passing as a template parameter for
 * fundamental_parser.*/
template<const auto& scalar_parse>
using parse_result = decltype(scalar_parse((const char*){}, (const char**){}));

/**A typed_cell_parser implemented in terms of a scalar_parse function that takes a
 * range of characters and returns a value of fundamental type.*/
template<const auto& scalar_parse>
class fundamental_parser : public typed_cell_parser_base<parse_result<scalar_parse>> {
public:
    using type = typename typed_cell_parser_base<parse_result<scalar_parse>>::type;

    std::tuple<std::size_t, bool> chomp(char delim,
                                        std::size_t ix,
                                        const std::string_view& row,
                                        std::size_t offset) override {
        const char* first = &row.data()[offset];
        const char* last;
        type parsed = scalar_parse(first, &last);

        std::size_t size = last - first;
        if (*last != delim && size != row.size() - offset) {
            // error if we are not at the end of a cell nor at the end of a row
            std::string_view cell;
            auto end = std::memchr(first, delim, row.size() - offset);
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

        if (size > 0) {
            this->m_mask[ix] = true;
            this->m_parsed[ix] = parsed;
        }

        bool more = *last == delim;
        return {size + more, more};
    }
};

/** Special case of fundamental_parser for parsing floats that initializes output buffers
 *  with NaN.
 */
template<const auto& scalar_parse>
class float_parser : public fundamental_parser<scalar_parse> {
public:
    using type = typename fundamental_parser<scalar_parse>::type;

    virtual void set_num_lines(std::size_t nrows) override {
        this->m_parsed.resize(nrows, std::numeric_limits<type>::quiet_NaN());
        this->m_mask.resize(nrows);
    }
};

template<typename T>
T fast_unsigned_strtod(const char* ptr, const char** last) {
    T result;
    std::int64_t whole_part = 0;
    std::int64_t fractional_part = 0;
    std::int64_t fractional_denom = 1;

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

            result = whole_part + static_cast<double>(fractional_part) / fractional_denom;
            return result;
        }

        fractional_denom *= 10;
        fractional_part *= 10;
        fractional_part += value;
        ++ptr;
    }

begin_exponent:
    result = whole_part + static_cast<double>(fractional_part) / fractional_denom;

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
class typed_cell_parser<float> : public detail::float_parser<regular_strtod<float>> {};

template<>
class typed_cell_parser<double> : public detail::float_parser<regular_strtod<double>> {};

template<>
class typed_cell_parser<fast_float32> : public detail::float_parser<fast_strtod<float>> {
};

template<>
class typed_cell_parser<fast_float64> : public detail::float_parser<fast_strtod<double>> {
};

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
                      std::is_same_v<Unit, py::chrono::D> ||
                      std::is_same_v<Unit, py::chrono::ns>,
                  "can only parse datetime64 at second, daily, or nanosecond frequency");

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
        return (year % 4) == 0 && ((year % 100) != 0 || (year % 400) == 0);
    }

    static constexpr int leap_years_before(int year) {
        --year;
        return (year / 4) - (year / 100) + (year / 400);
    }

    static std::chrono::nanoseconds time_since_epoch(int year, int month, int day) {
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

    template<typename... Cs>
    static void expect_char(const std::string_view& raw, std::size_t ix, Cs... cs) {
        if (((raw[ix] != cs) && ...)) {
            std::string options;
            (options.push_back(cs), ...);
            throw detail::formatted_error("expected one of: \"",
                                          options,
                                          "\" at index ",
                                          ix,
                                          ": ",
                                          raw);
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

    static std::tuple<int, int, int, int>
    parse_hours_minutes_seconds_nanoseconds(const std::string_view& raw) {
        constexpr std::size_t no_fractional_second_size = 19;
        if (std::is_same_v<Unit, py::chrono::s>) {
            if (raw.size() != no_fractional_second_size) {
                throw detail::formatted_error(
                    "datetime string is not exactly 19 characters: ", raw);
            }
        }
        else if (raw.size() < no_fractional_second_size) {
            throw detail::formatted_error(
                "datetime string is not at least 19 characters: ", raw);
        }

        expect_char(raw, 10, ' ', 'T');
        int hours = parse_int<2>(raw.substr(11));
        if (hours < 0 || hours > 23) {
            throw detail::formatted_error("hour not in range [0, 24): ", raw);
        }

        expect_char(raw, 13, ':');
        int minutes = parse_int<2>(raw.substr(14));

        expect_char(raw, 16, ':');
        int seconds = parse_int<2>(raw.substr(17));

        int nanoseconds = 0;
        if (std::is_same_v<Unit, py::chrono::ns> &&
            raw.size() > no_fractional_second_size) {
            expect_char(raw, 19, '.');
            const char* end;
            const char* begin = raw.begin() + 20;
            nanoseconds = detail::fast_unsigned_strtol<int>(begin, &end);
            if (end != raw.end()) {
                throw detail::formatted_error(
                    "couldn't parse fractional seconds component: ", raw);
            }
            std::ptrdiff_t digits = end - begin;
            nanoseconds *= std::pow(10, 9 - digits);
        }

        return {hours, minutes, seconds, nanoseconds};
    }

public:
    std::tuple<std::size_t, bool> chomp(char delim,
                                        std::size_t ix,
                                        const std::string_view& row,
                                        std::size_t offset) override {
        auto [raw, consumed, more] = detail::isolate_unquoted_cell(row, offset, delim);
        if (!raw.size()) {
            return {consumed, more};
        }

        bool expect_time = !std::is_same_v<Unit, py::chrono::D>;
        auto value = std::apply(time_since_epoch, parse_year_month_day(raw, expect_time));
        if (expect_time) {
            auto [hours, minutes, seconds, nanoseconds] =
                parse_hours_minutes_seconds_nanoseconds(raw);
            value += std::chrono::hours(hours);
            value += std::chrono::minutes(minutes);
            value += std::chrono::seconds(seconds);
            value += std::chrono::nanoseconds(nanoseconds);
        }

        this->m_parsed[ix] = py::datetime64ns(value);
        this->m_mask[ix] = true;
        return {consumed, more};
    }
};

template<>
class typed_cell_parser<py::py_bool> : public typed_cell_parser_base<py::py_bool> {
public:
    std::tuple<std::size_t, bool> chomp(char delim,
                                        std::size_t ix,
                                        const std::string_view& row,
                                        std::size_t offset) override {
        auto [raw, consumed, more] = detail::isolate_unquoted_cell(row, offset, delim);
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
    std::tuple<std::size_t, bool> chomp(char delim,
                                        std::size_t,
                                        const std::string_view& row,
                                        std::size_t offset) override {
        this->m_parsed.emplace_back();
        auto& cell = this->m_parsed.back();
        return detail::chomp_quoted_string([&](char c) { cell.push_back(c); },
                                           delim,
                                           row,
                                           offset);
    }

    const std::vector<std::string>& parsed() const {
        return m_parsed;
    }
};

class skip_parser : public cell_parser {
public:
    std::tuple<std::size_t, bool> chomp(char delim,
                                        std::size_t,
                                        const std::string_view& row,
                                        std::size_t offset) override {
        return detail::chomp_quoted_string([](char) {}, delim, row, offset);
    }
};

namespace detail {
/** An alias for creating a vector of cell parsers which uses a "template template
    parameter" to be parametric over the type of smart pointer used. The two standard
    options for `ptr_type` are `std::shared_ptr` and `stack_allocated_unique_ptr`.

    @tparam ptr_type The type of smart pointer to `cell_parser` to use.
 */
template<template<typename> typename ptr_type>
using parser_types = std::vector<ptr_type<cell_parser>>;

/** Parse a single row and store the values in the vectors.

    @param row The row index.
    @param data The view over the row to parse.
    @param parsers The cell parsers.
 */
template<template<typename> typename ptr_type>
void parse_row(std::size_t row,
               char delim,
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
            auto [new_consumed, new_more] = parser->chomp(delim, row, data, consumed);
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

template<template<typename> typename ptr_type>
void parse_lines(const std::string_view data,
                 char delim,
                 std::size_t data_offset,
                 const std::vector<std::size_t>& line_sizes,
                 std::size_t line_end_size,
                 std::size_t offset,
                 parser_types<ptr_type>& parsers) {
    std::size_t ix = offset;
    for (const auto& size : line_sizes) {
        auto row = data.substr(data_offset, size);
        __builtin_prefetch(row.data() + size, 0, 0);
        __builtin_prefetch(row.data() + size + l1dcache_line_size, 0, 0);
        __builtin_prefetch(row.data() + size + 2 * l1dcache_line_size, 0, 0);
        parse_row<ptr_type>(ix, delim, row, parsers);
        data_offset += size + line_end_size;
        ++ix;
    }
}

template<template<typename> typename ptr_type>
void parse_lines_worker(std::mutex* exception_mutex,
                        std::vector<std::exception_ptr>* exceptions,
                        const std::string_view* data,
                        char delim,
                        const std::size_t data_offset,
                        const std::vector<std::size_t>* line_sizes,
                        std::size_t line_end_size,
                        std::size_t offset,
                        parser_types<ptr_type>* parsers) {
    try {
        parse_lines<ptr_type>(
            *data, delim, data_offset, *line_sizes, line_end_size, offset, *parsers);
    }
    catch (const std::exception&) {
        std::lock_guard<std::mutex> guard(*exception_mutex);
        exceptions->emplace_back(std::current_exception());
    }
}

void split_into_lines_loop(std::vector<std::size_t>& lines,
                           const std::string_view& data,
                           std::string_view::size_type* pos_ptr,
                           std::string_view::size_type end_ix,
                           const std::string_view& line_ending,
                           bool handle_tail) {
    auto pos = *pos_ptr;
    std::string_view::size_type end;

    if (pos >= line_ending.size() &&
        data.substr(pos - line_ending.size(), line_ending.size()) != line_ending) {
        end = data.find(line_ending, pos);
        if (end == std::string_view::npos) {
            *pos_ptr = pos = end_ix;
            return;
        }
        else {
            *pos_ptr = pos = end + line_ending.size();
        }
    }

    while ((end = data.find(line_ending, pos)) != std::string_view::npos) {
        auto size = end - pos;
        lines.emplace_back(size);

        if (end >= end_ix) {
            return;
        }

        // advance past line ending
        pos = end + line_ending.size();
        __builtin_prefetch(data.data() + end + size, 0, 0);
        __builtin_prefetch(data.data() + end + size + l1dcache_line_size, 0, 0);
    }

    if (handle_tail and pos < end_ix) {
        // add any data after the last newline if there is anything to add
        lines.emplace_back(end_ix - pos);
    }
}

void split_into_lines_worker(std::mutex* exception_mutex,
                             std::vector<std::exception_ptr>* exceptions,
                             std::vector<std::size_t>* lines,
                             const std::string_view* data,
                             std::string_view::size_type* pos,
                             std::string_view::size_type end_ix,
                             const std::string_view* line_ending,
                             bool handle_tail) {
    try {
        split_into_lines_loop(*lines, *data, pos, end_ix, *line_ending, handle_tail);
    }
    catch (const std::exception&) {
        std::lock_guard<std::mutex> guard(*exception_mutex);
        exceptions->emplace_back(std::current_exception());
    }
}

std::tuple<std::vector<std::size_t>, std::vector<std::vector<std::size_t>>>
split_into_lines(const std::string_view& data,
                 const std::string_view& line_ending,
                 std::size_t num_columns,
                 std::size_t skip_rows,
                 std::size_t num_threads) {
    if (num_threads == 0) {
        num_threads = 1;
    }

    std::size_t group_size = data.size() / num_threads + 1;
    if (group_size < min_split_lines_bytes_size) {
        // not really worth breaking this file up
        num_threads = 1;
    }

    std::vector<std::vector<std::size_t>> lines_per_thread(num_threads);
    std::vector<std::size_t> thread_starts(num_threads);
    for (auto& lines : lines_per_thread) {
        // assume that each column will take about 5 bytes of data on average
        lines.reserve(std::ceil(group_size / (4.0 * num_columns)));
    }

    // The current position into the input.
    std::string_view::size_type pos = 0;
    // The index of the next newline.
    std::string_view::size_type end;

    // optionally skip some rows
    for (std::size_t n = 0; n < skip_rows; ++n) {
        if ((end = data.find(line_ending, pos)) == std::string_view::npos) {
            break;
        }
        // advance past line ending
        pos = end + line_ending.size();
    }

    if (num_threads == 1) {
        split_into_lines_loop(lines_per_thread[0],
                              data,
                              &pos,
                              data.size(),
                              line_ending,
                              /* handle_tail */ true);
        thread_starts[0] = pos;
    }
    else {
        std::mutex exception_mutex;
        std::vector<std::exception_ptr> exceptions;

        std::vector<std::thread> threads;
        for (std::size_t n = 0; n < num_threads; ++n) {
            thread_starts[n] = pos + n * group_size;
            threads.emplace_back(
                std::thread(split_into_lines_worker,
                            &exception_mutex,
                            &exceptions,
                            &lines_per_thread[n],
                            &data,
                            &thread_starts[n],
                            std::min(thread_starts[n] + group_size, data.size()),
                            &line_ending,
                            /* handle_thread */ n == num_threads - 1));
        }

        for (auto& thread : threads) {
            thread.join();
        }

        for (auto& e : exceptions) {
            std::rethrow_exception(e);
        }
    }

    return {std::move(thread_starts), std::move(lines_per_thread)};
}

/** Parse a full CSV given the header and the types of the columns.

    @param data The CSV to parse.
    @param parsers The cell parsers.
    @param line_ending The string to split lines on.
 */
template<template<typename> typename ptr_type>
void parse_from_header(const std::string_view& data,
                       parser_types<ptr_type>& parsers,
                       char delimiter,
                       const std::string_view& line_ending,
                       std::size_t num_threads,
                       std::size_t skip_rows = 0) {
    auto [thread_starts, line_sizes_per_thread] =
        split_into_lines(data, line_ending, parsers.size(), skip_rows, num_threads);

    std::size_t lines = std::accumulate(line_sizes_per_thread.begin(),
                                        line_sizes_per_thread.end(),
                                        0,
                                        [](auto sum, const auto& vec) {
                                            return sum + vec.size();
                                        });

    for (auto& parser : parsers) {
        parser->set_num_lines(lines);
    }

    if (line_sizes_per_thread.size() == 1) {
        parse_lines<ptr_type>(data,
                              delimiter,
                              thread_starts[0],
                              line_sizes_per_thread[0],
                              line_ending.size(),
                              0,
                              parsers);
    }
    else {
        std::mutex exception_mutex;
        std::vector<std::exception_ptr> exceptions;

        std::vector<std::thread> threads;
        std::size_t start = 0;
        for (std::size_t n = 0; n < num_threads; ++n) {
            threads.emplace_back(std::thread(parse_lines_worker<ptr_type>,
                                             &exception_mutex,
                                             &exceptions,
                                             &data,
                                             delimiter,
                                             thread_starts[n],
                                             &line_sizes_per_thread[n],
                                             line_ending.size(),
                                             start,
                                             &parsers));
            start += line_sizes_per_thread[n].size();
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

    static stack_allocated_unique_ptr<cell_parser> create_parser(void* addr) {
        // placement new the `typed_cell_parser` into our storage and then create a
        // `unique_ptr` to manage the lifetime of that object.
        return stack_allocated_unique_ptr<cell_parser>(new (addr) typed_cell_parser<T>{});
    }
};

template<typename...>
struct create_parser;

template<typename T, typename... Ts>
struct create_parser<T, Ts...> {
    static stack_allocated_unique_ptr<cell_parser> f(PyObject* dtype, void* addr) {
        using option = dtype_option<T>;
        if (!option::matches(dtype)) {
            return create_parser<Ts...>::f(dtype, addr);
        }

        return option::create_parser(addr);
    }
};

template<>
struct create_parser<> {
    [[noreturn]] static stack_allocated_unique_ptr<cell_parser> f(PyObject* dtype,
                                                                  void*) {
        throw py::exception(PyExc_TypeError, "unknown dtype: ", dtype);
    }
};

template<template<typename> typename ptr_type, typename GetParser, typename Init>
std::tuple<std::string_view, parser_types<ptr_type>>
parse_header(const std::string_view& data,
             char delimiter,
             const std::string_view& line_ending,
             Init&& init,
             GetParser&& get_parser) {
    constexpr bool is_shared = std::is_same_v<ptr_type<void>, std::shared_ptr<void>>;
    constexpr bool is_unique =
        std::is_same_v<ptr_type<void>, stack_allocated_unique_ptr<void>>;
    static_assert(is_shared || is_unique,
                  "ptr_type must be std::shared_ptr or stack_allocated_unique_ptr");

    auto line_end = data.find(line_ending, 0);
    auto line = data.substr(0, line_end);

    header_parser header_parser;
    for (auto [consumed, more] = std::make_tuple(0, true); more;) {
        auto [new_consumed, new_more] = header_parser.chomp(delimiter, 0, line, consumed);
        consumed += new_consumed;
        more = new_more;
    }

    init(header_parser.parsed().size());

    std::unordered_set<std::string> column_names;
    parser_types<ptr_type> parsers(header_parser.parsed().size());
    std::size_t ix = 0;
    for (const auto& cell : header_parser.parsed()) {
        if (column_names.count(cell)) {
            throw detail::formatted_error("column name duplicated: ", cell);
        }
        column_names.emplace(cell);
        parsers[ix] = get_parser(ix, cell);
        ++ix;
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

    auto get_parser = [&](std::size_t, const auto& cell) -> std::shared_ptr<cell_parser> {
        auto search = types.find(cell);
        if (search == types.end()) {
            return std::make_shared<skip_parser>();
        }

        return search->second;
    };
    auto [to_parse, parsers] = detail::parse_header<std::shared_ptr>(
        data, delimiter, line_ending, [](std::size_t) {}, get_parser);

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
    Py_ssize_t num_dtypes = PyDict_Size(dtypes);
    if (num_dtypes < 0) {
        // use `PyDict_Size` to ensure this is a dict with a reasonable error message
        return nullptr;
    }

    if (num_dtypes == 0) {
        // empty dict, data doesn't matter
        return PyDict_New();
    }

    // allocate all of the parser objects on the stack in a contiguous buffer to reduce
    // the cache pressure in the `parse_lines` loop
    using cell_parser_storage =
        std::aligned_union_t<0, skip_parser, typed_cell_parser<possible_types>...>;
    std::vector<cell_parser_storage> parser_storage;

    std::vector<py::scoped_ref<>> header;

    auto init = [&](std::size_t num_cols) {
        parser_storage.resize(num_cols);
        header.resize(num_cols);
    };

    auto get_parser = [&](std::size_t ix, const auto& cell) {
        auto& cell_ob = header[ix] = py::to_object(cell);

        cell_parser_storage* addr = &parser_storage[ix];

        PyObject* dtype = PyDict_GetItem(dtypes, cell_ob.get());
        if (dtype) {
            return detail::create_parser<possible_types...>::f(dtype, addr);
        }
        else {
            // placement new the `skip_parser` into our storage and then create
            // a `unique_ptr` to manage the lifetime of that object.
            return detail::stack_allocated_unique_ptr<cell_parser>(new (addr)
                                                                       skip_parser{});
        }
    };

    auto [to_parse, parsers] = detail::parse_header<detail::stack_allocated_unique_ptr>(
        data, delimiter, line_ending, init, get_parser);

    detail::verify_dtypes_dict(dtypes, header);
    detail::parse_from_header<detail::stack_allocated_unique_ptr>(to_parse,
                                                                  parsers,
                                                                  delimiter,
                                                                  line_ending,
                                                                  num_threads);

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

namespace detail {
class rope_adapter {
private:
    std::vector<std::string> m_buffers;
    std::size_t m_size = 0;

public:
    inline void write(std::string&& data, std::size_t size) {
        data.erase(data.begin() + size, data.end());
        m_buffers.emplace_back(std::move(data));
        m_size += size;
    }

    inline void write(std::string& data, std::size_t size) {
        std::string entry(data.size(), '\0');
        std::swap(data, entry);
        write(std::move(entry), size);
    }

    inline void read(char* out_buffer, std::size_t size) {
        std::size_t buffer_ix = 0;
        char* end = out_buffer + size;
        while (out_buffer < end) {
            const std::string& buffer = m_buffers[buffer_ix++];
            std::size_t to_read_from_buffer = std::min(size, buffer.size());
            std::memcpy(out_buffer, buffer.data(), to_read_from_buffer);
            size -= to_read_from_buffer;
            out_buffer += to_read_from_buffer;
        }
    }

    inline std::size_t size() const {
        return m_size;
    }
};

template<typename T>
class ostream_adapter {
private:
    T& m_stream;

public:
    ostream_adapter(T& stream) : m_stream(stream) {}

    void write(const std::string& data, std::size_t size) {
        m_stream.write(data.data(), size);
    }
};

template<typename T>
class iobuffer {
private:
    // we need this to be large enough that it will be safe to do writes of floats,
    // ints, and datetimes after a flush without double checking the `space_left`
    static constexpr std::size_t min_buffer_size = 1 << 8;

    T& m_stream;
    std::string m_buffer;
    std::size_t m_ix;
    std::int64_t m_float_coef;
    std::int64_t m_expected_frac_digits;

public:
    iobuffer(T& stream, std::size_t buffer_size, std::uint8_t float_precision)
        : m_stream(stream),
          m_buffer(buffer_size, '\0'),
          m_ix(0),
          m_float_coef(std::pow(10, float_precision)),
          m_expected_frac_digits(float_precision) {
        if (__builtin_popcountll(buffer_size) != 1) {
            throw std::runtime_error("buffer_size must be power of 2");
        }

        if (buffer_size < min_buffer_size) {
            std::stringstream stream;
            stream << "buffer_size must be at least " << min_buffer_size
                   << " got: " << buffer_size;
            throw std::runtime_error(stream.str());
        }
    }

    void flush() {
        if (m_ix) {
            m_stream.write(m_buffer, m_ix);
            m_ix = 0;
        }
    }

    std::tuple<char*, char*> buffer() {
        return {m_buffer.data() + m_ix, m_buffer.data() + m_buffer.size()};
    }

    void consume(std::size_t amount) {
        m_ix += amount;
    }

    std::size_t space_left() const {
        return m_buffer.size() - m_ix;
    }

    void write(std::string&& data) {
        if (space_left() < data.size()) {
            flush();
            if (space_left() < data.size()) {
                m_stream.write(std::move(data), data.size());
                return;
            }
        }
        std::memcpy(m_buffer.data() + m_ix, data.data(), data.size());
        m_ix += data.size();
    }

    void write(const std::string& data) {
        std::string copy = data;
        write(std::move(copy));
    }

    void write(char c) {
        if (!space_left()) {
            flush();
        }
        m_buffer[m_ix++] = c;
    }

    void write(double f) {
        if (f > 1LL << 54 || f < -(1LL << 54)) {
            // abs(f) > (1 << 54) cannot be perfectly represented as an int, so the whole
            // component will lose precision. We don't expect this case to be common so we
            // just defer to stringstream
            std::stringstream ss;
            ss.precision(m_expected_frac_digits);
            ss << f;
            write(ss.str());
            return;
        }

        std::int64_t whole_component = f;
        write(whole_component);
        write('.');
        std::int64_t frac = (f - whole_component) * m_float_coef + 0.5;
        frac = std::abs(frac);
        std::int64_t digits = std::floor(std::log10(frac));
        digits += 1;
        std::int64_t padding = m_expected_frac_digits - digits;
        if (padding > 0) {
            if (static_cast<std::int64_t>(space_left()) < padding) {
                flush();
            }
            std::memset(m_buffer.data() + m_ix, '0', padding);
            m_ix += padding;
        }
        write(frac);
    }

    void write(std::int64_t v) {
        using namespace py::cs::literals;
        constexpr std::int64_t max_int_size = "-9223372036854775808"_arr.size();
        if (space_left() < max_int_size) {
            flush();
        }
        auto begin = m_buffer.data() + m_ix;
        auto [p, errc] = std::to_chars(begin, m_buffer.data() + m_buffer.size(), v);
        m_ix += p - begin;
    }

    ~iobuffer() {
        flush();
    }
};

template<typename T>
void format_any(iobuffer<T>& buf, const py::any_ref& value) {
    std::stringstream stream;
    stream << value;
    buf.write(stream.str());
}

template<typename T>
void format_pyobject(iobuffer<T>& buf, const py::any_ref& any_value) {
    const auto& as_ob = *reinterpret_cast<const py::scoped_ref<>*>(any_value.addr());
    if (as_ob.get() == Py_None) {
        return;
    }

    std::string_view text = py::util::pystring_to_string_view(as_ob);
    buf.write('"');
    for (char c : text) {
        if (c == '"') {
            buf.write('/');
        }
        buf.write(c);
    }
    buf.write('"');
}

template<typename T, typename F>
void format_float(iobuffer<T>& buf, const py::any_ref& any_value) {
    const auto& as_float = *reinterpret_cast<const F*>(any_value.addr());
    if (as_float != as_float) {
        return;
    }
    buf.write(as_float);
}

template<typename T, typename I>
void format_int(iobuffer<T>& buf, const py::any_ref& any_value) {
    std::int64_t as_int = *reinterpret_cast<const I*>(any_value.addr());
    buf.write(as_int);
}

template<typename T, typename unit>
void format_datetime64(iobuffer<T>& buf, const py::any_ref& any_value) {
    const auto& as_M8 = *reinterpret_cast<const py::datetime64<unit>*>(any_value.addr());
    if (as_M8.isnat()) {
        return;
    }
    if (buf.space_left() < py::detail::max_size<unit>) {
        buf.flush();
    }
    auto [begin, end] = buf.buffer();
    auto [p, errc] = py::to_chars(begin, end, as_M8);
    buf.consume(p - begin);
}

template<typename T>
using format_function = void (*)(detail::iobuffer<T>&, const py::any_ref&);

template<typename T>
std::vector<format_function<T>>
get_format_functions(std::vector<py::array_view<py::any_ref>>& columns) {
    std::size_t num_rows = columns[0].size();
    std::vector<detail::format_function<T>> formatters;
    for (const auto& column : columns) {
        if (column.size() != num_rows) {
            throw std::runtime_error("mismatched column lengths");
        }

        const auto& vtable = column.vtable();
        if (vtable == py::any_vtable::make<double>()) {
            formatters.emplace_back(detail::format_float<T, double>);
        }
        else if (vtable == py::any_vtable::make<float>()) {
            formatters.emplace_back(detail::format_float<T, float>);
        }
        else if (vtable == py::any_vtable::make<std::int64_t>()) {
            formatters.emplace_back(detail::format_int<T, std::int64_t>);
        }
        else if (vtable == py::any_vtable::make<std::int32_t>()) {
            formatters.emplace_back(detail::format_int<T, std::int32_t>);
        }
        else if (vtable == py::any_vtable::make<std::int16_t>()) {
            formatters.emplace_back(detail::format_int<T, std::int16_t>);
        }
        else if (vtable == py::any_vtable::make<std::int8_t>()) {
            formatters.emplace_back(detail::format_int<T, std::int8_t>);
        }
        else if (vtable == py::any_vtable::make<std::uint64_t>()) {
            formatters.emplace_back(detail::format_int<T, std::uint64_t>);
        }
        else if (vtable == py::any_vtable::make<std::uint32_t>()) {
            formatters.emplace_back(detail::format_int<T, std::uint32_t>);
        }
        else if (vtable == py::any_vtable::make<std::uint16_t>()) {
            formatters.emplace_back(detail::format_int<T, std::uint16_t>);
        }
        else if (vtable == py::any_vtable::make<std::uint8_t>()) {
            formatters.emplace_back(detail::format_int<T, std::uint8_t>);
        }
        else if (vtable == py::any_vtable::make<py::scoped_ref<>>()) {
            formatters.emplace_back(detail::format_pyobject<T>);
        }
        else if (vtable == py::any_vtable::make<py::datetime64<py::chrono::ns>>()) {
            formatters.emplace_back(detail::format_datetime64<T, py::chrono::ns>);
        }
        else if (vtable == py::any_vtable::make<py::datetime64<py::chrono::us>>()) {
            formatters.emplace_back(detail::format_datetime64<T, py::chrono::us>);
        }
        else if (vtable == py::any_vtable::make<py::datetime64<py::chrono::ms>>()) {
            formatters.emplace_back(detail::format_datetime64<T, py::chrono::ms>);
        }
        else if (vtable == py::any_vtable::make<py::datetime64<py::chrono::s>>()) {
            formatters.emplace_back(detail::format_datetime64<T, py::chrono::s>);
        }
        else if (vtable == py::any_vtable::make<py::datetime64<py::chrono::m>>()) {
            formatters.emplace_back(detail::format_datetime64<T, py::chrono::m>);
        }
        else if (vtable == py::any_vtable::make<py::datetime64<py::chrono::h>>()) {
            formatters.emplace_back(detail::format_datetime64<T, py::chrono::h>);
        }
        else if (vtable == py::any_vtable::make<py::datetime64<py::chrono::D>>()) {
            formatters.emplace_back(detail::format_datetime64<T, py::chrono::D>);
        }
        else {
            formatters.emplace_back(detail::format_any<T>);
        }
    }

    return formatters;
}

template<typename T>
void write_header(iobuffer<T>& buf, const std::vector<std::string>& column_names) {
    auto names_it = column_names.begin();
    buf.write(*names_it);
    for (++names_it; names_it != column_names.end(); ++names_it) {
        buf.write(',');
        buf.write(*names_it);
    }
    buf.write('\n');
}

template<typename T>
void write_worker_impl(iobuffer<T>& buf,
                       std::vector<py::array_view<py::any_ref>>& columns,
                       std::int64_t begin,
                       std::int64_t end,
                       const std::vector<format_function<T>>& formatters) {
    for (std::int64_t ix = begin; ix < end; ++ix) {
        auto columns_it = columns.begin();
        auto format_it = formatters.begin();
        (*format_it)(buf, (*columns_it)[ix]);
        for (++columns_it, ++format_it; columns_it != columns.end();
             ++columns_it, ++format_it) {
            buf.write(',');
            (*format_it)(buf, (*columns_it)[ix]);
        }
        buf.write('\n');
    }
}

template<typename T>
void write_worker(std::mutex* exception_mutex,
                  std::vector<std::exception_ptr>* exceptions,
                  iobuffer<T>* buf,
                  std::vector<py::array_view<py::any_ref>>* columns,
                  std::int64_t begin,
                  std::int64_t end,
                  const std::vector<format_function<T>>* formatters) {
    try {
        write_worker_impl<T>(*buf, *columns, begin, end, *formatters);
    }
    catch (const std::exception&) {
        std::lock_guard<std::mutex> guard(*exception_mutex);
        exceptions->emplace_back(std::current_exception());
    }
}

inline PyObject* write_in_memory(const std::vector<std::string>& column_names,
                                 std::vector<py::array_view<py::any_ref>>& columns,
                                 std::size_t buffer_size,
                                 int num_threads,
                                 std::uint8_t float_precision = 10) {
    if (columns.size() != column_names.size()) {
        throw std::runtime_error("mismatched column_names and columns");
    }

    if (!columns.size()) {
        return py::to_object("").escape();
    }

    if (num_threads <= 0) {
        num_threads = 1;
    }

    std::size_t num_rows = columns[0].size();
    auto formatters = get_format_functions<rope_adapter>(columns);

    std::vector<rope_adapter> streams(num_threads);
    {
        std::vector<iobuffer<rope_adapter>> bufs;
        for (auto& stream : streams) {
            bufs.emplace_back(stream, buffer_size, float_precision);
        }

        write_header(bufs[0], column_names);

        if (num_threads <= 1) {
            write_worker_impl(bufs[0], columns, 0, num_rows, formatters);
        }
        else {
            std::mutex exception_mutex;
            std::vector<std::exception_ptr> exceptions;

            std::size_t group_size = num_rows / num_threads + 1;
            std::vector<std::thread> threads;
            for (int n = 0; n < num_threads; ++n) {
                std::int64_t begin = n * group_size;
                threads.emplace_back(std::thread(write_worker<rope_adapter>,
                                                 &exception_mutex,
                                                 &exceptions,
                                                 &bufs[n],
                                                 &columns,
                                                 begin,
                                                 std::min(begin + group_size, num_rows),
                                                 &formatters));
            }

            for (auto& thread : threads) {
                thread.join();
            }

            for (auto& e : exceptions) {
                std::rethrow_exception(e);
            }
        }
    }

    std::vector<std::size_t> sizes;
    std::size_t outsize = 0;
    for (auto& stream : streams) {
        sizes.emplace_back(stream.size());
        outsize += sizes.back();
    }

    char* underlying_buffer;
#if PY_MAJOR_VERSION == 2
    py::scoped_ref out(PyString_FromStringAndSize(nullptr, outsize));
    if (!out) {
        return nullptr;
    }
    underlying_buffer = PyString_AS_STRING(out.get());
#else
    py::scoped_ref out(PyBytes_FromStringAndSize(nullptr, outsize));
    if (!out) {
        return nullptr;
    }
    underlying_buffer = PyBytes_AS_STRING(out.get());
#endif

    std::size_t ix = 0;
    for (auto [stream, size] : py::zip(streams, sizes)) {
        stream.read(underlying_buffer + ix, size);
        ix += size;
    }

    return std::move(out).escape();
}
}  // namespace detail

/** Format a CSV from an array of columns.

    @param stream The ostream to write into.
    @param column_names The names to write into the column header.
    @param columns The arrays of values for each column. Columns are written in the order
                   they appear here, and  must be aligned with `column_names`.
    @param buffer_size The number of bytes to buffer between calls to `stream.write`.
                       This must be a power of 2 greater than or equal to 2 ** 8.
    @param float_precision The number of digits of precision to write floats as.
*/
template<typename T>
void write(T& stream,
           const std::vector<std::string>& column_names,
           std::vector<py::array_view<py::any_ref>>& columns,
           std::size_t buffer_size,
           std::uint8_t float_precision = 10) {
    if (columns.size() != column_names.size()) {
        throw std::runtime_error("mismatched column_names and columns");
    }

    if (!columns.size()) {
        return;
    }

    std::size_t num_rows = columns[0].size();
    auto formatters = detail::get_format_functions<detail::ostream_adapter<T>>(columns);
    detail::ostream_adapter stream_adapter(stream);
    detail::iobuffer<detail::ostream_adapter<T>> buf(stream_adapter,
                                                     buffer_size,
                                                     float_precision);
    detail::write_header(buf, column_names);
    detail::write_worker_impl(buf, columns, 0, num_rows, formatters);
}

/** Format a CSV from an array of columns. This is meant to be exposed to Python with
    `py::automethod`.

    @param file A python object which is either a string to be interpreted as a file name,
                or None, in which case the data will be returned as a Python string.
    @param column_names The names to write into the column header.
    @param columns The arrays of values for each column. Columns are written in the order
                   they appear here, and  must be aligned with `column_names`.
    @return Either the data as a Python string, or None.
*/
inline PyObject* py_write(PyObject*,
                          const py::scoped_ref<>& file,
                          const std::vector<std::string>& column_names,
                          std::vector<py::array_view<py::any_ref>>& columns,
                          std::size_t buffer_size,
                          int num_threads,
                          std::uint8_t float_precision) {
    if (file.get() == Py_None) {
        return detail::write_in_memory(column_names,
                                       columns,
                                       buffer_size,
                                       num_threads,
                                       float_precision);
    }
    else if (num_threads > 1) {
        py::raise(PyExc_ValueError)
            << "cannot pass num_threads > 1 with file-backed output";
        return nullptr;
    }
    else {
        const char* text = py::util::pystring_to_cstring(file);
        if (!text) {
            return nullptr;
        }
        std::ofstream stream(text, std::ios::binary);
        if (!stream) {
            py::raise(PyExc_OSError) << "failed to open file";
            return nullptr;
        }
        write(stream, column_names, columns, buffer_size, float_precision);
        if (!stream) {
            py::raise(PyExc_OSError) << "failed to write csv";
            return nullptr;
        }
        Py_RETURN_NONE;
    }
}
}  // namespace py::csv
