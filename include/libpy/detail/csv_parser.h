#pragma once

#include <variant>

#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>

#include "libpy/automethod.h"
#include "libpy/datetime64.h"
#include "libpy/exception.h"
#include "libpy/itertools.h"
#include "libpy/numpy_utils.h"
#include "libpy/scope_guard.h"
#include "libpy/scoped_ref.h"
#include "libpy/stream.h"

#if LIBPY_NO_CSV_PREFETCH
#define __builtin_prefetch(...)
#endif

/** The CSV parser is broken up into 2 concepts:

    - `column_spec`
    - `cell_parser`


    ## `column_spec`

    `column_spec` is a user-provides type which specifies information about both
    the type of a column as well as information about how the cells should be
    parsed. Information about how cells should be parsed includes things like a
    date columns format (YYYY-MM-DD, MM/DD/YYYY, etc.) or a boolean's format
    (0/1, true/false, etc.).

    `column_spec` objects are like numpy dtypes. `column_spec` objects can be
    reused across multiple parses, for potentially multiple files. `column_spec`
    objects are stateless.

    A `column_spec` is responsible for creating the `cell_parser` for a
    particular parsing run.

    ## `cell_parser`

    `cell_parser` is a lower-level type which implements the concrete parsing of
    individual cells. A `cell_parser` is stateful, and manages the state of a
    given parser.

    # Concepts

    `column_spec` should read whatever configuration it has been given and
    allocate a specialized `cell_parser` object for the particular
    configuration. `cell_parser` objects should not try to read configuration on
    a per-cell basis because this will hurt performance.
 */
namespace py::csv::parser {
/** Exception used to indicate that parsing the CSV failed for some reason.
 */
class parse_error : public std::runtime_error {
public:
    inline parse_error(const std::string& msg) : std::runtime_error(msg) {}
};

class cell_parser;

/** A column type is a specification for both the type and the parsing rules for
    a column of a CSV. `column_spec` objects are responsible for constructing
    `cell_parser` objects for a given parsing run.
 */
class column_spec {
public:
    virtual ~column_spec() = default;

    struct alloc_info {
        std::size_t size;
        std::size_t align;

        template<typename T>
        static constexpr alloc_info make() {
            return {sizeof(T), alignof(T)};
        }
    };

    /** Return the size and alignement of the cell parser objects returned by
        this column type.

        @return The `sizeof` and `alignof` the concrete cell_parser subclass
                required by this column.
     */
    virtual alloc_info cell_parser_alloc_info() const = 0;

    /** Placement new a new cell parser for this column.

        @param storage Storage with at least `cell_parser_size()` bytes
               allocated to store a cell_parser for a given parse.
        @param num_rows The number of rows in the csv which the `cell_parser`
               will parse. This should be used to pre-allocate space for
               parsing into.
        @return A pointer to the new `cell_parser`. This pointer should be the
                same address as `storage`, but should point to the newly
                constructed object.
     */
    virtual cell_parser* emplace_cell_parser(void* storage) const = 0;
};

/** A cell parser is an object that represents the parse state for a single
    column in a given CSV.

    Subclasses are required to implement `chomp``.
 */
class cell_parser {
public:
    virtual ~cell_parser() = default;

    /** Set the line count. This should pre-allocate space for `num_lines` values to be
        parsed.
     */
    virtual void set_num_lines(std::size_t);

    /** Initialize any data that depends on the number of threads.
     */
    virtual void set_num_threads(int);

    /** "chomp" text from a row and parse the given cell.

        @param delim The delimiter.
        @param row_ix The row number (0-indexed) being parsed.
        @param row The entire row being parsed.
        @param offset The offset into `row` where the cell starts.
        @param thread_ix The thread index. This is used when a parser requires some thread
               local storage.
        @return A tuple of the number of characters consumed from the row and a boolean
                which is true if we expect there to be more columns to parse in this row.
     */
    virtual std::tuple<std::size_t, bool> chomp(char delim,
                                                std::size_t row_ix,
                                                std::string_view row,
                                                std::size_t offset,
                                                int thread_ix) = 0;

    /** Move the state of this column to a Python tuple of numpy arrays.

        If this parser doesn't produce values, it just returns a new reference to
        `Py_None`.
     */
    virtual py::scoped_ref<> move_to_python_tuple() && {
        Py_INCREF(Py_None);
        return py::scoped_ref(Py_None);
    }
};

/** Parser for skipping a column.
 */
class skip_parser : public cell_parser {
public:
    std::tuple<std::size_t, bool> chomp(char delim,
                                        std::size_t,
                                        std::string_view row,
                                        std::size_t offset,
                                        int) override;
};

/** Base class for cell parsers that produce statically typed vectors of values.
 */
template<typename T>
class typed_cell_parser : public cell_parser {
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

private:
    struct has_to_numpy_array {
    private:
        template<typename U>
        static decltype(py::dispatch::new_dtype<U>::get(), std::true_type{}) test(int);

        template<typename>
        static std::false_type test(long);

    public:
        static constexpr bool value =
            std::is_same_v<decltype(test<T>(0)), std::true_type>;
    };

public:
    py::scoped_ref<> move_to_python_tuple() && override {
        if constexpr (!has_to_numpy_array::value) {
            Py_INCREF(Py_None);
            return py::scoped_ref(Py_None);
        }
        else {
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
    }

    std::tuple<std::vector<T>, std::vector<py::py_bool>> move_to_tuple() && {
        return {std::move(m_parsed), std::move(m_mask)};
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
class fundamental_parser : public typed_cell_parser<parse_result<scalar_parse>> {
public:
    using type = typename typed_cell_parser<parse_result<scalar_parse>>::type;

    std::tuple<std::size_t, bool> chomp(char delim,
                                        std::size_t ix,
                                        std::string_view row,
                                        std::size_t offset,
                                        int) override {
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
            throw util::formatted_error<parse_error>("invalid digit in ",
                                                     py::util::type_name<type>().get(),
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

/** Adapt a fundamental parser function which only handle positive values to
    accept positive or negative values.
 */
template<auto F>
auto signed_adapter(const char* ptr, const char** last) -> decltype(F(ptr, last)) {
    bool negate = *ptr == '-';
    if (negate) {
        ++ptr;
        return -F(ptr, last);
    }
    return F(ptr, last);
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
}  // namespace detail

/** A faster, but less accepting, implementation of `strtol`.
    @tparam T The type of integer to parse as. This should be a signed integral type.
    @param ptr The beginning of the string to parse.
    @param last An output argument to take a pointer to the first character not parsed.
    @return As much of `ptr` parsed as a `T` as possible.
 */
template<typename T,
         typename = std::enable_if_t<std::is_integral_v<T> && std::is_signed_v<T>>>
auto fast_strtol = detail::signed_adapter<detail::fast_unsigned_strtol<T>>;

class int8_parser : public detail::fundamental_parser<fast_strtol<std::int8_t>> {};
class int16_parser : public detail::fundamental_parser<fast_strtol<std::int16_t>> {};
class int32_parser : public detail::fundamental_parser<fast_strtol<std::int32_t>> {};
class int64_parser : public detail::fundamental_parser<fast_strtol<std::int64_t>> {};

namespace detail {
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
    std::size_t whole_part = 0;
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
                exponent = -exponent;
            }
            result *= std::pow(10, exponent);
            return result;
        }

        exponent *= 10;
        exponent += value;
        ++ptr;
    }
}

/** A wrapper around `std::strtod` to give it the same interface as `fast_strtod`.
    @tparam T Either `float` or `double` to switch between `std::strtof` and
              `std::strtod`.
    @param ptr The beginning of the string to parse.
    @param last An output argument to take a pointer to the first character not parsed.
    @return As much of `ptr` parsed as a double as possible.
 */
template<typename T>
T precise_strtod(const char* ptr, const char** last) {
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
}  // namespace detail

class precise_float32_parser
    : public detail::float_parser<detail::precise_strtod<float>> {};
class precise_float64_parser
    : public detail::float_parser<detail::precise_strtod<double>> {};

class fast_float32_parser : public detail::float_parser<detail::fast_strtod<float>> {};
class fast_float64_parser : public detail::float_parser<detail::fast_strtod<double>> {};

namespace detail {
struct pcre2_code_deleter {
    inline void operator()(pcre2_code* p) const noexcept {
        pcre2_code_free(p);
    }
};

using pcre2_code_ptr = std::unique_ptr<pcre2_code, pcre2_code_deleter>;

struct pcre2_match_context_deleter {
    inline void operator()(pcre2_match_context* p) const noexcept {
        pcre2_match_context_free(p);
    }
};

using pcre2_match_context_ptr =
    std::unique_ptr<pcre2_match_context, pcre2_match_context_deleter>;

struct pcre2_match_data_deleter {
    inline void operator()(pcre2_match_data* p) const noexcept {
        pcre2_match_data_free(p);
    }
};

using pcre2_match_data_ptr = std::unique_ptr<pcre2_match_data, pcre2_match_data_deleter>;

inline std::tuple<std::string_view, std::size_t, bool>
isolate_unquoted_cell(std::string_view row, std::size_t offset, char delim) {
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

class initialize_runtime_format_datetime_parser_formats;
}  // namespace detail

class runtime_format_datetime_parser : public typed_cell_parser<py::datetime64ns> {
private:
    struct thread_state {
        detail::pcre2_match_context_ptr match_context;
        detail::pcre2_match_data_ptr match_data;
    };

    enum class parse_func {
        date,
        datetime,
        datetime_tz,
    };

private:
    friend class detail::initialize_runtime_format_datetime_parser_formats;

    // global map from format string to regex
    static std::unordered_map<std::string, detail::pcre2_code_ptr> format_map;

private:
    // borrowed reference out of `format_map`
    const pcre2_code* m_code;
    std::vector<thread_state> m_thread_state;

    int m_year_group;
    int m_month_group;
    int m_day_group;
    int m_hour_group;
    int m_min_group;
    int m_sec_group;
    int m_frac_sec_group;
    int m_offset_sign_group;
    int m_offset_h_group;
    int m_offset_m_group;

    parse_func m_parse_func;

    py::datetime64ns parse_date(std::size_t* ovector, std::string_view cell) const;
    py::datetime64ns parse_datetime(std::size_t* ovector, std::string_view cell) const;
    py::datetime64ns parse_datetime_tz(std::size_t* ovector, std::string_view cell) const;

    friend class datetime_column;

    /** resolve a string into a borrowed reference to the compiled pattern.

        @param format The format string.
        @return The compiled pattern. If no such pattern exists, an exception is thrown.
    */
    static const pcre2_code* resolve_format(const std::string& format);

    // private constructor for `datetime_column` to use
    runtime_format_datetime_parser(const pcre2_code* pattern);

public:
    runtime_format_datetime_parser(const std::string& format);

    void set_num_threads(int) override;

    std::tuple<std::size_t, bool> chomp(char delim,
                                        std::size_t ix,
                                        std::string_view row,
                                        std::size_t offset,
                                        int thread_ix) override;
};

namespace detail {
template<typename falses, typename trues>
class single_char_bool_parser : public typed_cell_parser<py_bool> {
private:
    template<char... cs>
    bool contains(py::cs::char_sequence<cs...>, char c) {
        return ((c == cs) || ...);
    }

public:
    std::tuple<std::size_t, bool> chomp(char delim,
                                        std::size_t ix,
                                        std::string_view row,
                                        std::size_t offset,
                                        int) override {
        auto [raw, consumed, more] = detail::isolate_unquoted_cell(row, offset, delim);
        if (raw.size() == 0) {
            return {consumed, more};
        }

        auto raise_invalid = [&]() {
            std::string valid_values;
            for (char c : py::cs::to_array(falses{})) {
                if (c) {
                    valid_values.push_back('"');
                    valid_values.push_back(c);
                    valid_values.push_back('"');
                    valid_values.push_back(',');
                    valid_values.push_back(' ');
                }
            }
            for (char c : py::cs::to_array(trues{})) {
                if (c) {
                    valid_values.push_back('"');
                    valid_values.push_back(c);
                    valid_values.push_back('"');
                    valid_values.push_back(',');
                    valid_values.push_back(' ');
                }
            }
            valid_values.pop_back();
            valid_values.pop_back();
            throw util::formatted_error<parse_error>("invalid boolean value: \"",
                                                     raw,
                                                     "\", valid values: {",
                                                     valid_values,
                                                     '}');
        };
        if (raw.size() != 1) {
            raise_invalid();
        }

        bool value;
        if (contains(falses{}, raw[0])) {
            value = false;
        }
        else if (contains(trues{}, raw[0])) {
            value = true;
        }
        else {
            raise_invalid();
        }

        this->m_parsed[ix] = value;
        this->m_mask[ix] = true;
        return {consumed, more};
    }
};
}  // namespace detail

using bool_01_parser =
    detail::single_char_bool_parser<cs::char_sequence<'0'>, cs::char_sequence<'1'>>;

using bool_ft_parser =
    detail::single_char_bool_parser<cs::char_sequence<'f'>, cs::char_sequence<'t'>>;

using bool_FT_parser =
    detail::single_char_bool_parser<cs::char_sequence<'F'>, cs::char_sequence<'T'>>;

class bool_ft_case_insensitive_parser
    : public detail::single_char_bool_parser<cs::char_sequence<'f', 'F'>,
                                             cs::char_sequence<'t', 'T'>> {};

namespace detail {
template<typename false_cs, typename true_cs>
class multi_char_bool_parser : public typed_cell_parser<py::py_bool> {
private:
    constexpr static std::array false_arr = py::cs::to_array(false_cs{});
    constexpr static std::string_view false_view = std::string_view{false_arr.data(),
                                                                    false_arr.size() - 1};

    constexpr static std::array true_arr = py::cs::to_array(true_cs{});
    constexpr static std::string_view true_view = std::string_view{true_arr.data(),
                                                                   true_arr.size() - 1};

public:
    std::tuple<std::size_t, bool> chomp(char delim,
                                        std::size_t ix,
                                        std::string_view row,
                                        std::size_t offset,
                                        int) override {
        auto [raw, consumed, more] = detail::isolate_unquoted_cell(row, offset, delim);
        if (raw.size() == 0) {
            return {consumed, more};
        }

        bool value;
        if (raw == false_view) {
            value = false;
        }
        else if (raw == true_view) {
            value = true;
        }
        else {
            throw util::formatted_error<parse_error>("invalid boolean value: \"",
                                                     raw,
                                                     "\", valid values: {\"",
                                                     false_view,
                                                     "\", \"",
                                                     true_view,
                                                     "\"}");
        }

        this->m_parsed[ix] = value;
        this->m_mask[ix] = true;
        return {consumed, more};
    }

};
}  // namespace detail

using namespace py::cs::literals;

using bool_FALSE_TRUE_parser =
    detail::multi_char_bool_parser<decltype("FALSE"_cs), decltype("TRUE"_cs)>;

using bool_False_True_parser =
    detail::multi_char_bool_parser<decltype("False"_cs), decltype("True"_cs)>;

class runtime_fixed_width_string_parser : public cell_parser {
private:
    std::size_t m_itemsize;
    std::vector<char> m_parsed;
    std::vector<py::py_bool> m_mask;

public:
    runtime_fixed_width_string_parser(std::size_t itemsize);

    virtual void set_num_lines(std::size_t nrows) override;

    virtual std::tuple<std::size_t, bool> chomp(char delim,
                                                std::size_t ix,
                                                std::string_view row,
                                                std::size_t offset,
                                                int) override;

    virtual py::scoped_ref<> move_to_python_tuple() && override;
};

namespace detail {
enum class quote_state {
    quoted,
    not_quoted,
};

template<typename F>
std::tuple<bool, std::tuple<std::size_t, bool>>
chomp_quoted_string(F&& f, char delim, std::string_view row, std::size_t offset) {
    quote_state st = quote_state::not_quoted;
    std::size_t started_quote;

    auto cell = row.substr(offset);

    std::size_t ix;
    for (ix = 0; ix < cell.size(); ++ix) {
        char c = cell[ix];

        if (c == '\\') {
            if (++ix == cell.size()) {
                throw util::formatted_error<parse_error>(
                    "line ", cell, ": row ends with escape character: ", row);
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
                return {ix > 0, {ix + 1, true}};
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
        throw util::formatted_error<parse_error>(
            "row ends while quoted, quote begins at index ",
            started_quote,
            ":\n",
            row,
            underline);
    }

    return {ix > 0, {ix, false}};
}
}  // namespace detail

template<std::size_t itemsize>
class fixed_width_string_parser : public typed_cell_parser<std::array<char, itemsize>> {
public:
    std::tuple<std::size_t, bool> chomp(char delim,
                                        std::size_t ix,
                                        std::string_view row,
                                        std::size_t offset,
                                        int) override {
        auto& cell = this->m_parsed[ix];
        std::size_t cell_ix = 0;
        auto [mask, ret] = detail::chomp_quoted_string(
            [&](char c) {
                if (cell_ix < cell.size()) {
                    cell[cell_ix++] = c;
                }
            },
            delim,
            row,
            offset);
        this->m_mask[ix] = mask;
        return ret;
    }
};

class vlen_string_parser : public typed_cell_parser<std::string> {
public:
    virtual std::tuple<std::size_t, bool> chomp(char delim,
                                                std::size_t ix,
                                                std::string_view row,
                                                std::size_t offset,
                                                int) override;

    virtual py::scoped_ref<> move_to_python_tuple() && override;
};

template<typename Parser>
class simple_column_spec : public column_spec {
public:
    virtual column_spec::alloc_info cell_parser_alloc_info() const override {
        return column_spec::alloc_info::make<Parser>();
    }

    virtual cell_parser* emplace_cell_parser(void* addr) const override {
        return new (addr) Parser{};
    }
};

class composed_column_spec : public column_spec {
protected:
    std::unique_ptr<column_spec> m_option;

public:
    composed_column_spec() = default;
    composed_column_spec(std::unique_ptr<column_spec>&& option);

    virtual column_spec::alloc_info cell_parser_alloc_info() const override;
    virtual cell_parser* emplace_cell_parser(void* addr) const override;
};

/** Column spec for indicating that a column should be skipped.
 */
using skip_column = simple_column_spec<skip_parser>;

using int8_column = simple_column_spec<int8_parser>;
using int16_column = simple_column_spec<int16_parser>;
using int32_column = simple_column_spec<int32_parser>;
using int64_column = simple_column_spec<int64_parser>;

using precise_float32_column = simple_column_spec<precise_float32_parser>;
using fast_float32_column = simple_column_spec<fast_float32_parser>;
class float32_column : public composed_column_spec {
public:
    float32_column(std::string_view);
};

using precise_float64_column = simple_column_spec<precise_float64_parser>;
using fast_float64_column = simple_column_spec<fast_float64_parser>;
class float64_column : public composed_column_spec {
public:
    float64_column(std::string_view);
};

class datetime_column : public column_spec {
private:
    // borrowed reference into `runtime_format_datetime_parser::format_map`
    const pcre2_code* m_code;

public:
    datetime_column(std::string_view);

    column_spec::alloc_info cell_parser_alloc_info() const override;
    cell_parser* emplace_cell_parser(void* addr) const override;
};

class bool_column : public composed_column_spec {
public:
    bool_column(std::string_view);
};

class string_column : public column_spec {
private:
    std::int64_t m_size;

public:
    string_column(std::int64_t size);

    column_spec::alloc_info cell_parser_alloc_info() const override;
    cell_parser* emplace_cell_parser(void*) const override;
};

using namespace py::cs::literals;

/** CSV parsing function.
    @param data The string data to parse as a CSV.
    @param types A mapping from column name to a cell parser for the given column. The
                 parsers will be updated in place with the parsed data.
    @param delimiter The delimiter between cells.
    @param line_ending The separator between line.
 */
void parse(std::string_view data,
           const std::unordered_map<std::string, std::shared_ptr<cell_parser>>& types,
           char delimiter,
           std::string_view line_ending,
           std::size_t num_threads);

/** Python CSV parsing function.

    @param data The string data to parse as a CSV.
    @param columns A Python dictionary from column name to column spec.
           Columns not present are ignored.
    @param delimiter The delimiter between cells.
    @param line_ending The separator between line.
    @return A Python dictionary from column name to a tuple of (value, mask)
   arrays.
 */
PyObject*
py_parse(PyObject*,
         std::string_view data,
         py::arg::keyword<decltype("column_specs"_cs), PyObject*> column_specs,
         py::arg::keyword<decltype("delimiter"_cs), char> delimiter,
         py::arg::keyword<decltype("line_ending"_cs), std::string_view> line_ending,
         py::arg::keyword<decltype("num_threads"_cs), std::size_t> num_threads);

/** Add all of the Python column spec types to a module.

    @param module The module to add the objects to.
    @return True with a Python exception set on failure, otherwise false.
 */
bool add_parser_pytypes(PyObject* module);
}  // namespace py::csv::parser
