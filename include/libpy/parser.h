#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <exception>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <type_traits>
#include <variant>
#include <vector>

#include "libpy/datetime64ns.h"
#include "libpy/exception.h"
#include "libpy/gil.h"
#include "libpy/itertools.h"
#include "libpy/numpy_utils.h"
#include "libpy/scoped_ref.h"
#include "libpy/to_object.h"
#include "libpy/valgrind.h"

namespace py::parser {
constexpr std::size_t min_group_size = 4096;

/** The types that can be parsed. To add support for a new type, just add it to this
    tuple.
 */
using possible_types = std::tuple<std::array<char, 1>,
                                  std::array<char, 2>,
                                  std::array<char, 3>,
                                  std::array<char, 4>,
                                  std::array<char, 6>,
                                  std::array<char, 8>,
                                  std::array<char, 9>,
                                  std::array<char, 12>,
                                  std::array<char, 30>,
                                  std::array<char, 40>,
                                  py::datetime64ns,
                                  py::py_bool,
                                  double>;

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

class cell_parser {
protected:
    char m_delim;
public:
    cell_parser(char delim) : m_delim(delim) {}

    virtual void set_num_lines(std::size_t) {}

    virtual std::tuple<std::size_t, bool>
    chomp(std::size_t, const std::string_view&, std::size_t) = 0;

    virtual py::scoped_ref<PyObject> move_to_python_tuple() && {
        Py_INCREF(Py_None);
        return py::scoped_ref(Py_None);
    }
};

template<typename T>
struct typed_cell_parser_base : public cell_parser {
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
};

template<typename T>
struct typed_cell_parser;

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

template<std::size_t n>
class typed_cell_parser<std::array<char, n>>
    : public typed_cell_parser_base<std::array<char, n>> {
public:
    typed_cell_parser(char delim) : typed_cell_parser_base<std::array<char, n>>(delim) {}

    virtual std::tuple<std::size_t, bool>
    chomp(std::size_t ix, const std::string_view& row, std::size_t offset) {
        auto& cell = this->m_parsed[ix];
        std::size_t cell_ix = 0;
        auto ret = chomp_quoted_string(
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

template<>
class typed_cell_parser<double> : public typed_cell_parser_base<double> {
public:
    typed_cell_parser(char delim) : typed_cell_parser_base<double>(delim) {}

    virtual std::tuple<std::size_t, bool>
    chomp(std::size_t ix, const std::string_view& row, std::size_t offset) {
        const char* first = &row.data()[offset];
        char* last;
        this->m_parsed[ix] = std::strtod(first, &last);

        std::size_t size = last - first;
        if (*last != this->m_delim && size != row.size() - offset) {
            std::string_view cell;
            auto end = std::memchr(first, this->m_delim, row.size() - offset);
            if (end) {
                cell = row.substr(offset, reinterpret_cast<const char*>(end) - first);
            }
            else {
                cell = row.substr(offset);
            }
            throw formatted_error("invalid digit in double: ", cell);
        }

        this->m_mask[ix] = size > 0;

        bool more = *last == this->m_delim;
        return {size + more, more};
    }
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
                throw formatted_error("invalid digit in int: ", cs.substr(ndigits));
            }

            result *= 10;
            result += c;
        }
        return result;
    }

public:
    typed_cell_parser(char delim) : typed_cell_parser_base<py::datetime64ns>(delim) {}

    virtual std::tuple<std::size_t, bool>
    chomp(std::size_t ix, const std::string_view& row, std::size_t offset) {
        std::string_view raw;

        auto subrow = row.substr(offset);
        const void* loc = std::memchr(subrow.data(), this->m_delim, subrow.size());
        std::size_t size;
        std::size_t consumed;
        if (loc) {
            size = reinterpret_cast<const char*>(loc) - subrow.data();
            consumed = size + 1;
        }
        else {
            size = consumed = subrow.size();
        }

        raw = subrow.substr(0, size);
        if (!raw.size()) {
            return {consumed, loc};
        }

        if (raw.size() != 10) {
            throw formatted_error("date string is not exactly 10 characters: ", raw);
        }

        int year = parse_int<4>(raw);

        if (raw[4] != '-') {
            throw formatted_error("expected hyphen at index 4: ", raw);
        }

        int month = parse_int<2>(raw.substr(5)) - 1;
        if (month < 0 || month > 11) {
            throw formatted_error("month not in range [1, 12]: ", raw);
        }

        if (raw[7] != '-') {
            throw formatted_error("expected hyphen at index 7: ", raw);
        }

        int leap_day = month == 1 && is_leapyear(year);

        // we subtract one to zero index day
        int day = parse_int<2>(raw.substr(8)) - 1;
        int max_day = days_in_month[month] + leap_day - 1;
        if (day < 0 || day > max_day) {
            throw formatted_error(
                "day out of bounds for month (max=", max_day + 1, "): ", raw);
        }

        time_t seconds = to_unix_time(year, month, day);

        // adapt to a `std::chrono::time_point` to properly handle time unit
        // conversions and time since epoch
        this->m_parsed[ix] = std::chrono::system_clock::from_time_t(seconds);
        this->m_mask[ix] = true;
        return {consumed, loc};
    }
};

template<>
class typed_cell_parser<py::py_bool> : public typed_cell_parser_base<py::py_bool> {
public:
    typed_cell_parser(char delim) : typed_cell_parser_base<py::py_bool>(delim) {}

    virtual std::tuple<std::size_t, bool>
    chomp(std::size_t ix, const std::string_view& row, std::size_t offset) {
        std::size_t size;
        std::size_t consumed;
        auto subrow = row.substr(offset);
        const void* loc = std::memchr(subrow.data(), this->m_delim, subrow.size());
        if (loc) {
            size = reinterpret_cast<const char*>(loc) - subrow.data();
            consumed = size + 1;
        }
        else {
            size = consumed = subrow.size();
        }

        if (size == 0) {
            return {consumed, loc};
        }
        if (size != 1) {
            throw formatted_error("bool is not 0 or 1: ", subrow.substr(0, size));
        }

        bool value;
        if (subrow[0] == '0') {
            value = false;
        }
        else if (subrow[0] == '1') {
            value = true;
        }
        else {
            throw formatted_error("bool is not 0 or 1: ", subrow[0]);
        }

        this->m_parsed[ix] = value;
        this->m_mask[ix] = true;
        return {consumed, loc};
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
        return chomp_quoted_string([&](char c) { cell.push_back(c); },
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
        return chomp_quoted_string([](char) {}, this->m_delim, row, offset);
    }
};

using parser_types = std::vector<std::unique_ptr<cell_parser>>;

/** Parse a single row and store the values in the vectors.

    @param row The row index.
    @param data The view over the row to parse.
    @param parsers The cell parsers.
 */
void parse_row(std::size_t row,
               const std::string_view& data,
               parser_types& parsers) {

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

void parse_lines(std::string_view* begin,
                 std::string_view* end,
                 std::size_t offset,
                 parser_types& parsers) {
    for (std::size_t ix = offset; begin != end; ++begin, ++ix) {
        parse_row(ix, *begin, parsers);
    }
}


void parse_lines_worker(std::mutex* exception_mutex,
                        std::vector<std::exception_ptr>* exceptions,
                        std::string_view* begin,
                        std::string_view* end,
                        std::size_t offset,
                        parser_types* parsers) {
    try {
        parse_lines(begin, end, offset, *parsers);
    }
    catch (const std::exception&) {
        std::lock_guard<std::mutex> guard(*exception_mutex);
        exceptions->emplace_back(std::current_exception());
    }
}

/** Parse a full CSV given the header and the types of the columns.

    @param data The CSV to parse.
    @param parsers The cell parsers.
    @param line_ending The string to split lines on.
 */
void parse_csv_from_header(const std::string_view& data,
                           parser_types& parsers,
                           const std::string_view& line_ending,
                           std::size_t num_threads) {
    // The current position into the input.
    std::string_view::size_type pos = 0;
    // The index of the next newline.
    std::string_view::size_type end;

    std::vector<std::string_view> lines;
    lines.reserve(min_group_size);

    while ((end = data.find(line_ending, pos)) != std::string_view::npos) {
        lines.emplace_back(data.substr(pos, end - pos));

        // advance past line ending
        pos = end + line_ending.size();
    }
    if (pos != data.size()) {
        // add any data after the last newline if there is anything to add
        lines.emplace_back(data.substr(pos));
    }

    for (auto& parser : parsers) {
        parser->set_num_lines(lines.size());
    }

    std::size_t group_size = lines.size() / num_threads;
    if (group_size < min_group_size) {
        parse_lines(&lines[0], &lines[lines.size()], 0, parsers);
    }
    else {
        std::mutex exception_mutex;
        std::vector<std::exception_ptr> exceptions;

        std::vector<std::thread> threads;
        std::size_t n;
        for (n = 0; n < num_threads - 1; ++n) {
            std::size_t start = n * group_size;
            threads.emplace_back(
                std::thread(parse_lines_worker,
                            &exception_mutex,
                            &exceptions,
                            &lines[start],
                            &lines[start + group_size],
                            start,
                            &parsers));
        }
        std::size_t start = n * group_size;
        threads.emplace_back(
            std::thread(parse_lines_worker,
                        &exception_mutex,
                        &exceptions,
                        &lines[start],
                        &lines[std::max(start + group_size, lines.size())],
                        start,
                        &parsers));

        for (auto& thread : threads) {
            thread.join();
        }

        for (auto& e : exceptions) {
            std::rethrow_exception(e);
        }
    }
}

namespace detail {
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

    static void
    create_vector(char delimiter, parser_types& parsers) {
        parsers.emplace_back(std::make_unique<typed_cell_parser<T>>(delimiter));
    }
};

template<typename T>
struct initialize_parser;

template<typename T, typename... Ts>
struct initialize_parser<std::tuple<T, Ts...>> {
    static void f(PyObject* dtype, char delimiter, parser_types& parsers) {
        using option = dtype_option<T>;
        if (!option::matches(dtype)) {
            initialize_parser<std::tuple<Ts...>>::f(dtype, delimiter, parsers);
            return;
        }

        option::create_vector(delimiter, parsers);
    }
};

template<>
struct initialize_parser<std::tuple<>> {
    static void f(PyObject*, char, parser_types&) {
        throw std::runtime_error("unknown dtype");
    }
};
}  // namespace detail

/** Python CSV parsing function.

    @param dtypes A Python dictionary from column name to dtype. Columns not present are
   ignored.
    @param data The string data to parse as a CSV.
    @param delimiter The delimiter between cells.
    @param line_ending The separator between line.
    @return A Python dictionary from column name to a tuple of (value, mask) arrays.
 */
PyObject* parse_csv(PyObject*,
                    const std::string_view& data,
                    PyObject* dtypes,
                    char delimiter,
                    const std::string_view& line_ending,
                    std::size_t num_threads) {
    py::valgrind::callgrind profile("parse_csv");

    if (PyDict_Size(dtypes) < 0) {
        // ensure this is a dict with a reasonable error message
        return nullptr;
    }

    auto line_end = data.find(line_ending, 0);
    auto line = data.substr(0, line_end);

    header_parser header_parser(delimiter);

    for (auto [consumed, more] = std::make_tuple(0, true); more;) {
        auto [new_consumed, new_more] = header_parser.chomp(0, line, consumed);
        consumed += new_consumed;
        more = new_more;
    }

    std::vector<py::scoped_ref<PyObject>> header;
    parser_types parsers;

    for (const auto& cell : header_parser.parsed()) {
        auto as_pystring = py::to_object(cell);

        PyObject* dtype = PyDict_GetItem(dtypes, as_pystring.get());
        if (!dtype) {
            parsers.emplace_back(std::make_unique<skip_parser>(delimiter));
        }
        else {
            // push back a new parser of the proper static type into `parsers` given the
            // runtime `dtype`
            detail::initialize_parser<possible_types>::f(dtype, delimiter, parsers);
        }

        header.emplace_back(std::move(as_pystring));
    };

    // Ensure that each key in dtypes has a corresponding column in the CSV header.
    auto expected_keys = py::scoped_ref(PySet_New(dtypes));
    if (!expected_keys) {
        return nullptr;
    }
    auto actual_keys = py::to_object(header);
    if (!actual_keys) {
        return nullptr;
    }
    auto actual_keys_set = py::scoped_ref(PySet_New(actual_keys.get()));
    if (!actual_keys_set) {
        return nullptr;
    }
    auto diff = py::scoped_ref(
        PyNumber_Subtract(expected_keys.get(), actual_keys_set.get()));
    if (!diff) {
        return nullptr;
    }
    if (PySet_GET_SIZE(diff.get())) {
        auto as_list = py::scoped_ref(PySequence_List(diff.get()));
        if (PyList_Sort(as_list.get()) < 0) {
            return nullptr;
        }
        py::raise(PyExc_ValueError) << "dtype keys not present in header: " << as_list
                                    << "\nheader: " << actual_keys;
        return nullptr;
    }

    // advance_past the line ending
    parse_csv_from_header(data.substr(line_end + line_ending.size()),
                          parsers,
                          line_ending,
                          num_threads);

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
}  // namespace py::parser
