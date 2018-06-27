#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

#include "libpy/datetime64ns.h"
#include "libpy/exception.h"
#include "libpy/itertools.h"
#include "libpy/numpy_utils.h"
#include "libpy/scoped_ref.h"
#include "libpy/to_object.h"
#include "libpy/valgrind.h"

namespace py::parser {
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

    virtual std::tuple<std::size_t, bool> chomp(const std::string_view&, std::size_t) = 0;

    virtual py::scoped_ref<PyObject> move_to_python_tuple() && {
        Py_INCREF(Py_None);
        return py::scoped_ref(Py_None);
    }
};

template<typename T>
struct typed_cell_parser : public cell_parser {
protected:
    std::vector<T> m_parsed;
    std::vector<py::py_bool> m_mask;

public:
    using type = T;

    typed_cell_parser(char delim) : cell_parser(delim) {}

    const std::vector<T>& parsed() const {
        return m_parsed;
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
class char_array_parser : public typed_cell_parser<std::array<char, n>> {
public:
    char_array_parser(char delim) : typed_cell_parser<std::array<char, n>>(delim) {}

    virtual std::tuple<std::size_t, bool> chomp(const std::string_view& row,
                                                std::size_t offset) {
        this->m_parsed.emplace_back(std::array<char, n>{0});
        auto& cell = this->m_parsed.back();
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
        this->m_mask.emplace_back(cell_ix > 0);
        return ret;
    }
};

class double_parser : public typed_cell_parser<double> {
public:
    double_parser(char delim) : typed_cell_parser<double>(delim) {}

    virtual std::tuple<std::size_t, bool> chomp(const std::string_view& row, std::size_t offset) {
        const char* first = &row.data()[offset];
        char* last;
        this->m_parsed.emplace_back(std::strtod(first, &last));

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

        this->m_mask.emplace_back(size > 0);

        bool more = *last == this->m_delim;
        return {size + more, more};
    }
};

class datetime64ns_parser : public typed_cell_parser<py::datetime64ns> {
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
    datetime64ns_parser(char delim) : typed_cell_parser<py::datetime64ns>(delim) {}

    virtual std::tuple<std::size_t, bool> chomp(const std::string_view& row,
                                                std::size_t offset) {
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
            this->m_parsed.emplace_back();
            this->m_mask.emplace_back(false);
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
        this->m_parsed.emplace_back(std::chrono::system_clock::from_time_t(seconds));
        this->m_mask.emplace_back(true);
        return {consumed, loc};
    }
};

class bool_parser : public typed_cell_parser<py::py_bool> {
public:
    bool_parser(char delim) : typed_cell_parser<py::py_bool>(delim) {}

    virtual std::tuple<std::size_t, bool> chomp(const std::string_view& row,
                                                std::size_t offset) {
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
            this->m_parsed.emplace_back(false);
            this->m_mask.emplace_back(false);
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

        this->m_parsed.emplace_back(value);
        this->m_mask.emplace_back(true);
        return {consumed, loc};
    }
};

class header_parser : public cell_parser {
protected:
    std::vector<std::string> m_parsed;
public:
    header_parser(char delim) : cell_parser(delim) {}

    virtual std::tuple<std::size_t, bool> chomp(const std::string_view& row,
                                                std::size_t offset) {
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

    virtual std::tuple<std::size_t, bool> chomp(const std::string_view& row,
                                                std::size_t offset) {
        return chomp_quoted_string([](char) {}, this->m_delim, row, offset);
    }
};

/** Parse a single row and store the values in the vectors.

    @param row The row index.
    @param data The view over the row to parse.
    @param parsers The cell parsers.
 */
void parse_row(std::size_t row,
               const std::string_view& data,
               std::vector<std::unique_ptr<cell_parser>>& parsers) {

    std::size_t col = 0;
    std::size_t consumed = 0;
    bool more = true;

    for (auto& parser : parsers) {
        if (!more) {
            throw formatted_error("line ",
                                  row,
                                  ": less columns than expected, got ",
                                  col,
                                  " but expected ",
                                  parsers.size());
        }
        try {
            auto [new_consumed, new_more] = parser->chomp(data, consumed);
            consumed += new_consumed;
            more = new_more;
        }
        catch (const std::exception& e) {
            throw position_formatted_error(row, col, e.what());
        }

        ++col;
    }

    if (consumed != data.size()) {
        throw formatted_error("line ",
                              row,
                              ": more columns than expected, expected ",
                              parsers.size());
    }
}

/** Parse a full CSV given the header and the types of the columns.

    @param data The CSV to parse.
    @param parsers The cell parsers.
    @param line_ending The string to split lines on.
 */
void parse_csv_from_header(const std::string_view& data,
                           std::vector<std::unique_ptr<cell_parser>>& parsers,
                           const std::string_view& line_ending) {
    // rows are 1 indexed for csv
    std::size_t row = 1;

    // The current position into the input.
    std::string_view::size_type pos = 0;
    // The index of the next newline.
    std::string_view::size_type end;

    while ((end = data.find(line_ending, pos)) != std::string_view::npos) {
        std::string_view line = data.substr(pos, end - pos);
        parse_row(++row, line, parsers);

        // advance past line ending
        pos = end + line_ending.size();
    }

    std::string_view line = data.substr(pos);
    if (line.size()) {
        // get the rest of the data after the final newline if there is any
        parse_row(++row, line, parsers);
    }
}

/** Check if two dtypes are equal. If a Python exception is raised, throw a C++ exception.

    @param a The first dtype.
    @param b The second dtype.
    @return Are they equal.
 */
bool dtype_eq(PyObject* a, py::scoped_ref<PyArray_Descr>& b) {
    int result = PyObject_RichCompareBool(a, static_cast<PyObject*>(b), Py_EQ);
    if (result < 0) {
        throw std::runtime_error("failed to compare objects");
    }
    return result;
}

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
                    const std::string_view& line_ending) {
    py::valgrind::callgrind profile("parse_csv");

    if (PyDict_Size(dtypes) < 0) {
        // ensure this is a dict with a reasonable error message
        return nullptr;
    }

    auto line_end = data.find(line_ending, 0);
    auto line = data.substr(0, line_end);

    header_parser header_parser(delimiter);

    for (auto [consumed, more] = std::make_tuple(0, true); more;) {
        auto [new_consumed, new_more] = header_parser.chomp(line, consumed);
        consumed += new_consumed;
        more = new_more;
    }

    std::array<py::scoped_ref<PyArray_Descr>, 11> possible_dtypes =
        {py::new_dtype<std::array<char, 1>>(),
         py::new_dtype<std::array<char, 2>>(),
         py::new_dtype<std::array<char, 3>>(),
         py::new_dtype<std::array<char, 4>>(),
         py::new_dtype<std::array<char, 6>>(),
         py::new_dtype<std::array<char, 8>>(),
         py::new_dtype<std::array<char, 9>>(),
         py::new_dtype<std::array<char, 40>>(),
         py::new_dtype<py::datetime64ns>(),
         py::new_dtype<py::py_bool>(),
         py::new_dtype<double>()};

    std::vector<py::scoped_ref<PyObject>> header;
    std::vector<std::unique_ptr<cell_parser>> parsers;

    for (const auto& cell : header_parser.parsed()) {
        auto as_pystring = py::to_object(cell);

        PyObject* dtype = PyDict_GetItem(dtypes, as_pystring.get());
        if (!dtype) {
            parsers.emplace_back(std::make_unique<skip_parser>(delimiter));
        }
        else if (dtype_eq(dtype, possible_dtypes[0])) {
            parsers.push_back(std::make_unique<char_array_parser<1>>(delimiter));
        }
        else if (dtype_eq(dtype, possible_dtypes[1])) {
            parsers.push_back(std::make_unique<char_array_parser<2>>(delimiter));
        }
        else if (dtype_eq(dtype, possible_dtypes[2])) {
            parsers.push_back(std::make_unique<char_array_parser<3>>(delimiter));
        }
        else if (dtype_eq(dtype, possible_dtypes[3])) {
            parsers.push_back(std::make_unique<char_array_parser<4>>(delimiter));
        }
        else if (dtype_eq(dtype, possible_dtypes[4])) {
            parsers.push_back(std::make_unique<char_array_parser<6>>(delimiter));
        }
        else if (dtype_eq(dtype, possible_dtypes[5])) {
            parsers.push_back(std::make_unique<char_array_parser<8>>(delimiter));
        }
        else if (dtype_eq(dtype, possible_dtypes[6])) {
            parsers.push_back(std::make_unique<char_array_parser<9>>(delimiter));
        }
        else if (dtype_eq(dtype, possible_dtypes[7])) {
            parsers.push_back(std::make_unique<char_array_parser<40>>(delimiter));
        }
        else if (dtype_eq(dtype, possible_dtypes[8])) {
            parsers.emplace_back(std::make_unique<datetime64ns_parser>(delimiter));
        }
        else if (dtype_eq(dtype, possible_dtypes[9])) {
            parsers.emplace_back(std::make_unique<bool_parser>(delimiter));
        }
        else if (dtype_eq(dtype, possible_dtypes[10])) {
            parsers.emplace_back(std::make_unique<double_parser>(delimiter));
        }
        else {
            py::raise(PyExc_TypeError) << "unknown dtype: " << dtype;
            throw std::runtime_error("unknown dtype");
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
                          line_ending);

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
