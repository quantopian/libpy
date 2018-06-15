#pragma once

#include <algorithm>
#include <array>
#include <chrono>
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
    return formatted_error(
        "line ", row, " column ", col, ": ", std::forward<Ts>(msg)...);
}

namespace dispatch {
/** Dispatched parser type to allow for partial template specialization.
 */
template<typename T>
struct parse;

template<std::size_t n>
struct parse<std::array<char, n>> {
    static std::array<char, n> f(const std::string_view& data) {
        std::array<char, n> out{0};
        std::copy_n(data.begin(), std::min(out.size(), data.size()), out.begin());
        return out;
    }
};

template<>
struct parse<double> {
    static double f(const std::string_view& data) {
        std::size_t pos;
        double out;
        try {
            out = std::stod(std::string(data), &pos);
        }
        catch (const std::exception&) {
            throw formatted_error("error in stod: ", data);
        }
        if (pos != data.size()) {
            throw formatted_error("invalid digit in double: ", data);
        }
        return out;
    }
};

template<>
struct parse<py::datetime64ns> {
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
    static py::datetime64ns
    f(const std::string_view& raw) {
        if (!raw.size()) {
            return py::datetime64ns{};
        }

        if (raw.size() != 10) {
            throw formatted_error("date string is not exactly 10 characters: ", raw);
        }

        int year = parse_int<4>(raw.data());


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
        return std::chrono::system_clock::from_time_t(seconds);
    }
};

template<>
struct parse<py::py_bool> {
    static py::py_bool
    f(const std::string_view& data) {
        if (data.size() != 1) {
            throw formatted_error("bool is not 0 or 1: ", data);
        }

        if (data[0] == '0') {
            return false;
        }
        else if (data[0] == '1') {
            return true;
        }
        else {
            throw formatted_error("bool is not 0 or 1: ", data);
        }
    }
};
}  // namespace dispatch

/** Parse the data as the given type.

    @tparam T The type to parse as.
    @param row The row index.
    @param col The column index.
    @param data The data to parse.
    @return The parsed value.
 */
template<typename T>
T parse_element(const std::string_view& data) {
    return dispatch::parse<T>::f(data);
}

/** Parse a value and set the mask.

    @param out The output value vector.
    @param mask The output mask.
    @param data The data to parse.
 */
template<typename T>
void parse_element_into_vector(std::vector<T>& out,
                               std::vector<py::py_bool>& mask,
                               const std::string_view& data) {
    if (data.size()) {
        out.emplace_back(parse_element<T>(data));
        mask.emplace_back(true);
    }
    else {
        out.emplace_back(T{});
        mask.emplace_back(false);
    }
}

/** The type of a vector to store parsed values.
 */
using value_vector = std::variant<std::vector<std::array<char, 1>>,
                                  std::vector<std::array<char, 2>>,
                                  std::vector<std::array<char, 3>>,
                                  std::vector<std::array<char, 4>>,
                                  std::vector<std::array<char, 6>>,
                                  std::vector<std::array<char, 8>>,
                                  std::vector<std::array<char, 9>>,
                                  std::vector<std::array<char, 40>>,
                                  std::vector<py::datetime64ns>,
                                  std::vector<py::py_bool>,
                                  std::vector<double>>;

/** The type of the vector of (value, pair) tuples.
 */
using vectors_type = std::vector<std::tuple<value_vector, std::vector<py::py_bool>>>;

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

/** Process cells with escaping.

    @param data The row to process.
    @param delimiter The delimiter between cells.
    @param f The function to call on each unquoted cell.
 */
template<typename F>
void process_quoted_cells(std::size_t row,
                          const std::string_view& data,
                          char delimiter,
                          F&& f) {
    std::string cell;
    quote_state st = quote_state::not_quoted;

    std::size_t started_quote;

    for (std::size_t ix = 0; ix < data.size(); ++ix) {
        char c = data[ix];

        if (c == '\\') {
            if (++ix == data.size()) {
                throw formatted_error("line ",
                                      row,
                                      ": row ends with escape character: ",
                                      data);
            }

            cell.push_back(data[ix]);
            continue;
        }

        switch (st) {
        case quote_state::not_quoted:
            if (c == '"') {
                st = quote_state::quoted;
                started_quote = ix;
            }
            else if (c == delimiter) {
                f(cell);
                cell.clear();
            }
            else {
                cell.push_back(c);
            }
            break;
        case quote_state::quoted:
            if (c == '"') {
                st = quote_state::not_quoted;
            }
            else {
                cell.push_back(c);
            }
        }
    }

    f(cell);

    if (st == quote_state::quoted) {
        std::string underline(started_quote + 2, ' ');
        underline[0] = '\n';
        underline[underline.size() - 1] = '^';
        throw formatted_error("line ",
                              row,
                              ": row ends while quoted, quote begins at index ",
                              started_quote,
                              ":\n",
                              data,
                              underline);
    }
}

/** Parse a single row and store the values in the vectors.

    @param row The row index.
    @param data The view over the row to parse.
    @param delimiter The delimiter between cells.
    @param vectors The output vectors.
 */
void parse_row(std::size_t row,
               const std::string_view& data,
               char delimiter,
               vectors_type& vectors) {

    auto vectors_it = vectors.begin();
    std::size_t col = 0;

    auto callback = [&](const std::string_view& cell) {
        if (vectors_it == vectors.end()) {
            throw formatted_error("line ",
                                  row,
                                  ": more columns than expected, expected ",
                                  vectors.size());
        }

        auto& [boxed_vector, mask] = *vectors_it++;
        std::visit(
            [&](auto& vector) {
                try {
                    parse_element_into_vector(vector, mask, cell);
                }
                catch (const std::exception& e) {
                    throw position_formatted_error(row, col, e.what());
                }

                ++col;
            },
            boxed_vector);
    };

    process_quoted_cells(row, data, delimiter, callback);

    if (vectors_it != vectors.end()) {
        throw formatted_error("line ",
                              row,
                              ": less columns than expected, got ",
                              std::distance(vectors.begin(), vectors_it),
                              " but expected ",
                              vectors.size());
    }
}

/** Parse a full CSV given the header and the types of the columns.

    @param vectors The output vectors.
    @param data The CSV to parse.
    @param delimiter The delimiter between cells in a row.
 */
void parse_csv_from_header(const std::string_view& data,
                           vectors_type& vectors,
                           char delimiter,
                           const std::string_view& line_ending) {
    // rows are 1 indexed for csv
    std::size_t row = 1;

    // The current position into the input.
    std::string_view::size_type pos = 0;
    // The index of the next newline.
    std::string_view::size_type end;

    while ((end = data.find(line_ending, pos)) != std::string_view::npos) {
        std::string_view line = data.substr(pos, end - pos);
        parse_row(++row, line, delimiter, vectors);

        // advance past line ending
        pos = end + line_ending.size();
    }

    std::string_view line = data.substr(pos);
    if (line.size()) {
        // get the rest of the data after the final newline if there is any
        parse_row(++row, line, delimiter, vectors);
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

    @param dtypes A Python dictionary from column name to dtype.
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
    vectors_type vectors;

    auto line_end = data.find(line_ending, 0);
    auto line = data.substr(0, line_end);

    auto callback = [&](const std::string& cell) {
        auto as_pystring = py::to_object(cell);
        PyObject* dtype = PyDict_GetItem(dtypes, as_pystring.get());
        if (!dtype) {
            py::raise(PyExc_KeyError) << as_pystring;
            throw std::runtime_error("unknown column");
        }

        std::vector<py::py_bool> mask;
        if (dtype_eq(dtype, possible_dtypes[0])) {
            vectors.emplace_back(std::vector<std::array<char, 1>>{}, mask);
        }
        else if (dtype_eq(dtype, possible_dtypes[1])) {
            vectors.emplace_back(std::vector<std::array<char, 2>>{}, mask);
        }
        else if (dtype_eq(dtype, possible_dtypes[2])) {
            vectors.emplace_back(std::vector<std::array<char, 3>>{}, mask);
        }
        else if (dtype_eq(dtype, possible_dtypes[3])) {
            vectors.emplace_back(std::vector<std::array<char, 4>>{}, mask);
        }
        else if (dtype_eq(dtype, possible_dtypes[4])) {
            vectors.emplace_back(std::vector<std::array<char, 6>>{}, mask);
        }
        else if (dtype_eq(dtype, possible_dtypes[5])) {
            vectors.emplace_back(std::vector<std::array<char, 8>>{}, mask);
        }
        else if (dtype_eq(dtype, possible_dtypes[6])) {
            vectors.emplace_back(std::vector<std::array<char, 9>>{}, mask);
        }
        else if (dtype_eq(dtype, possible_dtypes[7])) {
            vectors.emplace_back(std::vector<std::array<char, 40>>{}, mask);
        }
        else if (dtype_eq(dtype, possible_dtypes[8])) {
            vectors.emplace_back(std::vector<py::datetime64ns>{}, mask);
        }
        else if (dtype_eq(dtype, possible_dtypes[9])) {
            vectors.emplace_back(std::vector<py::py_bool>{}, mask);
        }
        else if (dtype_eq(dtype, possible_dtypes[10])) {
            vectors.emplace_back(std::vector<double>{}, mask);
        }
        else {
            py::raise(PyExc_TypeError) << "unknown dtype: " << dtype;
            throw std::runtime_error("unknown dtype");
        }

        header.emplace_back(std::move(as_pystring));
    };

    process_quoted_cells(1, line, delimiter, callback);

    Py_ssize_t dtypes_size = PyDict_Size(dtypes);
    if (dtypes_size < 0) {
        return nullptr;
    }
    if (header.size() != static_cast<std::size_t>(dtypes_size)) {
        auto keys = py::scoped_ref(PyDict_Keys(dtypes));
        if (!keys) {
            return nullptr;
        }
        auto pyheader = py::to_object(header);
        if (!pyheader) {
            return nullptr;
        }
        py::raise(PyExc_ValueError)
            << "mismatched headers and dtypes: " << pyheader << " != " << keys;
        return nullptr;
    }

    // advance_past the line ending
    parse_csv_from_header(data.substr(line_end + line_ending.size()),
                          vectors,
                          delimiter,
                          line_ending);

    auto out = py::scoped_ref(PyDict_New());
    if (!out) {
        return nullptr;
    }
    for (auto [name, values_and_mask] : py::zip(header, vectors)) {
        auto& [boxed_vector, mask] = values_and_mask;

        auto values = std::visit(
            [](auto& vector) { return py::move_to_numpy_array(std::move(vector)); },
            boxed_vector);
        if (!values) {
            return nullptr;
        }

        auto mask_array = py::move_to_numpy_array(std::move(mask));
        if (!mask_array) {
            return nullptr;
        }

        auto value = py::scoped_ref(PyTuple_Pack(2, values.get(), mask_array.get()));
        if (!value) {
            return nullptr;
        }

        if (PyDict_SetItem(out.get(), name.get(), value.get())) {
            return nullptr;
        }
    }

    return std::move(out).escape();
}
}  // namespace py::parser
