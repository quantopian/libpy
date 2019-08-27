#pragma once

#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

#include "libpy/any.h"
#include "libpy/automethod.h"
#include "libpy/datetime64.h"
#include "libpy/detail/python.h"
#include "libpy/ndarray_view.h"
#include "libpy/numpy_utils.h"

namespace py::csv::writer {
using namespace py::cs::literals;

/** Format a CSV from an array of columns.

    @param stream The ostream to write into.
    @param column_names The names to write into the column header.
    @param columns The arrays of values for each column. Columns are written in the order
                   they appear here, and  must be aligned with `column_names`.
    @param buffer_size The number of bytes to buffer between calls to `stream.write`.
                       This must be a power of 2 greater than or equal to 2 ** 8.
    @param delim The field delimiter.
    @param line_sep The line separator.
    @param float_sigfigs The number of significant figures to print floats with.
    @param preformatted_columns A set of column names which will be object type
           but should not be escaped.
*/
void write(std::ostream& stream,
           const std::vector<std::string>& column_names,
           const std::vector<py::array_view<py::any_cref>>& columns,
           std::size_t buffer_size,
           std::uint8_t float_sigfigs,
           char delim,
           std::string_view line_sep,
           const std::unordered_set<std::string>& preformatted_columns);

/** Format a CSV from an array of columns. This is meant to be exposed to Python with
    `py::automethod`.

    @param file A python object which is either a string to be interpreted as a file name,
                or None, in which case the data will be returned as a Python string.
    @param column_names The names to write into the column header.
    @param columns The arrays of values for each column. Columns are written in the order
                   they appear here, and  must be aligned with `column_names`.
    @param float_sigfigs The number of significant figures to print floats with.
    @param delim The field delimiter.
    @param line_sep The line separator.
    @param preformatted_columns A set of column names which will be object type
           but should not be escaped.
    @return Either the data as a Python string, or None.
*/
PyObject* py_write(
    PyObject*,
    const py::scoped_ref<>& file,
    py::arg::keyword<decltype("column_names"_cs), std::vector<std::string>> column_names,
    py::arg::keyword<decltype("columns"_cs), std::vector<py::array_view<py::any_cref>>>
        columns,
    py::arg::optional<py::arg::keyword<decltype("buffer_size"_cs), std::size_t>>
        buffer_size,
    py::arg::optional<py::arg::keyword<decltype("num_threads"_cs), int>> num_threads,
    py::arg::optional<py::arg::keyword<decltype("float_sigfigs"_cs), std::uint8_t>>
        float_sigfigs,
    py::arg::optional<py::arg::keyword<decltype("delimiter"_cs), char>> delim,
    py::arg::optional<py::arg::keyword<decltype("line_ending"_cs), std::string_view>>
        line_ending,
    py::arg::optional<py::arg::keyword<decltype("preformatted_columns"_cs),
                                       std::unordered_set<std::string>>>
        preformatted_columns);
}  // namespace py::csv::writer
