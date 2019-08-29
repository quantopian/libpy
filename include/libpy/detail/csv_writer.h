#pragma once

#include <iostream>
#include <string>
#include <vector>

#include "libpy/any.h"
#include "libpy/datetime64.h"
#include "libpy/detail/python.h"
#include "libpy/ndarray_view.h"
#include "libpy/numpy_utils.h"

namespace py::csv::writer {
/** Format a CSV from an array of columns.

    @param stream The ostream to write into.
    @param column_names The names to write into the column header.
    @param columns The arrays of values for each column. Columns are written in the order
                   they appear here, and  must be aligned with `column_names`.
    @param buffer_size The number of bytes to buffer between calls to `stream.write`.
                       This must be a power of 2 greater than or equal to 2 ** 8.
    @param float_sigfigs The number of significant figures to print floats with.
*/
void write(std::ostream& stream,
           const std::vector<std::string>& column_names,
           const std::vector<py::array_view<py::any_cref>>& columns,
           std::size_t buffer_size,
           std::uint8_t float_sigfigs);

/** Format a CSV from an array of columns. This is meant to be exposed to Python with
    `py::automethod`.

    @param file A python object which is either a string to be interpreted as a file name,
                or None, in which case the data will be returned as a Python string.
    @param column_names The names to write into the column header.
    @param columns The arrays of values for each column. Columns are written in the order
                   they appear here, and  must be aligned with `column_names`.
    @param float_sigfigs The number of significant figures to print floats with.
    @return Either the data as a Python string, or None.
*/
PyObject* py_write(PyObject*,
                   const py::scoped_ref<>& file,
                   const std::vector<std::string>& column_names,
                   const std::vector<py::array_view<py::any_cref>>& columns,
                   std::size_t buffer_size,
                   int num_threads,
                   std::uint8_t float_sigfigs);
}  // namespace py::csv::writer
