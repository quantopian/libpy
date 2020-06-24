#include <iostream>

#include <libpy/autofunction.h>
#include <libpy/automodule.h>
#include <libpy/ndarray_view.h>
#include <libpy/numpy_utils.h>

namespace libpy_tutorial {

py::owned_ref<> apply_kernel(py::ndarray_view<const std::uint8_t, 3> pixels,
                             py::ndarray_view<const std::int64_t, 2> kernel) {

    auto n_dimensions = pixels.shape()[2];
    auto n_rows = pixels.shape()[0];
    auto n_columns = pixels.shape()[1];

    auto k_rows = kernel.shape()[0];
    auto k_columns = kernel.shape()[1];
    std::vector<std::uint8_t> out(n_dimensions * n_rows * n_columns, 0);
    py::ndarray_view out_view(out.data(),
                              pixels.shape(),
                              {static_cast<int>(n_dimensions * n_rows),
                               static_cast<int>(n_dimensions),
                               1});

    for (std::size_t dim = 0; dim < n_dimensions; ++dim) {
        for (std::size_t row = 0; row < n_rows; ++row) {
            for (std::size_t column = 0; column < n_columns; ++column) {

                auto accumulated_sum = 0.0;

                for (std::size_t k_row = 0; k_row < k_rows; ++k_row) {
                    for (std::size_t k_column = 0; k_column < k_columns; ++k_column) {

                        auto input_row_idx = row + 1 - k_row;
                        auto input_column_idx = column + 1 - k_column;

                        if (input_row_idx < n_rows && input_column_idx < n_columns) {
                            accumulated_sum +=
                                pixels(input_row_idx, input_column_idx, dim) *
                                kernel(k_row, k_column);
                        }
                    }
                }
                if (accumulated_sum < 0) {
                    accumulated_sum = 0;
                }
                else if (accumulated_sum > 255) {
                    accumulated_sum = 255;
                }
                out_view(row, column, dim) = accumulated_sum;
            }
        }
    }
    return py::move_to_numpy_array(std::move(out),
                                   py::new_dtype<uint8_t>(),
                                   pixels.shape(),
                                   pixels.strides());
}

LIBPY_AUTOMODULE(libpy_tutorial,
                 ndarrays,
                 ({py::autofunction<apply_kernel>("apply_kernel")}))
}  // namespace libpy_tutorial
