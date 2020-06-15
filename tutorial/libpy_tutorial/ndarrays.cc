#include <iostream>

#include <libpy/abi.h>
#include <libpy/automethod.h>
#include <libpy/ndarray_view.h>
#include <libpy/numpy_utils.h>

namespace libpy_tutorial {

py::owned_ref<> apply_kernel(py::ndarray_view<const std::int64_t, 3> pixels,
                             py::ndarray_view<const std::int64_t, 2> kernel) {

    std::vector<std::int64_t> out;
    std::cout << "Test";
    auto n_dimensions = pixels.shape()[2];
    auto n_rows = pixels.shape()[0];
    auto n_columns = pixels.shape()[1];

    auto k_rows = kernel.shape()[0];
    auto k_columns = kernel.shape()[1];

    out.reserve(n_dimensions * n_rows * n_columns);

    for (std::size_t dim = 0; dim < n_dimensions; ++dim) {
        for (std::size_t column = 0; column < n_columns; ++column) {
            for (std::size_t row = 0; row < n_rows; ++row) {
                auto accumulated_sum = 0.0;

                for (std::size_t k_row = 0; k_row < k_rows; ++k_row) {
                    // flipped idx - TODO - just for loop this?
                    auto row_idx = k_rows - 1 - k_row;
                    for (std::size_t k_column = 0; k_column < k_columns; ++k_column) {
                        // flipped ix - TODO - just for loop this?
                        auto col_idx = k_columns - 1 - k_column;

                        auto input_row_idx = row + 1 - row_idx;
                        auto input_column_idx = column + 1 - col_idx;

                        if (input_row_idx < n_rows && input_column_idx < n_columns) {
                            accumulated_sum +=
                                pixels.at({input_row_idx, input_column_idx, dim}) *
                                kernel.at({row_idx, col_idx});
                        }
                    }
                }
                out.emplace_back(accumulated_sum);
            }
        }
    }
    return py::move_to_numpy_array(std::move(out));
}

namespace {
PyMethodDef methods[] = {
    py::autofunction<apply_kernel>("apply_kernel"),
    py::end_method_list,
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "libpy_tutorial.ndarrays",
    nullptr,
    -1,
    methods,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};

PyMODINIT_FUNC PyInit_ndarrays() {
    if (py::abi::ensure_compatible_libpy_abi()) {
        return nullptr;
    }
    import_array();
    return PyModule_Create(&module);
}
}  // namespace

}  // namespace libpy_tutorial
