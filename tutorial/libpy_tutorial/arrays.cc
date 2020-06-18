#include <string>
#include <vector>

#include <libpy/abi.h>
#include <libpy/automethod.h>
#include <libpy/ndarray_view.h>
#include <libpy/numpy_utils.h>

namespace libpy_tutorial {

std::int64_t simple_sum(py::array_view<const std::int64_t> values) {
    std::int64_t out = 0;
    for (auto value : values) {
        out += value;
    }
    return out;
}

std::int64_t simple_sum_iterator(py::array_view<const std::int64_t> values) {
    return std::accumulate(values.begin(), values.end(), 0);
}

bool check_prime(std::int64_t n) {
    if (n <= 3) {
        return n > 1;
    }
    else if (n % 2 == 0 || n % 3 == 0) {
        return false;
    }
    for (auto i = 5; std::pow(i, 2) < n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0) {
            return false;
        }
    }
    return true;
}

py::owned_ref<> is_prime(py::array_view<const std::int64_t> values) {
    std::vector<py::py_bool> out(values.size());
    std::transform(values.begin(), values.end(), out.begin(), check_prime);

    return py::move_to_numpy_array(std::move(out));
}

namespace {
PyMethodDef methods[] = {
    py::autofunction<simple_sum>("simple_sum"),
    py::autofunction<simple_sum_iterator>("simple_sum_iterator"),
    py::autofunction<is_prime>("is_prime"),
    py::end_method_list,
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "libpy_tutorial.arrays",
    nullptr,
    -1,
    methods,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};

PyMODINIT_FUNC PyInit_arrays() {
    if (py::abi::ensure_compatible_libpy_abi()) {
        return nullptr;
    }
    import_array();
    return PyModule_Create(&module);
}
}  // namespace
}  // namespace libpy_tutorial
