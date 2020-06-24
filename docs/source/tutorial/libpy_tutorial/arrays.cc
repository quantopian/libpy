#include <string>
#include <vector>

#include <libpy/autofunction.h>
#include <libpy/automodule.h>
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

LIBPY_AUTOMODULE(libpy_tutorial,
                 arrays,
                 ({py::autofunction<simple_sum>("simple_sum"),
                   py::autofunction<simple_sum_iterator>("simple_sum_iterator"),
                   py::autofunction<is_prime>("is_prime")}))
(py::borrowed_ref<>) {
    return false;
}

}  // namespace libpy_tutorial
