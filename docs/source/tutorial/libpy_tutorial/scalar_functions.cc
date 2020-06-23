#include <cmath>
#include <random>

#include <libpy/abi.h>
#include <libpy/autofunction.h>
#include <libpy/build_tuple.h>
#include <libpy/char_sequence.h>

namespace libpy_tutorial {

bool bool_scalar(bool a) {
    return !a;
}

double monte_carlo_pi(int n_samples) {
    int accumulator = 0;

    std::random_device rd;   // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd());  // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(0, 1);

    for (int i = 0; i < n_samples; ++i) {
        auto x = dis(gen);
        auto y = dis(gen);
        if ((std::pow(x, 2) + std::pow(y, 2)) < 1.0) {
            accumulator += 1;
        }
    }
    return 4.0 * accumulator / n_samples;
}

using namespace py::cs::literals;

std::string optional_arg(py::arg::optional<std::string> opt_arg) {
    return opt_arg.get().value_or("default value");
}

py::owned_ref<> keyword_args(
    py::arg::kwd<decltype("kw_arg_kwd"_cs), int> kw_arg_kwd,
    py::arg::opt_kwd<decltype("opt_kw_arg_kwd"_cs), int>
        opt_kw_arg_kwd) {

    return py::build_tuple(kw_arg_kwd.get(), opt_kw_arg_kwd.get());
}

namespace {
PyMethodDef methods[] = {
    py::autofunction<bool_scalar>("bool_scalar"),
    py::autofunction<monte_carlo_pi>("monte_carlo_pi"),
    py::autofunction<optional_arg>("optional_arg"),
    py::autofunction<keyword_args>("keyword_args"),
    py::end_method_list,
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "libpy_tutorial.scalar_functions",
    nullptr,
    -1,
    methods,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};

PyMODINIT_FUNC PyInit_scalar_functions() {
    if (py::abi::ensure_compatible_libpy_abi()) {
        return nullptr;
    }
    import_array();
    return PyModule_Create(&module);
}
}  // namespace
}  // namespace libpy_tutorial
