#include <cmath>
#include <random>

#include <libpy/abi.h>
#include <libpy/automethod.h>

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

namespace {
PyMethodDef methods[] = {
    py::autofunction<bool_scalar>("bool_scalar"),
    py::autofunction<monte_carlo_pi>("monte_carlo_pi"),
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
