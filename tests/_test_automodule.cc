#include "libpy/autoclass.h"
#include "libpy/autofunction.h"
#include "libpy/automodule.h"

bool is_42(int arg) {
    return arg == 42;
}

bool is_true(bool arg) {
    return arg;
}

using int_float_pair = std::pair<int, float>;

int first(const int_float_pair& ob) {
    return ob.first;
}

float second(const int_float_pair& ob) {
    return ob.second;
}

LIBPY_AUTOMODULE(tests,
                 _test_automodule,
                 ({py::autofunction<is_42>("is_42"),
                   py::autofunction<is_true>("is_true")}))
(py::borrowed_ref<> m) {
    py::autoclass<int_float_pair>(m, "int_float_pair")
        .new_<int, float>()
        .comparisons<int_float_pair>()
        .def<first>("first")
        .def<second>("second")
        .type();
    return false;
}
