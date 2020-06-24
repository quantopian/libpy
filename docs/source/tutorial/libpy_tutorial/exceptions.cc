#include <stdexcept>
#include <string>
#include <vector>

#include <libpy/autofunction.h>
#include <libpy/automodule.h>
#include <libpy/exception.h>

namespace libpy_tutorial {

void throw_value_error(int a) {
    throw py::exception(PyExc_ValueError, "You passed ", a, " and this is the exception");
}

void raise_from_cxx() {
    throw std::invalid_argument("Supposedly a bad argument was used");
}

LIBPY_AUTOMODULE(libpy_tutorial,
                 exceptions,
                 ({py::autofunction<throw_value_error>("throw_value_error"),
                   py::autofunction<raise_from_cxx>("raise_from_cxx")}))
(py::borrowed_ref<>) {
    return false;
}

}  // namespace libpy_tutorial
