#include <libpy/automodule.h>
#include <libpy/autofunction.h>

namespace libpy_tutorial {
double fma(double a, double b, double c) {
    return a * b + c;
}

PyMethodDef fma_methoddef = py::autofunction<fma>("fma", "Fused Multiply Add");

LIBPY_AUTOMODULE(libpy_tutorial, function, ({fma_methoddef}))
    (py::borrowed_ref<>) {
    return false;
}
}  // namespace libpy_tutorial
