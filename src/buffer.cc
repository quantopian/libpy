#include "libpy/buffer.h"

namespace py {
py::buffer get_buffer(py::borrowed_ref<> ob, int flags) {
    py::buffer buf(new Py_buffer);
    if (PyObject_GetBuffer(ob.get(), buf.get(), flags)) {
        throw py::exception{};
    }

    return buf;
}
}  // namespace py
