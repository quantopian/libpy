#pragma once

#include "libpy/owned_ref.h"
#include "libpy/to_object.h"

namespace py {
/** Build a Python tuple from a variadic amount of arguments.

    All parameters are adapted using `py::to_object`.

    @param args The arguments to adapt into Python objects and pack into a tuple.
    @return A new Python tuple or `nullptr` with a Python exception set.
    @see py::to_object
 */
template<typename... Args>
py::owned_ref<> build_tuple(const Args&... args) {
    py::owned_ref out(PyTuple_New(sizeof...(args)));
    if (!out) {
        return nullptr;
    }

    Py_ssize_t ix = 0;
    (PyTuple_SET_ITEM(out.get(), ix++, py::to_object(args).escape()), ...);
    for (ix = 0; ix < static_cast<std::int64_t>(sizeof...(args)); ++ix) {
        if (!PyTuple_GET_ITEM(out.get(), ix)) {
            return nullptr;
        }
    }

    return out;
}
}  // namespace py
