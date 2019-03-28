#pragma once

#include "libpy/scoped_ref.h"
#include "libpy/to_object.h"

namespace py {
template<typename... Args>
py::scoped_ref<> build_tuple(const Args&... args) {
    py::scoped_ref out(PyTuple_New(sizeof...(args)));
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
