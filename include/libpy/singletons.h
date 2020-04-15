#pragma once

#include "libpy/owned_ref.h"

namespace py {
inline py::owned_ref<> none{py::owned_ref<>::new_reference(Py_None)};
inline py::owned_ref<> ellipsis{py::owned_ref<>::new_reference(Py_Ellipsis)};
inline py::owned_ref<> not_implemented{py::owned_ref<>::new_reference(Py_NotImplemented)};
}  // namespace py
