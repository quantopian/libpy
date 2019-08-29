#pragma once

#include "libpy/scoped_ref.h"

namespace py {
inline py::scoped_ref<> none{py::scoped_ref<>::new_reference(Py_None)};
inline py::scoped_ref<> ellipsis{py::scoped_ref<>::new_reference(Py_Ellipsis)};
inline py::scoped_ref<> not_implemented{
    py::scoped_ref<>::new_reference(Py_NotImplemented)};
}  // namespace py
