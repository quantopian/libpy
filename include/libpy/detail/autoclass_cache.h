#pragma once
/** This file is lifted out of `autoclass.h` because both `from_object.h` and
    `autoclass.h` need to read this cache, but `autoclass.h` depends on `from_object.h`.
 */
#include <Python.h>
#if PY_MAJOR_VERSION != 2

#include <forward_list>
#include <typeindex>
#include <unordered_map>
#include <vector>

#include <Python.h>

#include "libpy/scoped_ref.h"

namespace py::detail {
// A map from a C++ RTTI object to the Python class that was created to wrap it using
// `py::autoclass`.
extern std::unordered_map<std::type_index, py::scoped_ref<>> autoclass_type_cache;

struct autoclass_storage {
    std::vector<PyMethodDef> methods;
    std::forward_list<std::string> strings;
};

// Use a `std::forward_list` to have reference stability on insert.
extern std::forward_list<autoclass_storage> autoclass_storage_cache;

/** Clear the cached types from autoclass.
 */
void clear_autoclass_cache();
}  // namespace py::detail
#endif
