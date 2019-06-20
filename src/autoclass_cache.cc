#include <Python.h>
#if PY_MAJOR_VERSION != 2

#include <forward_list>
#include <typeindex>
#include <unordered_map>

#include "libpy/detail/autoclass_cache.h"

namespace py::detail {
std::unordered_map<std::type_index, autoclass_storage> autoclass_type_cache{};
}  // namespace py::detail
#endif
