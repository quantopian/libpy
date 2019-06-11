#include <Python.h>
#if PY_MAJOR_VERSION != 2

#include <list>
#include <typeindex>
#include <unordered_map>

#include "libpy/detail/autoclass_cache.h"

namespace py::detail {
std::unordered_map<std::type_index, py::scoped_ref<>> autoclass_type_cache{};

std::list<autoclass_storage> autoclass_storage_cache{};

void clear_autoclass_cache() {
    autoclass_type_cache.clear();
    autoclass_storage_cache.clear();
}
}  // namespace py::detail
#endif
