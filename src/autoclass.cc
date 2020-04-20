#include <cstring>
#include <forward_list>
#include <typeindex>
#include <unordered_map>

#include "libpy/detail/python.h"
#include <object.h>
#include <structmember.h>

#include "libpy/detail/autoclass_cache.h"

namespace py::detail {
no_destruct_wrapper<
    std::unordered_map<std::type_index, std::unique_ptr<autoclass_storage>>>
    autoclass_type_cache{};
}  // namespace py::detail
