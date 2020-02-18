#pragma once
/* This file is lifted out of `autoclass.h` because both `from_object.h` and
   `autoclass.h` need to read this cache, but `autoclass.h` depends on `from_object.h`.
*/
#include <forward_list>
#include <string>
#include <typeindex>
#include <unordered_map>
#include <vector>

#include "libpy/borrowed_ref.h"
#include "libpy/detail/api.h"
#include "libpy/detail/no_destruct_wrapper.h"
#include "libpy/detail/python.h"
#include "libpy/owned_ref.h"

namespace py::detail {
using unbox_fn = void* (*)(py::borrowed_ref<>);
struct autoclass_storage {
    // Pointer to the function which handles unboxing objects of this type. The resulting
    // pointer can be safely cast to the static type of the contained object.
    unbox_fn unbox;

    // Borrowed reference to the type that this struct contains storage for.
    py::borrowed_ref<PyTypeObject> type;

    // The method storage for `type`. We may use a vector because this is just a
    // collection of pointers and ints. `PyMethodDef` objects may move around until
    // we call `PyType_FromSpec`, at that point pointer will be taken to these objects
    // and we cannot relocate them.
    std::vector<PyMethodDef> methods;

    // The storage for `type.tp_name`, and the method `name` and `doc` fields. This uses a
    // forward list because the `PyMethodDef` objects and `PyTypeObject` point to strings
    // in this list. Because of small buffer optimization (SBO), `std::string` does not
    // have reference stability on it's contents across a move. `std::forward_list` gives
    // us the reference stability that we need. We don't use `std::list` because we want
    // to limit the overhead of this structure.
    std::forward_list<std::string> strings;

    // Storage for the `PyMethodDef` of the callback owned by `cleanup_wr`. This is not
    // in the `methods` array, because that array is passed as the `Py_tp_methods` slot
    // and will become the methods of the type. This is a free function.
    PyMethodDef callback_method;

    // A Python weakref that will delete this object from `autoclass_type_cache` when the
    // type dies.
    py::owned_ref<> cleanup_wr;

    // The Python base class for this type.
    py::owned_ref<PyTypeObject> m_pybase;

    autoclass_storage() = default;

    autoclass_storage(unbox_fn unbox, std::string&& name)
        : unbox(unbox),
          type(nullptr),
          strings({std::move(name)}),
          callback_method({nullptr, nullptr, 0, nullptr}) {}
};

// A map from a C++ RTTI object to the Python class that was created to wrap it using
// `py::autoclass`.
LIBPY_EXPORT extern no_destruct_wrapper<
    std::unordered_map<std::type_index, std::unique_ptr<autoclass_storage>>>
    autoclass_type_cache;
}  // namespace py::detail
