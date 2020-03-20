#pragma once

#include "libpy/borrowed_ref.h"
#include "libpy/detail/python.h"

namespace py::detail {
template<typename T, typename b = PyObject>
struct autoclass_object : public b {
    using base = b;

    T value;

    static T& unbox(py::borrowed_ref<base> ob) {
        return static_cast<autoclass_object*>(ob.get())->value;
    }

    template<typename c = base, typename = std::enable_if_t<!std::is_same_v<c, PyObject>>>
    static T& unbox(py::borrowed_ref<> self) {
        return unbox(reinterpret_cast<base*>(self.get()));
    }
};

template<typename I>
struct autoclass_interface_object : public PyObject {
    using base = PyObject;

    I* virt_storage_ptr;

    static I& unbox(py::borrowed_ref<> ob) {
        return *std::launder(
            static_cast<autoclass_interface_object*>(ob.get())->virt_storage_ptr);
    }
};

template<typename T, typename I>
using autoclass_interface_instance_object =
    autoclass_object<T, autoclass_interface_object<I>>;
}  // namespace py::detail
