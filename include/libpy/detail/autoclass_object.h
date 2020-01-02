#pragma once

#include "libpy/borrowed_ref.h"
#include "libpy/detail/python.h"

namespace py::detail {
template<typename T, typename base = PyObject>
struct autoclass_object : public base {
    T value;

    static T& unbox(py::borrowed_ref<base> ob) {
        return static_cast<autoclass_object*>(ob.get())->value;
    }

    template<typename b = base, typename = std::enable_if_t<!std::is_same_v<b, PyObject>>>
    static T& unbox(py::borrowed_ref<> self) {
        return unbox(reinterpret_cast<base*>(self.get()));
    }
};
}  // namespace py::detail
