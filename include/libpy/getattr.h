#pragma once

#include <string>

#include "libpy/borrowed_ref.h"
#include "libpy/exception.h"
#include "libpy/owned_ref.h"

namespace py {
/** Look up an attribute on a Python object and return a new owning reference.

    @param ob The object to look up the attribute on.
    @param attr The name of the attribute to lookup.
    @return A new reference to `gettattr(ob, attr)` or `nullptr` with a Python
            exception set on failure.
 */
inline py::owned_ref<> getattr(py::borrowed_ref<> ob, const std::string& attr) {
    return py::owned_ref{PyObject_GetAttrString(ob.get(), attr.data())};
}

/** Look up an attribute on a Python object and return a new owning reference.

    @param ob The object to look up the attribute on.
    @param attr The name of the attribute to lookup.
    @return A new reference to `gettattr(ob, attr)`. If the attribute doesn't
            exist, a `py::exception` will be thrown.
 */
inline py::owned_ref<> getattr_throws(py::borrowed_ref<> ob, const std::string& attr) {
    PyObject* res = PyObject_GetAttrString(ob.get(), attr.data());
    if (!res) {
        throw py::exception{};
    }
    return py::owned_ref{res};
}

inline py::owned_ref<> nested_getattr(py::borrowed_ref<> ob) {
    return py::owned_ref<>::new_reference(ob);
}

/** Perform nested getattr calls with intermediate error checking.

    @param ob The root object to look up the attribute on.
    @param attrs The name of the attributes to lookup.
    @return A new reference to `getattr(gettattr(ob, attrs[0]), attrs[1]), ...`
            or `nullptr` with a Python exception set on failure.
 */
template<typename T, typename... Ts>
py::owned_ref<> nested_getattr(py::borrowed_ref<> ob, const T& head, const Ts&... tail) {
    py::owned_ref<> result = getattr(ob, head);
    if (!result) {
        return nullptr;
    }
    return nested_getattr(result, tail...);
}

/** Perform nested getattr calls with intermediate error checking.

    @param ob The root object to look up the attribute on.
    @param attrs The name of the attributes to lookup.
    @return A new reference to `getattr(gettattr(ob, attrs[0]), attrs[1]), ...`.
            If an attribute in the chain doesn't exist, a `py::exception` will
            be thrown.
 */
template<typename... Ts>
py::owned_ref<> nested_getattr_throws(const Ts&... args) {
    auto res = nested_getattr(args...);
    if (!res) {
        throw py::exception{};
    }
    return res;
}
}  // namespace py
