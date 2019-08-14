#pragma once

#include <string>

#include "libpy/exception.h"
#include "libpy/scoped_ref.h"

namespace py {
/** Look up an attribute on a Python object and return a new owning reference.

    @param ob The object to look up the attribute on.
    @param attr The name of the attribute to lookup.
    @return A new reference to `gettattr(ob, attr)` or `nullptr` with a Python
            exception set on failure.
 */
inline py::scoped_ref<> getattr(PyObject* ob, const std::string& attr) {
    return py::scoped_ref{PyObject_GetAttrString(ob, attr.data())};
}

/** Look up an attribute on a Python object and return a new owning reference.

    @param ob The object to look up the attribute on.
    @param attr The name of the attribute to lookup.
    @return A new reference to `gettattr(ob, attr)` or `nullptr` with a Python
            exception set on failure.
 */
inline py::scoped_ref<> getattr(const py::scoped_ref<>& ob, const std::string& attr) {
    return getattr(ob.get(), attr);
}

/** Look up an attribute on a Python object and return a new owning reference.

    @param ob The object to look up the attribute on.
    @param attr The name of the attribute to lookup.
    @return A new reference to `gettattr(ob, attr)`. If the attribute doesn't
            exist, a `py::exception` will be thrown.
 */
inline py::scoped_ref<> getattr_throws(PyObject* ob, const std::string& attr) {
    PyObject* res = PyObject_GetAttrString(ob, attr.data());
    if (!res) {
        throw py::exception{};
    }
    return py::scoped_ref{res};
}

/** Look up an attribute on a Python object and return a new owning reference.

    @param ob The object to look up the attribute on.
    @param attr The name of the attribute to lookup.
    @return A new reference to `gettattr(ob, attr)`. If the attribute doesn't
            exist, a `py::exception` will be thrown.
 */
inline py::scoped_ref<> getattr_throws(const py::scoped_ref<>& ob,
                                       const std::string& attr) {
    return getattr_throws(ob.get(), attr);
}

inline py::scoped_ref<> nested_getattr(PyObject* ob) {
    Py_INCREF(ob);
    return py::scoped_ref{ob};
}
/** Perform nested getattr calls with intermediate error checking.

    @param ob The root object to look up the attribute on.
    @param attrs The name of the attributes to lookup.
    @return A new reference to `getattr(gettattr(ob, attrs[0]), attrs[1]), ...`
            or `nullptr` with a Python exception set on failure.
 */
template<typename T, typename... Ts>
py::scoped_ref<> nested_getattr(PyObject* ob, const T& head, const Ts&... tail) {
    py::scoped_ref<> result = getattr(ob, head);
    if (!result) {
        return nullptr;
    }
    return nested_getattr(result, tail...);
}

/** Perform nested getattr calls with intermediate error checking.

    @param ob The root object to look up the attribute on.
    @param attrs The name of the attributes to lookup.
    @return A new reference to `getattr(gettattr(ob, attrs[0]), attrs[1]), ...`
            or `nullptr` with a Python exception set on failure.
 */
template<typename... Ts>
py::scoped_ref<> nested_getattr(const py::scoped_ref<> ob, const Ts&... attrs) {
    return nested_getattr(ob.get(), attrs...);
}

/** Perform nested getattr calls with intermediate error checking.

    @param ob The root object to look up the attribute on.
    @param attrs The name of the attributes to lookup.
    @return A new reference to `getattr(gettattr(ob, attrs[0]), attrs[1]), ...`.
            If an attribute in the chain doesn't exist, a `py::exception` will
            be thrown.
 */
template<typename... Ts>
py::scoped_ref<> nested_getattr_throws(const Ts&... args) {
    auto res = nested_getattr(args...);
    if (!res) {
        throw py::exception{};
    }
    return res;
}
}  // namespace py
