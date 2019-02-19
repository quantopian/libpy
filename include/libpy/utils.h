#pragma once

#include <Python.h>

/** Miscellaneous utilities.
 */
namespace py::utils {
/** Check if all parameters are equal.
 */
template<typename T, typename... Ts>
bool all_equal(T&& head, Ts&&... tail) {
    return (... && (head == tail));
}

inline bool all_equal() {
    return true;
}

#if PY_MAJOR_VERSION == 2
inline const char* pystring_to_cstring(PyObject* ob) {
    return PyString_AsString(ob);
}
#else
inline const char* pystring_to_cstring(PyObject* ob) {
    return PyUnicode_AsUTF8(ob);
}
#endif
}  // namespace py::utils
