#pragma once

#include <Python.h>

#include "libpy/scoped_ref.h"

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

inline const char* pystring_to_cstring(PyObject* ob) {
#if PY_MAJOR_VERSION == 2
    return PyString_AsString(ob);
#else
    return PyUnicode_AsUTF8(ob);
#endif
}

/* Taken from google benchmark, this is useful for debugging.

   The DoNotOptimize(...) function can be used to prevent a value or
   expression from being optimized away by the compiler. This function is
   intended to add little to no overhead.
   See: https://youtu.be/nXaxk27zwlk?t=2441
*/
template<typename T>
inline __attribute__((always_inline)) void do_not_optimize(const T& value) {
    asm volatile("" : : "r,m"(value) : "memory");
}

template<typename T>
inline __attribute__((always_inline)) void do_not_optimize(T& value) {
#if defined(__clang__)
    asm volatile("" : "+r,m"(value) : : "memory");
#else
    asm volatile("" : "+m,r"(value) : : "memory");
#endif
}
}  // namespace py::utils
