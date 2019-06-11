#pragma once

#include <algorithm>
#include <stdexcept>
#include <string_view>

#include <Python.h>

#include "libpy/scoped_ref.h"

/** Miscellaneous utilities.
 */
namespace py::util {
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
    if (!PyString_Check(ob)) {
        throw std::runtime_error("ob is not a string");
    }
    return PyString_AS_STRING(ob);
#else
    return PyUnicode_AsUTF8(ob);
#endif
}

inline const char* pystring_to_cstring(const py::scoped_ref<>& ob) {
    return pystring_to_cstring(ob.get());
}

inline std::string_view pystring_to_string_view(PyObject* ob) {
    Py_ssize_t size;
    const char* cs;
#if PY_MAJOR_VERSION == 2
    if (!PyString_Check(ob)) {
        throw std::runtime_error("ob is not a string");
    }
    size = PyString_GET_SIZE(ob);
    cs = PyString_AS_STRING(ob);
#else
    cs = PyUnicode_AsUTF8AndSize(ob, &size);
    if (!cs) {
        throw std::runtime_error("failed to get string and size");
    }
#endif
    return {cs, static_cast<std::size_t>(size)};
}

inline std::string_view pystring_to_string_view(const py::scoped_ref<>& ob) {
    return pystring_to_string_view(ob.get());
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

/** Find lower bound index for needle within contianer.
 */
template<typename C, typename T>
std::int64_t searchsorted_l(const C& container, const T& needle) {
    auto begin = container.begin();
    return std::lower_bound(begin, container.end(), needle) - begin;
}

/** Find upper bound index for needle within container.
 */
template<typename C, typename T>
std::int64_t searchsorted_r(const C& container, const T& needle) {
    auto begin = container.begin();
    return std::upper_bound(begin, container.end(), needle) - begin;
}
}  // namespace py::util
