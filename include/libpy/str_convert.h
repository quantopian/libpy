#pragma once

#include "libpy/char_sequence.h"
#include "libpy/scoped_ref.h"

namespace py {

enum class str_type {
    bytes,    // str in py2, bytes in py3.
    str,      // str, in py2 and py3
    unicode,  // unicode in py2, str in py3.
};

/** Convert a compile-time string into a Python string-like value.

    @param s Char sequence whose type encodes a compile-time string.
    @param type Enum representing the type into which to convert `s`.

    If the requested output py::str_type::str or py::str_type::unicode, the input string
    will be must be valid utf-8.
 */
template<char... cs>
scoped_ref<> to_stringlike(py::cs::char_sequence<cs...> s, py::str_type type) {
    const auto as_null_terminated_array = py::cs::to_array(s);
    const char* data = as_null_terminated_array.data();
    Py_ssize_t size = sizeof...(cs);

    switch (type) {
    case py::str_type::bytes: {
        return scoped_ref<>{PyBytes_FromStringAndSize(data, size)};
    }
    case py::str_type::str: {
#if PY_MAJOR_VERSION == 2
        return scoped_ref<>{PyString_FromStringAndSize(data, size)};
#else
        return scoped_ref<>{PyUnicode_FromStringAndSize(data, size)};
#endif
    }
    case py::str_type::unicode: {
        return scoped_ref<>{PyUnicode_FromStringAndSize(data, size)};
    }
    }
    __builtin_unreachable();
}

}  // namespace py
