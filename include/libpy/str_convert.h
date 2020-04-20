#pragma once

#include "libpy/char_sequence.h"
#include "libpy/owned_ref.h"

namespace py {

enum class str_type {
    bytes,
    str,
};

/** Convert a compile-time string into a Python string-like value.

    @param s Char sequence whose type encodes a compile-time string.
    @param type Enum representing the type into which to convert `s`.

    If the requested output py::str_type::str the input string must
    be valid utf-8.
 */
template<char... cs>
owned_ref<> to_stringlike(py::cs::char_sequence<cs...> s, py::str_type type) {
    const auto as_null_terminated_array = py::cs::to_array(s);
    const char* data = as_null_terminated_array.data();
    Py_ssize_t size = sizeof...(cs);

    switch (type) {
    case py::str_type::bytes: {
        return owned_ref<>{PyBytes_FromStringAndSize(data, size)};
    }
    case py::str_type::str: {
        return owned_ref<>{PyUnicode_FromStringAndSize(data, size)};
    }
    }
    __builtin_unreachable();
}

}  // namespace py
