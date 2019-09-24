#pragma once

namespace py {

enum class str_type {
    bytes,    // str in py2, bytes in py3.
    str,      // str, in py2 and py3
    unicode,  // unicode in py2, str in py3.
};

}  // namespace py
