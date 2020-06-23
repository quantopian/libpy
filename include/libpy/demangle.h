#pragma once

#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <typeinfo>

#include "libpy/detail/api.h"

namespace py::util {
LIBPY_BEGIN_EXPORT
/** Exception raised when an invalid `py::demangle_string` call is performed.
 */
class demangle_error : public std::exception {
private:
    std::string m_msg;

public:
    inline demangle_error(const std::string& msg) : m_msg(msg) {}

    inline const char* what() const noexcept override {
        return m_msg.data();
    }
};

class invalid_mangled_name : public demangle_error {
public:
    inline invalid_mangled_name() : demangle_error("invalid mangled name") {}
};

/** Demangle the given string.

    @param cs The mangled symbol or type name.
    @return The demangled string.
 */
std::string demangle_string(const char* cs);

/** Demangle the given string.

    @param cs The mangled symbol or type name.
    @return The demangled string.
 */
std::string demangle_string(const std::string& cs);
LIBPY_END_EXPORT

/** Get the name for a given type. If the demangled name cannot be given, returns the
    mangled name.

    @tparam T The type to get the name of.
    @return The type's name.
 */
template<typename T>
std::string type_name() {
    const char* name = typeid(T).name();
    std::string out;
    try {
        out = demangle_string(name);
    }
    catch (const invalid_mangled_name&) {
        out = name;
    }

    if (std::is_lvalue_reference_v<T>) {
        out.push_back('&');
    }
    else if (std::is_rvalue_reference_v<T>) {
        out.insert(out.end(), 2, '&');
    }

    return out;
}
}  // namespace py::util
