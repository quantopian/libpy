#pragma once

#include <cstring>
#include <cxxabi.h>
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

namespace detail {
struct demangle_deleter {
    inline void operator()(char* p) const {
        std::free(p);
    }
};
}  // namespace detail

/** A nul-terminated `char*` deleted with `std::free`.
 */
using demangled_cstring = std::unique_ptr<char, detail::demangle_deleter>;

/** Demangle the given string.

    @param cs The mangled symbol or type name.
    @return The demangled string.
 */
demangled_cstring demangle_string(const char* cs);

/** Demangle the given string.

    @param cs The mangled symbol or type name.
    @return The demangled string.
 */
demangled_cstring demangle_string(const std::string& cs);
LIBPY_END_EXPORT

/** Get the name for a given type. If the demangled name cannot be given, returns the
    mangled name.

    @tparam The type to get the name of.
    @return The demangled name.
 */
template<typename T>
demangled_cstring type_name() {
    const char* name = typeid(T).name();
    try {
        return demangle_string(name);
    }
    catch (const invalid_mangled_name&) {
        // we need to allocate this memory with `std::malloc` to line up with the
        // `std::free` in `detail::demangle_deleter`.
        std::size_t size = std::strlen(name);
        char* buf = reinterpret_cast<char*>(std::malloc(size));
        std::memcpy(buf, name, size);
        return demangled_cstring(buf);
    }
}
}  // namespace py::util
