#pragma once

#include <exception>
#include <memory>
#include <string>
#include <typeinfo>

#include <cxxabi.h>

namespace py::util {
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
inline demangled_cstring demangle_string(const char* cs) {
    int status;
    char* demangled = ::abi::__cxa_demangle(cs, nullptr, nullptr, &status);

    switch (status) {
    case 0:
        return demangled_cstring(demangled);
    case -1:
        throw demangle_error("memory error");
    case -2:
        throw demangle_error("invalid mangled_name");
    case -3:
        throw demangle_error("invalid argument to cxa_demangle");
    default:
        throw demangle_error("unknown failure");
    }
}

/** Demangle the given string.

    @param cs The mangled symbol or type name.
    @return The demangled string.
 */
inline demangled_cstring demangle_string(const std::string& cs) {
    return demangle_string(cs.data());
}

/** Get the demangled name for a given type.

    @tparam The type to get the name of.
    @return The demangled name.
 */
template<typename T>
demangled_cstring type_name() {
    return demangle_string(typeid(T).name());
}
}  // namespace py::util
