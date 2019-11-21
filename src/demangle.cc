#include <cxxabi.h>

#include "libpy/demangle.h"

namespace py::util {
demangled_cstring demangle_string(const char* cs) {
    int status;
    char* demangled = ::abi::__cxa_demangle(cs, nullptr, nullptr, &status);

    switch (status) {
    case 0:
        return demangled_cstring(demangled);
    case -1:
        throw demangle_error("memory error");
    case -2:
        throw invalid_mangled_name();
    case -3:
        throw demangle_error("invalid argument to cxa_demangle");
    default:
        throw demangle_error("unknown failure");
    }
}

demangled_cstring demangle_string(const std::string& cs) {
    return demangle_string(cs.data());
}
}  // namespace py
