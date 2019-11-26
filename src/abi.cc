#include "libpy/abi.h"

namespace py::abi {
abi_version libpy_abi_version = detail::header_libpy_abi_version;

std::ostream& operator<<(std::ostream& stream, abi_version v) {
    return stream << v.major << '.' << v.minor << '.' << v.patch;
}
}  // namespace py::abi
