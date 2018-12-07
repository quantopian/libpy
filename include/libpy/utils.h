#pragma once

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
}  // namespace py::utils
