#pragma once

namespace py {
/** Combine two hashes into one. This algorithm is taken from `boost`.
 */
template<typename T>
T hash_combine(T hash_1, T hash_2) {
    return hash_1 ^ (hash_2 + 0x9e3779b9 + (hash_1 << 6) + (hash_1 >> 2));
}
}  // namespace py
