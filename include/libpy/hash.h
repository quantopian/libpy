#pragma once

#include <array>
#include <string_view>
#include <tuple>

namespace py {
namespace detail {
/** Combine two hashes into one. This algorithm is taken from `boost`.
 */
template<typename T>
T bin_hash_combine(T hash_1, T hash_2) {
    return hash_1 ^ (hash_2 + 0x9e3779b9 + (hash_1 << 6) + (hash_1 >> 2));
}
}  // namespace detail

/** Combine two or more hashes into one.
 */
template<typename T, typename... Ts>
auto hash_combine(T head, Ts... tail) {
    ((head = detail::bin_hash_combine(head, tail)), ...);
    return head;
}

/** Hash multiple values by `hash_combine`ing them together.
 */
template<typename... Ts>
auto hash_many(const Ts&... vs) {
    return hash_combine(std::hash<Ts>{}(vs)...);
}

/** Hash a tuple by `hash_many`ing all the values together.
 */
template<typename... Ts>
auto hash_tuple(const std::tuple<Ts...>& t) {
    return std::apply<decltype(hash_many<Ts...>)>(hash_many, t);
}

/** Hash a buffer of characters using the same algorithm as
    `std::hash<std::string_view>`

    @param buf The buffer to hash.
    @param len The length of the buffer.
    @return The hash of the string.
 */
inline std::size_t hash_buffer(const char* buf, std::size_t len) {
    return std::hash<std::string_view>{}(std::string_view{buf, len});
}
}  // namespace py
