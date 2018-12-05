#pragma once

#include <array>
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
    return std::apply(hash_many, t);
}

namespace detail {
inline std::size_t unaligned_load(const char* p) {
    std::size_t result;
    __builtin_memcpy(&result, p, sizeof(result));
    return result;
}

// Loads n bytes, where 1 <= n < 8.
inline std::size_t load_bytes(const char* p, int n) {
    std::size_t result = 0;
    --n;
    do {
        result = (result << 8) + static_cast<unsigned char>(p[n]);
    } while (--n >= 0);
    return result;
}

inline std::size_t shift_mix(std::size_t v) {
    return v ^ (v >> 47);
}

constexpr std::size_t hash_seed = static_cast<std::size_t>(0xc70f6907UL);
constexpr std::size_t hash_mul = (static_cast<std::size_t>(0xc6a4a793UL) << 32UL) +
                                 static_cast<std::size_t>(0x5bd1e995UL);
}  // namespace detail

/** Hash a buffer of characters using the same algorithm a libstdc++
    `std::hash<std::string>`.

    @param buf The buffer to hash.
    @param len The length of the buffer.
    @return The hash of the string.
 */
inline std::size_t hash_buffer(const char* buf, std::size_t len) {
    // Remove the bytes not divisible by the sizeof(size_t).  This
    // allows the main loop to process the data as 64-bit integers.
    const int len_aligned = len & ~0x7;
    const char* const end = buf + len_aligned;
    std::size_t hash = detail::hash_seed ^ (len * detail::hash_mul);
    for (const char* p = buf; p != end; p += 8) {
        const std::size_t data = detail::shift_mix(detail::unaligned_load(p) *
                                                   detail::hash_mul) *
                                 detail::hash_mul;
        hash ^= data;
        hash *= detail::hash_mul;
    }
    if ((len & 0x7) != 0) {
        const std::size_t data = detail::load_bytes(end, len & 0x7);
        hash ^= data;
        hash *= detail::hash_mul;
    }
    hash = detail::shift_mix(hash) * detail::hash_mul;
    hash = detail::shift_mix(hash);
    return hash;
}
}  // namespace py
