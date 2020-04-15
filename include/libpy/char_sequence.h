#pragma once

#include <array>
#include <type_traits>
#include <utility>

namespace py::cs {
/** A compile time sequence of characters.
 */
template<char... cs>
using char_sequence = std::integer_sequence<char, cs...>;

inline namespace literals {
/** User defined literal for creating a `py::cs::char_sequence` value.

    \code
    using a = py::cs::char_sequence<'a', 'b', 'c', 'd'>;
    using b = decltype("abcd"_cs);

    static_assert(std::is_same_v<a, b>);
    \endcode
*/
template<typename Char, Char... cs>
constexpr char_sequence<cs...> operator""_cs() {
    return {};
}

/** User defined literal for creating a `std::array` of characters.

    \code
    constexpr std::array a = {'a', 'b', 'c', 'd'};
    constexpr std::array b = "abcd"_arr;

    static_assert(a == b);
    \endcode

    @note This does not add a nul terminator to the array.
 */
template<typename Char, Char... cs>
constexpr std::array<char, sizeof...(cs)> operator""_arr() {
    return {cs...};
}
};  // namespace literals

namespace detail {
template<char... cs, char... ds>
constexpr auto binary_cat(char_sequence<cs...>, char_sequence<ds...>) {
    return char_sequence<cs..., ds...>{};
}
}  // namespace detail

/** Concatenate character sequences.
 */
constexpr char_sequence<> cat() {
    return {};
}

/** Concatenate character sequences.
 */
template<typename Cs>
constexpr auto cat(Cs cs) {
    return cs;
}

/** Concatenate character sequences.
 */
template<typename Cs, typename Ds>
constexpr auto cat(Cs cs, Ds ds) {
    return detail::binary_cat(cs, ds);
}

/** Concatenate character sequences.
 */
template<typename Cs, typename Ds, typename... Ts>
constexpr auto cat(Cs cs, Ds ds, Ts... es) {
    return detail::binary_cat(detail::binary_cat(cs, ds), cat(es...));
}

/** Convert a character sequence into a `std::array` with a trailing null byte.
 */
template<char... cs>
constexpr auto to_array(char_sequence<cs...>) {
    return std::array<char, sizeof...(cs) + 1>{cs..., '\0'};
}

namespace detail {
template<char c, typename Cs>
struct intersperse;

// recursive base case
template<char c>
struct intersperse<c, char_sequence<>> {
    constexpr static char_sequence<> value{};
};

template<char c, char head, char... tail>
struct intersperse<c, char_sequence<head, tail...>> {
    constexpr static auto value = cat(char_sequence<head, c>{},
                                      intersperse<c, char_sequence<tail...>>::value);
};
};  // namespace detail

/** Intersperse a character between all the characters of a `char_sequence`.

    @tparam c The character to intersperse into the sequence
    @param cs The sequence to intersperse the character into.
 */
template<char c, typename Cs>
constexpr auto intersperse(Cs) {
    return detail::intersperse<c, Cs>::value;
}

namespace detail {
template<typename J, typename... Cs>
struct join;

// recursive base case
template<typename J>
struct join<J> {
    constexpr static char_sequence<> value{};
};

template<char... j, char... cs, typename... Tail>
struct join<char_sequence<j...>, char_sequence<cs...>, Tail...> {
private:
    using joiner = char_sequence<j...>;

public:
    constexpr static auto value =
        cs::cat(char_sequence<cs...>{},
                std::conditional_t<sizeof...(Tail) != 0, joiner, char_sequence<>>{},
                join<joiner, Tail...>::value);
};
}  // namespace detail

/** Join a sequence of compile-time strings together with another compile-time
    string.

    This is like `joiner.join(cs)` in Python.

    @param joiner The string to join with.
    @param cs... The strings to join together.
 */
template<typename J, typename... Cs>
constexpr auto join(J, Cs...) {
    return detail::join<J, Cs...>::value;
}
}  // namespace py::cs
