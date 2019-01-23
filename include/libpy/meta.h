#pragma once

#include <tuple>
#include <type_traits>

/** Utilities for metaprogramming.
 */
namespace py ::meta {
/** Remove reference then cv-qualifiers.
 */
template<typename T>
using remove_remove_cvref = std::remove_reference_t<std::remove_cv_t<T>>;

/** Boolean variable template for checking if type appears in the fields of a
 * std::tuple.
 *
 *  @tparam Search The type to be searched for.
 *  @tparam Elements The types of parameters appearing in the tuple.
 *
 *  ### Examples
 *
 *  - `element_of<int, std::tuple<float, int>>` evaluates to true.
 *  - `element_of<int, std::tuple<float, float>>` evaluates to false.
 */
template<typename Search, typename Elements>
constexpr bool element_of = false;

// Base Case: Search is not an element of the empty tuple.
template<typename Search>
constexpr bool element_of<Search, std::tuple<>> = false;

// Recursive Case: Search is an element of tuple<Head, Rest...> if it's equal to Head
// or if it's an element of tuple<Rest...>.
template<typename Search, typename Head, typename... Rest>
constexpr bool element_of<Search, std::tuple<Head, Rest...>> =
    std::is_same_v<Search, Head> || element_of<Search, std::tuple<Rest...>>;

namespace detail {

template<std::size_t ix, typename Needle, typename... Haystack>
struct search_impl;

template<std::size_t ix, typename Needle, typename... Tail>
struct search_impl<ix, Needle, std::tuple<Needle, Tail...>> {
    constexpr static std::size_t value = ix;
};

template<std::size_t ix, typename Needle, typename Head, typename... Tail>
struct search_impl<ix, Needle, std::tuple<Head, Tail...>> {
    constexpr static std::size_t value =
        search_impl<ix + 1, Needle, std::tuple<Tail...>>::value;
};
}  // namespace detail

/** Variable template for getting the index at which a type appears in the fields of a
 * std::tuple.

 @tparam Needle The type to be searched for.
 @tparam Haystack `std::tuple` of types in which to search.

 ### Examples

 - `search_tuple<int, std::tuple<float, double, int>` evaluates to 2.
 - `search_tuple<int, std::tuple<float, int, double>` evaluates to 1.
 - `search_tuple<int, std::tuple<float, double>` will fail to compile.

*/
template<typename Needle, typename Haystack>
constexpr std::size_t search_tuple = detail::search_impl<0, Needle, Haystack>::value;

namespace detail {
template<typename... Tuples>
struct type_cat_impl;

template<>
struct type_cat_impl<> {
    using type = std::tuple<>;
};

template<typename... Ts>
struct type_cat_impl<std::tuple<Ts...>> {
    using type = std::tuple<Ts...>;
};

template<typename... As, typename... Bs>
struct type_cat_impl<std::tuple<As...>, std::tuple<Bs...>> {
    using type = std::tuple<As..., Bs...>;
};

template<typename Head, typename... Tail>
struct type_cat_impl<Head, Tail...> {
    using type =
        typename type_cat_impl<Head, typename type_cat_impl<Tail...>::type>::type;
};
}  // namespace detail

template<typename... Ts>
using type_cat = typename detail::type_cat_impl<Ts...>::type;

namespace detail {
template<typename A, typename B>
struct set_diff_impl;

template<typename... As, typename B>
struct set_diff_impl<std::tuple<As...>, B> {
    using type =
        type_cat<std::conditional_t<element_of<As, B>, std::tuple<>, std::tuple<As>>...>;
};
}  // namespace detail

/** Take the set difference of the types in A and B.

    @tparam A The left hand side of the difference.
    @tparam B The right hand side of the difference.

    ### Examples

    - `set_diff<std::tuple<int, float, double>, std::tuple<int>>` evaluates to
      `std::tuple<float, double>`.
    - `set_diff<std::tuple<int, long, float, double>, std::tuple<long, int>>` evaluates to
      `std::tuple<float, double>`.
 */
template<typename A, typename B>
using set_diff = typename detail::set_diff_impl<A, B>::type;

template<typename>
struct print_t;

template<auto>
struct print_v;
}  // namespace py::meta
