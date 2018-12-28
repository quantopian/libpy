#pragma once

#include <type_traits>

/** Utilities for metaprogramming.
 */
namespace py ::meta {

/** Boolean variable template for checking if type appears in the fields of a
 * std::tuple.
 *
 *  @tparam Search The type to be searched for.
 *  @tparam Elements The types of parameters appearing in the tuple.
 *
 *  ### Examples
 *
 *  ``element_of<int, std::tuple<float, int>>`` evaluates to true.
 *  ``element_of<int, std::tuple<float, float>>`` evaluates to false.
 */
template<typename Search, typename... Elements>
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
 @tparam Haystack ``std::tuple`` of types in which to search.

 ### Examples

 ``search_tuple<int, std::tuple<float, double, int>`` evaluates to 2.
 ``search_tuple<int, std::tuple<float, int, double>`` evaluates to 1.
 ``search_tuple<int, std::tuple<float, double>`` will fail to compile.

*/
template<typename Needle, typename Haystack>
constexpr std::size_t search_tuple = detail::search_impl<0, Needle, Haystack>::value;
}  // namespace py::meta
