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
constexpr bool element_of;

// Base Case: Search is not an element of the empty tuple.
template<typename Search>
constexpr bool element_of<Search, std::tuple<>> = false;

// Recursive Case: Search is an element of tuple<Head, Rest...> if it's equal to Head
// or if it's an element of tuple<Rest...>.
template<typename Search, typename Head, typename... Rest>
constexpr bool element_of<Search, std::tuple<Head, Rest...>> =
    std::is_same_v<Search, Head> || element_of<Search, std::tuple<Rest...>>;

}  // namespace py::meta
