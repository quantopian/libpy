#pragma once

#include <tuple>
#include <type_traits>

/** Utilities for metaprogramming.
 */
namespace py::meta {
/** Debugging helper to get the compiler to dump a type as an error message.

    @tparam T The type to print.

    ### Usage

    When debugging template meta-functions it is sometimes helpful to see the resulting
    type of an expression. To get the compiler to emit a message, instantiate `print_t`
    with the type of interest and access any member inside the type, for example `::a`.

    \code
    using T = std::remove_cv_t<std::remove_reference_t<const int&>>;

    print_t<T>::a;
    \endcode

    \code
    error: invalid use of incomplete type ‘struct py::meta::print_t<int>’
     print_t<T>::a;
             ^
    \endcode

    In the message we see: `py::meta::print_t<int>`, where `int` is the result of
    `std::remove_cv_t<std::remove_reference_t<const int&>>`.

    ### Notes

    It doesn't matter what member is requested, `::a` is chosen because it is short and
    easy to type, any other name will work.
 */
template<typename>
struct print_t;

/** Debugging helper to get the compiler to dump a compile-time value as an error message.

    @tparam v The value to print.

    ### Usage

    When debugging template meta-functions it is sometimes helpful to see the resulting
    value of an expression. To get the compiler to emit a message, instantiate `print_v`
    with the value of interest and access any member inside the type, for example `::a`.

    \code
    template<auto v>
    unsigned long fib = fib<v - 1> + fib<v - 2>;

    template<>
    unsigned long fib<1> = 1;

    template<>
    unsigned long fib<2> = 1;

    unsigned long v = fib<24>;

    print_v<T>::a;
    \endcode

    \code
    error: invalid use of incomplete type ‘struct py::meta::print_v<46368>’
     print_v<v>::a;
    \endcode

    In the message we see: `py::meta::print_v<46368>`, where `46368` is the value of
    `fib<24>`.

    ### Notes

    It doesn't matter what member is requested, `::a` is chosen because it is short and
    easy to type, any other name will work.
 */
template<const auto&>
struct print_v;

/** Remove reference then cv-qualifiers.
 */
template<typename T>
using remove_cvref = std::remove_cv_t<std::remove_reference_t<T>>;

/** Boolean variable template for checking if type appears in the fields of a
    std::tuple.

    Examples:

    - `element_of<int, std::tuple<float, int>>` evaluates to true.
    - `element_of<int, std::tuple<float, float>>` evaluates to false.

    @tparam Search The type to be searched for.
    @tparam Elements The types of parameters appearing in the tuple.
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
    std::tuple.

    Examples

    - `search_tuple<int, std::tuple<float, double, int>` evaluates to 2.
    - `search_tuple<int, std::tuple<float, int, double>` evaluates to 1.
    - `search_tuple<int, std::tuple<float, double>` will fail to compile.

    @tparam Needle The type to be searched for.
    @tparam Haystack `std::tuple` of types in which to search.
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

/** Concatenate the types in a tuple into a single flattened tuple.

    @tparam Ts The tuples to concat.
 */
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

    Examples:

    - `set_diff<std::tuple<int, float, double>, std::tuple<int>>` evaluates to
      `std::tuple<float, double>`.
    - `set_diff<std::tuple<int, long, float, double>, std::tuple<long, int>>` evaluates to
      `std::tuple<float, double>`.

      @tparam A The left hand side of the difference.
      @tparam B The right hand side of the difference.
 */
template<typename A, typename B>
using set_diff = typename detail::set_diff_impl<A, B>::type;

/** A collection of function objects that implement simple operators.

    These are useful for metaprogramming contexts where you want generic operators as
    function objects for higher order templates and functions.
 */
namespace op {
/** Define a new operator function type named `name` that implements the `op` binary
    operator.

    # Example

    \code
    DEFINE_BINOP(+, add)
    \endcode

    Creates a struct named `add` that implements `operator()` to forward to `lhs + rhs`.
 */
#define DEFINE_BINOP(op, name)                                                           \
    struct name {                                                                        \
        template<typename LHS, typename RHS>                                             \
        constexpr auto operator()(LHS&& lhs, RHS&& rhs) noexcept(noexcept(lhs op rhs))   \
            -> decltype(lhs op rhs) {                                                    \
            return (lhs op rhs);                                                         \
        }                                                                                \
    };

DEFINE_BINOP(+, add)
DEFINE_BINOP(-, sub)
DEFINE_BINOP(*, mul)
DEFINE_BINOP(%, rem)
DEFINE_BINOP(/, div)
DEFINE_BINOP(<<, lshift)
DEFINE_BINOP(>>, rshift)
DEFINE_BINOP(&, and_)
DEFINE_BINOP(^, xor_)
DEFINE_BINOP(|, or_)
DEFINE_BINOP(>, gt)
DEFINE_BINOP(>=, ge)
DEFINE_BINOP(==, eq)
DEFINE_BINOP(<=, le)
DEFINE_BINOP(<, lt)
DEFINE_BINOP(!=, ne)

#undef DEFINE_BINOP

#define DEFINE_UNOP(op, name)                                                            \
    struct name {                                                                        \
        template<typename T>                                                             \
        constexpr auto operator()(T&& value) noexcept(noexcept(op value))                \
            -> decltype(op value) {                                                      \
            return op value;                                                             \
        }                                                                                \
    };

DEFINE_UNOP(-, neg)
DEFINE_UNOP(+, pos)
DEFINE_UNOP(~, inv)

#undef DEFINE_UNOP
}  // namespace op

namespace detail {
/** Recursive base case for flattening a tuple that forms a deep right tree. For example:

    (0, (1, (2, (3, (4, 5))))) -> (0, 1, 2, 3, 4, 5)

    @param t The tree to flatten.
    @return The flat tuple.
 */
template<typename T>
constexpr auto flatten_right_tree(const T& v) noexcept {
    return std::make_tuple(v);
}

/** Flatten a tuple that forms a deep right tree. For example:

    (0, (1, (2, (3, (4, 5))))) -> (0, 1, 2, 3, 4, 5)

    @param t The tree to flatten.
    @return The flat tuple.
 */
template<typename L, typename R>
constexpr auto flatten_right_tree(const std::tuple<L, R>& t) noexcept {
    const auto& [l, r] = t;
    return std::tuple_cat(std::make_tuple(l), flatten_right_tree(r));
}
}  // namespace detail

/** Cartesian product of two `std::tuple` objects.

    @param a The first tuple.
    @param b The second tuple.
    @return A `std::tuple` of length 2 tuples containing the pairwise combinations from
            `a` and `b`.
 */
template<typename A, typename B>
constexpr auto tuple_prod(const A& a, const B& b) noexcept {
    return std::apply(
        [&](const auto&... as) {
            return std::tuple_cat(std::apply(
                [&](const auto& a_scalar, const auto&... bs) {
                    return std::make_tuple(std::make_tuple(a_scalar, bs)...);
                },
                std::tuple_cat(std::make_tuple(as), b))...);
        },
        a);
}

/** Cartesian product of two or more `std::tuple` objects.

    @param head The first tuple.
    @param tail The rest of the tuples.
    @return A `std::tuple` of length `sizeof...(tail) + 1` tuples containing the n-wise
            combinations from each input tuple.
 */
template<typename Head, typename... Tail>
constexpr auto tuple_prod(const Head& head, const Tail&... tail) noexcept {
    return std::apply(
        [](auto... ts) { return std::make_tuple(detail::flatten_right_tree(ts)...); },
        tuple_prod(head, tuple_prod(tail...)));
}
}  // namespace py::meta
