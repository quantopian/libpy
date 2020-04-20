#pragma once

#include <tuple>
#include <vector>

#include "libpy/meta.h"
#include "libpy/ndarray_view.h"

namespace py {
/** Interface to provide runtime type-checked statically typed access to a
    collection of type-erased `ndarray_views`.

    @tparam ndim The number of dimensions for the underlying `py::ndarray_view` objects.
    @tparam Ts A sequence of tuples of types that convey the possible combinations of
            arrays.
 */
template<std::size_t ndim, typename... Ts>
struct nwise_nddevirtualize {
private:
    static_assert(util::all_equal(std::tuple_size_v<Ts>...),
                  "all potential devirtualized signatures must be the same arity");
    static constexpr std::size_t arity = (std::tuple_size_v<Ts>, ...);

    static_assert(arity > 0, "arity must be non-zero");
    static_assert(ndim > 0, "ndim must be non-zero");

    template<typename>
    struct views_impl;

    template<typename... Us>
    struct views_impl<std::tuple<Us...>> {
        using type = std::tuple<py::ndarray_view<Us, ndim>...>;
    };

    template<typename U>
    using views = typename views_impl<U>::type;

    template<typename U>
    using type_entry = std::vector<std::pair<views<U>, std::size_t>>;

    std::tuple<type_entry<Ts>...> m_arrays;

    template<typename, typename...>
    struct maybe_add_typed_entry_impl;

    template<typename... Vs, typename... Args>
    struct maybe_add_typed_entry_impl<std::tuple<Vs...>, Args...> {
    private:
        static_assert(util::all_equal(std::is_same_v<Args, py::any_ref> ||
                                      std::is_same_v<Args, py::any_cref>...),
                      "type-erased array input must be py::ndarray_view<py::any_ref, "
                      "ndim> or py::ndarray_view<py::any_cref, ndim>");

        template<typename V, typename Arg>
        static bool scalar_match(const py::ndarray_view<Arg, ndim>& arr) {
            return std::is_same_v<V, Arg> || std::is_same_v<V, py::any_cref> ||
                   arr.vtable() == py::any_vtable::make<V>() ||
                   arr.vtable() == py::any_vtable::make<std::remove_const_t<V>>();
        }

    public:
        static bool f(type_entry<std::tuple<Vs...>>& maybe_out,
                      std::size_t ix,
                      const py::array_view<Args>&... args) {
            if ((scalar_match<Vs>(args) && ...)) {
                maybe_out.push_back(
                    {views<std::tuple<Vs...>>(args.template cast<Vs>()...), ix});
                return true;
            }
            return false;
        }
    };

    template<typename V, typename... Args>
    static bool maybe_add_typed_entry(type_entry<V>& maybe_out,
                                      std::size_t ix,
                                      const py::array_view<Args>&... args) {
        return maybe_add_typed_entry_impl<V, Args...>::f(maybe_out, ix, args...);
    }

    template<std::size_t tuple_ix, typename Head, typename... Tail, typename... Args>
    void add_typed_entry(std::size_t ix, const py::ndarray_view<Args, ndim>&... args) {
        if (!maybe_add_typed_entry<Head>(std::get<tuple_ix>(m_arrays), ix, args...)) {
            add_typed_entry<tuple_ix + 1, Tail...>(ix, args...);
        }
    }

    template<std::size_t tuple_ix, typename... Args>
    [[noreturn]] void add_typed_entry(std::size_t,
                                      const py::ndarray_view<Args, ndim>&...) {
        throw std::bad_any_cast{};
    }

public:
    /** Construct an `nwise_devirtualize` from `arity` collections of
        `py::ndarray_view<py::any_ref>` or `py::array_view<py::any_cref, ndim>`.

        @param array_collections One sequence of array views per `arity`.
     */
    template<typename... Args>
    nwise_nddevirtualize(const Args&... array_collections) {
        static_assert(sizeof...(Args) == arity, "Size of array_collections != arity");

        if (!util::all_equal(array_collections.size()...)) {
            throw std::invalid_argument{"array collections are not all the same size"};
        }

        std::size_t max_size = std::get<0>(std::make_tuple(array_collections...)).size();
        for (std::size_t ix = 0; ix < max_size; ++ix) {
            add_typed_entry<0, Ts...>(ix, array_collections[ix]...);
        }
    }

private:
    template<typename F, typename E>
    void for_each_helper(F&& f, const std::vector<E>& entries) const {
        for (const auto& [views, ix] : entries) {
            std::apply(f, views);
        }
    }

public:
    /** Call `f` on each set of arrays but pass the arrays with their static type.

        `f` is not guaranteed to be called on the arrays in the order they were passed
        to the constructor. If you would like to know the original array index of the
        arguments, use `for_each_with_ix`.

        @param f A function object which can be called with with a signature of:
               `f(const py::ndarray_view<Sig, ndim>&...)` for each candidate signature
               in `Ts`.
     */
    template<typename F>
    void for_each(F&& f) const {
        std::apply(
            [&](const auto&... args) {
                (for_each_helper(std::forward<F>(f), args), ...);
            },
            m_arrays);
    }

private:
    template<typename F, typename E>
    void for_each_with_ix_helper(F&& f, const std::vector<E>& entries) const {
        for (const auto& [views, ix] : entries) {
            std::apply([&, &ix = ix](const auto&... args) { f(ix, args...); }, views);
        }
    }

public:
    /** Call `f` on each set of arrays but pass the arrays with their static type.

        `f` is not guaranteed to be called on the arrays in the order they were passed
        to the constructor. The `ix` argument provided will represent the index of the
        array views from the original constructor call.

        @param f A function object which can be called with with a signature of:
               `f(std::size_t ix, const py::ndarray_view<Sig, ndim>&...)` for each
               candidate signature in `Ts`.
     */
    template<typename F>
    void for_each_with_ix(F&& f) const {
        std::apply(
            [&](const auto&... args) {
                (for_each_with_ix_helper(std::forward<F>(f), args), ...);
            },
            m_arrays);
    }
};

/** nwise devirtualize for 1-dimensional array views.

    @tparam Ts A sequence of tuples of types that convey the possible combinations of
            arrays.
 */
template<typename... Ts>
using nwise_devirtualize = nwise_nddevirtualize<1, Ts...>;

/** Devirtualize for 1-dimensional array views.

    @tparam Ts The possible types of the array to devirtualize.
 */
template<typename... Ts>
using devirtualize = nwise_devirtualize<std::tuple<Ts>...>;

/** Devirtualize for ndimensional array views.

    @tparam Ts The possible types of the array to devirtualize.
 */
template<std::size_t ndim, typename... Ts>
using nddevirtualize = nwise_nddevirtualize<ndim, std::tuple<Ts>...>;
}  // namespace py
