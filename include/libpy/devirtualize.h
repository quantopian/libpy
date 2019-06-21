#pragma once

#include <tuple>
#include <vector>

#include "libpy/meta.h"
#include "libpy/ndarray_view.h"

namespace py {
/** Interface to provide typed access to a collection of type-erased array_views.

    @tparam arity The number of arrays to access at one time.
    @tparam Ts The types to provided optimized access through. If `py::any_ref` or
            `py::any_cref` appears in the list, it will catch any type of array_view.
 */
template<std::size_t arity, typename... Ts>
class nwise_devirtualize {
private:
    static_assert(arity > 0, "arity must be non-zero");

    template<typename T>
    using views = std::array<py::array_view<T>, arity>;

    template<typename T>
    using type_entry = std::pair<views<T>, std::size_t>;

    std::tuple<std::vector<type_entry<Ts>>...> m_arrays;

    template<std::size_t tuple_ix, typename Head, typename... Tail>
    void add_typed_entry(const views<py::any_ref>& args, std::size_t ix) {
        if constexpr (std::is_same_v<Head, py::any_ref> ||
                      std::is_same_v<Head, py::any_cref>) {

            std::get<tuple_ix>(m_arrays).emplace_back(args, ix);
        }
        else if (std::get<0>(args).vtable() == py::any_vtable::make<Head>()) {
            std::get<tuple_ix>(m_arrays).emplace_back(
                std::apply(
                    [](const auto&... untyped) {
                        return views<Head>{untyped.template cast<Head>()...};
                    },
                    args),
                ix);
        }
        else {
            add_typed_entry<tuple_ix + 1, Tail...>(args, ix);
        }
    }

    template<std::size_t>
    void add_typed_entry(const views<py::any_ref>&, std::size_t) {
        throw std::bad_any_cast{};
    }

    void add_entry(const views<py::any_ref>& args, std::size_t ix) {
        if (!std::apply([](auto... args) { return util::all_equal(args.vtable()...); },
                        args)) {
            throw std::bad_any_cast{};
        }

        add_typed_entry<0, Ts...>(args, ix);
    }

public:
    /** Construct an `nwise_devirtualize` from `arity` collections of
        `py::array_view<py::any_ref>` or `py::array_view<py::any_cref>`.

        @param array_collections One sequence of array views per `arity`.
     */
    template<typename... Args>
    nwise_devirtualize(const Args&... array_collections) {
        static_assert(sizeof...(Args) == arity, "Size of array_collections != arity");

        if (!util::all_equal(array_collections.size()...)) {
            throw std::invalid_argument{"array collections are not all the same size"};
        }

        std::size_t max_size = std::get<0>(std::make_tuple(array_collections...)).size();
        for (std::size_t ix = 0; ix < max_size; ++ix) {
            views<py::any_ref> items{array_collections[ix]...};
            add_entry(items, ix);
        }
    }

private:
    template<typename F, typename T>
    void for_each_helper(F&& f, const std::vector<type_entry<T>>& entries) {
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
               `f(const py::array_view<T>&...)` for each `T` in `Ts` and with an arity
               that matches `arity`.
     */
    template<typename F>
    void for_each(F&& f) {
        std::apply(
            [&](const auto&... args) {
                (for_each_helper(std::forward<F>(f), args), ...);
            },
            m_arrays);
    }

private:
    template<typename F, typename T>
    void for_each_with_ix_helper(F&& f, const std::vector<type_entry<T>>& entries) {
        for (const auto& [views, ix] : entries) {
            std::apply([&](const auto&... args) { f(ix, args...); }, views);
        }
    }

public:
    /** Call `f` on each set of arrays but pass the arrays with their static type.

        `f` is not guaranteed to be called on the arrays in the order they were passed
        to the constructor. The `ix` argument provided will represent the index of the
        array views from the original constructor call.

        @param f A function object which can be called with with a signature of:
               `f(std::size_t ix, const py::array_view<T>&...)` for each `T` in `Ts` and
               with an arity that matches `arity`.
     */
    template<typename F>
    void for_each_with_ix(F&& f) {
        std::apply(
            [&](const auto&... args) {
                (for_each_with_ix_helper(std::forward<F>(f), args), ...);
            },
            m_arrays);
    }
};

template<typename... Ts>
using devirtualize = nwise_devirtualize<1, Ts...>;
}  // namespace py
