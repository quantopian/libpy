#pragma once

#include <functional>
#include <tuple>

#include "libpy/meta.h"
#include "libpy/util.h"

namespace py {
namespace detail {
template<typename... Ts>
class zipper {
private:
    std::tuple<Ts...> m_iterables;

    template<typename... Us>
    class generic_iterator {
    private:
        std::tuple<Us...> m_iterators;

    protected:
        friend class zipper;

        template<typename... Vs>
        constexpr generic_iterator(Vs&&... iterators)
            : m_iterators(std::forward<Vs>(iterators)...) {}

    public:
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::forward_iterator_tag;

        constexpr bool operator!=(const generic_iterator& other) const {
            return m_iterators != other.m_iterators;
        }

        constexpr bool operator==(const generic_iterator& other) const {
            return m_iterators == other.m_iterators;
        }

        constexpr generic_iterator& operator++() {
            std::apply([](Us&... iterators) { (..., ++iterators); }, m_iterators);
            return *this;
        }

        auto operator*() {
            return std::apply(
                [](Us&... iterators) {
                    return std::tuple<typename std::iterator_traits<Us>::reference...>(
                        *iterators...);
                },
                m_iterators);
        }
    };

public:
    using iterator =
        generic_iterator<py::meta::remove_cvref<decltype(std::declval<Ts>().begin())>...>;
    using const_iterator = generic_iterator<
        py::meta::remove_cvref<decltype(std::declval<const Ts>().begin())>...>;

    constexpr zipper(Ts&&... iterables) : m_iterables(std::forward<Ts>(iterables)...) {
        if (!py::util::all_equal(iterables.size()...)) {
            throw std::invalid_argument("iterables must be same length");
        }
    }

    constexpr iterator begin() {
        return std::apply(
            [](auto&... iterables) { return iterator(iterables.begin()...); },
            m_iterables);
    }

    constexpr iterator end() {
        return std::apply([](auto&... iterables) { return iterator(iterables.end()...); },
                          m_iterables);
    }

    constexpr const_iterator begin() const {
        return std::apply(
            [](auto&... iterables) { return iterator(iterables.begin()...); },
            m_iterables);
    }

    constexpr const_iterator end() const {
        return std::apply([](auto&... iterables) { return iterator(iterables.end()...); },
                          m_iterables);
    }
};
}  // namespace detail

/** Create an iterator that zips its inputs together. This is designed for pair-wise
    iteration over other iterables.

    This requires all the iterables to be the same length, otherwise an exception is
    thrown.
 */
template<typename... Ts>
auto zip(Ts&&... iterables) {
    return detail::zipper<Ts...>(std::forward<Ts>(iterables)...);
}

namespace detail {
template<typename T>
class enumerator {
private:
    T m_iterable;

    template<typename U>
    class generic_iterator {
    private:
        U m_iterator;
        std::size_t m_ix = 0;

    protected:
        friend class enumerator<T>;

        generic_iterator(U iterator) : m_iterator(iterator) {}

    public:
        bool operator!=(const generic_iterator& other) const {
            return m_iterator != other.m_iterator;
        }

        bool operator==(const generic_iterator& other) const {
            return m_iterator == other.m_iterator;
        }

        generic_iterator& operator++() {
            ++m_iterator;
            ++m_ix;
            return *this;
        }

        std::pair<std::size_t, decltype(*std::declval<U>())> operator*() {
            return {m_ix, *m_iterator};
        }
    };

public:
    using iterator = generic_iterator<decltype(std::declval<T>().begin())>;
    using const_iterator = generic_iterator<decltype(std::declval<const T>().begin())>;

    enumerator(T&& iterable) : m_iterable(std::forward<T>(iterable)) {}

    iterator begin() {
        return m_iterable.begin();
    }

    iterator end() {
        return m_iterable.end();
    }

    const_iterator begin() const {
        return m_iterable.begin();
    }

    const_iterator end() const {
        return m_iterable.end();
    }
};
}  // namespace detail

/** Create an iterator that iterates as pairs of {index, value}.
 */
template<typename T>
auto enumerate(T&& iterable) {
    return detail::enumerator<T>(std::forward<T>(iterable));
}

namespace detail {
template<typename F, typename T>
class imapper {
private:
    F m_func;
    T m_iterable;

    template<typename U>
    class generic_iterator {
    private:
        F m_func;
        U m_iterator;

    protected:
        friend class imapper;

        generic_iterator(F func, U iterator) : m_func(func), m_iterator(iterator) {}

    public:
        bool operator!=(const generic_iterator& other) const {
            return m_iterator != other.m_iterator;
        }

        bool operator==(const generic_iterator& other) const {
            return m_iterator == other.m_iterator;
        }

        generic_iterator& operator++() {
            ++m_iterator;
            return *this;
        }

        auto operator*() {
            return m_func(*m_iterator);
        }
    };

public:
    using iterator =
        generic_iterator<py::meta::remove_cvref<decltype(std::declval<T>().begin())>>;
    using const_iterator = generic_iterator<
        py::meta::remove_cvref<decltype(std::declval<const T>().begin())>>;

    imapper(F&& func, T&& iterable)
        : m_func(std::forward<F>(func)), m_iterable(std::forward<T>(iterable)) {}

    iterator begin() {
        return {m_func, m_iterable.begin()};
    }

    iterator end() {
        return {m_func, m_iterable.end()};
    }

    const_iterator begin() const {
        return {m_func, m_iterable.begin()};
    }

    const_iterator end() const {
        return {m_func, m_iterable.end()};
    }
};
}  // namespace detail

/** Create an iterator that lazily applies `f` to every element of `iterable`.

    \code
    for (auto v : py::imap(f, it)) {
    // ...
    }
    \endcode

    behaves the same as:

    \code
    for (auto& underlying : it) {
        auto v = f(underlying);
        // ...
    }
    \endcode

    @param f The function to apply.
    @param iterable The iterable to apply the function to.
 */
template<typename F, typename T>
auto imap(F&& f, T&& iterable) {
    return detail::imapper<F, T>(std::forward<F>(f), std::forward<T>(iterable));
}
}  // namespace py
