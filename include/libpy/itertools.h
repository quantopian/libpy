#pragma once

#include <functional>
#include <tuple>

namespace py {
namespace detail {
template<typename T, typename... Ts>
bool all_equal(T&& head, Ts&&... tail) {
    for (auto&& v : {tail...}) {
        if (head != v) {
            return false;
        }
    }
    return true;
}

inline bool all_equal() {
    return true;
}

template<typename... Ts>
class zipper {
private:
    std::tuple<Ts...> m_iterables;

    template<typename... Us>
    class generic_iterator {
    private:
        std::tuple<Us...> m_iterators;

    protected:
        friend class zipper<Ts...>;

        generic_iterator(Us... iterators) : m_iterators(iterators...) {}

    public:
        bool operator!=(const generic_iterator& other) const {
            return m_iterators != other.m_iterators;
        }

        bool operator==(const generic_iterator& other) const {
            return m_iterators == other.m_iterators;
        }

        generic_iterator& operator++() {
            std::apply([](Us&... iterators) { (..., ++iterators); }, m_iterators);
            return *this;
        }

        auto operator*() {
            return std::apply(
                [](Us&... iterators) { return std::forward_as_tuple(*iterators...); },
                m_iterators);
        }

        auto operator->() {
            return std::apply(
                [](Us&... iterators) {
                    return std::forward_as_tuple(iterators.operator->()...);
                },
                m_iterators);
        }
    };

public:
    using iterator = generic_iterator<decltype(std::declval<Ts>().begin())...>;
    using const_iterator =
        generic_iterator<decltype(std::declval<const Ts>().begin())...>;

    zipper(Ts&&... iterables) : m_iterables(std::forward<Ts>(iterables)...) {
        if (!detail::all_equal(iterables.size()...)) {
            throw std::invalid_argument("iterables must be same length");
        }
    }

    iterator begin() {
        return std::apply(
            [](auto&&... iterables) {
                return iterator(std::forward<Ts>(iterables).begin()...);
            },
            m_iterables);
    }

    iterator end() {
        return std::apply(
            [](auto&&... iterables) {
                return iterator(std::forward<Ts>(iterables).end()...);
            },
            m_iterables);
    }

    const_iterator begin() const {
        return std::apply(
            [](auto&&... iterables) {
                return iterator(std::forward<Ts>(iterables).begin()...);
            },
            m_iterables);
    }

    const_iterator end() const {
        return std::apply(
            [](auto&&... iterables) {
                return iterator(std::forward<Ts>(iterables).end()...);
            },
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
            return {m_ix, std::ref(*m_iterator)};
        }

        std::pair<std::size_t, decltype(std::declval<U>().operator->())> operator->() {
            return {m_ix, m_iterator.operator->()};
        }
    };

public:
    using iterator = generic_iterator<decltype(std::declval<T>().begin())>;
    using const_iterator =
        generic_iterator<decltype(std::declval<const T>().begin())>;

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
}  // namespace py
