#pragma once

#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "libpy/any.h"

namespace py {
class any_vector {
private:
    py::any_vtable m_vtable;
    char* m_storage;
    std::size_t m_size;
    std::size_t m_capacity;

    inline std::ptrdiff_t pos_to_index(std::ptrdiff_t pos) const {
        return pos * m_vtable.size();
    }

    template<typename T>
    void typecheck(const T&) const {
        if (any_vtable::make<T>() != m_vtable) {
            throw std::bad_any_cast{};
        }
    }

    inline void typecheck(const any_ref& other) const {
        if (m_vtable != other.vtable()) {
            throw std::bad_any_cast{};
        }
    }

    inline void typecheck(const any_cref& other) const {
        if (m_vtable != other.vtable()) {
            throw std::bad_any_cast{};
        }
    }

    template<typename ptr, typename R, typename C>
    struct generic_iterator {
    protected:
        friend any_vector;

        ptr m_ptr;
        std::int64_t m_stride;
        any_vtable m_vtable;

        generic_iterator(ptr buffer, std::int64_t stride, any_vtable vtable)
            : m_ptr(buffer), m_stride(stride), m_vtable(vtable) {}

    public:
        using difference_type = std::ptrdiff_t;
        using value_type = C;
        using pointer = C*;
        using reference = R;
        using const_reference = C;
        using iterator_category = std::random_access_iterator_tag;

        reference operator*() {
            return {m_ptr, m_vtable};
        }

        const_reference operator*() const {
            return {m_ptr, m_vtable};
        }

        reference operator[](difference_type ix) {
            return *(*this + ix);
        }

        const_reference operator[](difference_type ix) const {
            return *(*this + ix);
        }

        generic_iterator& operator++() {
            m_ptr += m_stride;
            return *this;
        }

        generic_iterator operator++(int) {
            generic_iterator out = *this;
            m_ptr += m_stride;
            return out;
        }

        generic_iterator& operator+=(difference_type n) {
            m_ptr += m_stride * n;
            return *this;
        }

        generic_iterator operator+(difference_type n) const {
            return {m_ptr + n * m_stride, m_stride, m_vtable};
        }

        generic_iterator& operator--() {
            m_ptr -= m_stride;
            return *this;
        }

        generic_iterator operator--(int) {
            generic_iterator out = *this;
            m_ptr -= m_stride;
            return out;
        }

        generic_iterator& operator-=(difference_type n) {
            m_ptr -= m_stride * n;
            return *this;
        }

        difference_type operator-(const generic_iterator& other) const {
            return (m_ptr - other.m_ptr) / m_stride;
        }

        bool operator!=(const generic_iterator& other) const {
            return m_ptr != other.m_ptr;
        }

        bool operator==(const generic_iterator& other) const {
            return m_ptr == other.m_ptr;
        }

        bool operator<(const generic_iterator& other) const {
            return m_ptr < other.m_ptr;
        }

        bool operator<=(const generic_iterator& other) const {
            return m_ptr <= other.m_ptr;
        }

        bool operator>(const generic_iterator& other) const {
            return m_ptr > other.m_ptr;
        }

        bool operator>=(const generic_iterator& other) const {
            return m_ptr >= other.m_ptr;
        }
    };

    template<typename F>
    void map_old_new(F&& f, char* new_data, char* old_data) {
        std::size_t itemsize = m_vtable.size();
        for (std::size_t ix = 0; ix < size(); ++ix) {
            f(new_data, old_data);
            new_data += itemsize;
            old_data += itemsize;
        }
    }

    /** Grow the vector in place, `count` must be greater than `capacity`.
     */
    inline void grow(std::size_t count) {
        if (!count) {
            count = 1;
        }

        char* new_data;
        char* old_data = m_storage;

        if (m_vtable.is_trivially_copyable()) {
            // the data is trivially copyable, so we will attempt to grow the current
            // allocation falling back to `std::memcpy`.
            if (m_vtable.align() <= alignof(std::max_align_t)) {
                new_data = static_cast<char*>(
                    std::realloc(m_storage, m_vtable.size() * count));
                if (!new_data) {
                    throw std::bad_alloc{};
                }
            }
            else {
                // `std::realloc` uses `std::malloc` when it can't grow the allocation,
                // this doesn't respect any explicit over-alignment so it is only safe for
                // types whose `align()` is less than or equal to that of the
                // `max_align_t`.
                new_data = static_cast<char*>(m_vtable.alloc(count));
                if (!new_data) {
                    throw std::bad_alloc{};
                }
                std::memcpy(new_data, old_data, size() * m_vtable.size());
                std::free(old_data);
            }
        }
        else {
            new_data = static_cast<char*>(m_vtable.alloc(count));
            if (!new_data) {
                throw std::bad_alloc{};
            }

            if (m_vtable.move_is_noexcept() && m_vtable.is_trivially_destructible()) {
                // move can't throw and the object doesn't need to be destructed: just
                // move from the old to the new and free the old in one big chunk
                map_old_new([&](void* new_,
                                void* old) { m_vtable.move_construct(new_, old); },
                            new_data,
                            old_data);
            }
            else if (m_vtable.move_is_noexcept()) {
                // move can't throw, but the object has a non trivial destructor:
                // move from the old to the new and then destruct the old right away
                map_old_new(
                    [&](void* new_, void* old) {
                        m_vtable.move_construct(new_, old);
                        m_vtable.destruct(old);
                    },
                    new_data,
                    old_data);
            }
            else {
                // Move can throw so we need to copy from the old to the new.
                // If copying throws, we need to destruct all of successfully copied
                // objects in the new array

                // track one past the last successfully copied element, if an exception
                // occurs, we need to destruct everything up to, but not including, this
                // object
                char* new_end;
                try {
                    map_old_new(
                        [&](char* new_, char* old) {
                            new_end = new_;
                            m_vtable.copy_construct(new_, old);
                        },
                        new_data,
                        old_data);
                }
                catch (...) {
                    // if the copying throws an exception, unwind the new vector but leave
                    // our original storage and capacity alone
                    for (char* p = new_data; p != new_end; p += m_vtable.size()) {
                        m_vtable.destruct(p);
                    }

                    // free the new array and re-raise the exception without modifying our
                    // state (`m_storage` nor `m_capacity`).
                    std::free(new_data);
                    throw;
                }

                // now that we have successfully copied everything to the `new_data`, we
                // can go through and destruct the elements of `old_data`
                char* p = old_data;
                std::size_t itemsize = m_vtable.size();
                for (std::size_t ix = 0; ix < size(); ++ix) {
                    m_vtable.destruct(p);
                    p += itemsize;
                }
            }

            std::free(old_data);
        }

        // if we make it here, we have successfully initialized `new_data` and unwound
        // `old_data`, now we can swap our new state over to the new array
        m_storage = new_data;
        m_capacity = count;
    }

    template<typename F, typename T>
    void push_back(F&& construct, T&& value) {
        typecheck(value);

        if (size() == capacity()) {
            grow(capacity() * 2);
        }

        char* addr = m_storage + pos_to_index(size());
        if constexpr (std::is_same_v<T, any_ref> || std::is_same_v<T, any_cref>) {
            construct(addr, value.addr());
        }
        else {
            construct(addr, std::addressof(value));
        }

        ++m_size;
    }

public:
    any_vector() = delete;

    inline any_vector(const any_vtable& vtable, std::size_t count = 0)
        : m_vtable(vtable),
          m_storage(static_cast<char*>(vtable.default_construct_alloc(count))),
          m_size(count),
          m_capacity(count) {

        if (!m_storage) {
            throw std::bad_alloc{};
        }

        if (!m_vtable.is_trivially_default_constructible()) {
            char* data = m_storage;
            std::size_t itemsize = m_vtable.size();
            try {
                for (std::size_t ix = 0; ix < count; ++ix) {
                    vtable.default_construct(data);
                    data += itemsize;
                }
            }
            catch (...) {
                // if an exception occurs default constructing the vector, be sure to
                // unwind the partially initialized state
                for (char* p = m_storage; p < data; p += itemsize) {
                    vtable.destruct(p);
                }
                std::free(m_storage);
                throw;
            }
        }
    }

    template<typename T>
    inline any_vector(const any_vtable& vtable, std::size_t count, const T& value)
        : m_vtable(vtable),
          m_storage(static_cast<char*>(vtable.alloc(count))),
          m_size(count),
          m_capacity(count) {

        if (!m_storage) {
            throw std::bad_alloc{};
        }

        typecheck(value);

        std::size_t itemsize = m_vtable.size();
        char* data = m_storage;
        try {
            for (std::size_t ix = 0; ix < count; ++ix) {
                if constexpr (std::is_same_v<T, any_ref> || std::is_same_v<T, any_cref>) {
                    m_vtable.copy_construct(data, value.addr());
                }
                else {
                    m_vtable.copy_construct(data, std::addressof(value));
                }
                data += itemsize;
            }
        }
        catch (...) {
            // if an exception occurs default constructing the vector, be sure to
            // unwind the partially initialized state
            for (char* p = m_storage; p < data; p += itemsize) {
                vtable.destruct(p);
            }
            std::free(m_storage);
            throw;
        }
    }

    inline any_vector(const any_vector& cpfrom)
        : m_vtable(cpfrom.m_vtable),
          m_storage(static_cast<char*>(cpfrom.vtable().alloc(cpfrom.size()))),
          m_size(cpfrom.size()),
          m_capacity(cpfrom.size()) {

        if (!m_storage) {
            throw std::bad_alloc{};
        }

        if (m_vtable.is_trivially_copy_constructible()) {
            std::memcpy(m_storage, cpfrom.m_storage, m_vtable.size() * size());
        }
        else {
            std::size_t itemsize = m_vtable.size();
            char* new_data = m_storage;
            char* old_data = cpfrom.m_storage;
            for (std::size_t ix = 0; ix < size(); ++ix) {
                m_vtable.copy_construct(new_data, old_data);
                new_data += itemsize;
                old_data += itemsize;
            }
        }
    }

    inline any_vector(any_vector&& mvfrom) noexcept
        : m_vtable(mvfrom.m_vtable),
          m_storage(mvfrom.m_storage),
          m_size(mvfrom.size()),
          m_capacity(mvfrom.size()) {
        mvfrom.m_storage = nullptr;
        mvfrom.m_size = 0;
        mvfrom.m_capacity = 0;
    }

    inline any_vector& operator=(const any_vector& cpfrom) {
        any_vector cp(cpfrom);
        swap(cp);
        return *this;
    }

    inline any_vector& operator=(any_vector&& mvfrom) noexcept {
        swap(mvfrom);
        return *this;
    }

    inline void swap(any_vector& other) noexcept {
        std::swap(m_vtable, other.m_vtable);
        std::swap(m_storage, other.m_storage);
        std::swap(m_size, other.m_size);
        std::swap(m_capacity, other.m_capacity);
    }

    inline ~any_vector() {
        clear();
        std::free(m_storage);
    }

    using reference = any_ref;
    using const_reference = any_cref;
    using iterator = generic_iterator<char*, any_ref, any_cref>;
    using reverse_iterator = iterator;
    using const_iterator = generic_iterator<const char*, any_cref, any_cref>;
    using const_reverse_iterator = const_iterator;

    inline iterator begin() {
        return {m_storage, static_cast<std::int64_t>(m_vtable.size()), m_vtable};
    }

    inline const_iterator begin() const {
        return {m_storage, static_cast<std::int64_t>(m_vtable.size()), m_vtable};
    }

    inline iterator end() {
        return {m_storage + pos_to_index(size()),
                static_cast<std::int64_t>(m_vtable.size()),
                m_vtable};
    }

    inline const_iterator end() const {
        return {m_storage + pos_to_index(size()),
                static_cast<std::int64_t>(m_vtable.size()),
                m_vtable};
    }

    inline std::size_t size() const {
        return m_size;
    }

    inline std::size_t capacity() const {
        return m_capacity;
    }

    inline reference operator[](std::ptrdiff_t pos) {
        return {&m_storage[pos_to_index(pos)], m_vtable};
    }

    inline reference at(std::ptrdiff_t pos) {
        if (pos < 0 || static_cast<size_t>(pos) >= size()) {
            throw std::out_of_range("pos out of bounds");
        }

        return (*this)[pos];
    }

    inline const_reference operator[](std::ptrdiff_t pos) const {
        return {&m_storage[pos_to_index(pos)], m_vtable};
    }

    inline const_reference at(std::ptrdiff_t pos) const {
        if (pos < 0 || static_cast<size_t>(pos) >= size()) {
            throw std::out_of_range("pos out of bounds");
        }

        return (*this)[pos];
    }

    inline reference front() {
        return (*this)[0];
    }

    inline const_reference front() const {
        return (*this)[0];
    }

    inline reference back() {
        return (*this)[size() - 1];
    }

    inline const_reference back() const {
        return (*this)[size() - 1];
    }

    inline void clear() {
        if (!m_vtable.is_trivially_destructible()) {
            std::size_t itemsize = m_vtable.size();
            char* data = m_storage;
            for (std::size_t ix = 0; ix < size(); ++ix) {
                m_vtable.destruct(data);
                data += itemsize;
            }
        }
        m_size = 0;
    }

    template<typename T>
    void push_back(const T& value) {
        push_back([&](void* new_,
                      const void* old) { m_vtable.copy_construct(new_, old); },
                  value);
    }

    template<typename T>
    void push_back(T&& value) {
        push_back([&](void* new_, void* old) { m_vtable.move_construct(new_, old); },
                  std::move(value));
    }

    inline void pop_back() {
        m_vtable.destruct(m_storage + pos_to_index(size() - 1));
        --m_size;
    }

    inline void* data() noexcept {
        return m_storage;
    }

    inline const void* data() const noexcept {
        return m_storage;
    }

    inline const any_vtable& vtable() const {
        return m_vtable;
    }
};
}  // namespace py
