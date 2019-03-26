#pragma once

#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "libpy/any.h"

namespace py {
class any_vector {
private:
    py::any_ref_vtable m_vtable;
    char* m_storage;
    std::size_t m_size;
    std::size_t m_capacity;

    inline std::ptrdiff_t pos_to_index(std::ptrdiff_t pos) const {
        return pos * m_vtable.align();
    }

    template<typename T>
    void typecheck(const T&) const {
        if (any_ref_vtable::make<T>() != m_vtable) {
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
        any_ref_vtable m_vtable;

        generic_iterator(ptr buffer, std::int64_t stride, any_ref_vtable vtable)
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

    void realloc(std::size_t count) {
        char* new_data = static_cast<char*>(
            aligned_alloc(m_vtable.align(), m_vtable.align() * count));
        if (!new_data) {
            throw std::bad_alloc{};
        }
        char* old_data = m_storage;

        if (m_vtable.is_trivially_copy_constructible() ||
            m_vtable.is_trivially_move_constructible()) {
            std::memcpy(new_data, old_data, m_vtable.align() * size());
        }
        else {
            std::size_t itemsize = m_vtable.align();
            for (std::size_t ix = 0; ix < size(); ++ix) {
                m_vtable.move_if_noexcept(new_data, old_data);
                m_vtable.destruct(old_data);
                old_data += itemsize;
                new_data += itemsize;
            }
        }

        std::free(m_storage);
        m_capacity = count;
    }

public:
    any_vector() = delete;
    inline any_vector(const any_ref_vtable& vtable)
        : m_vtable(vtable), m_storage(nullptr), m_size(0), m_capacity(0) {}

    inline any_vector(const any_ref_vtable& vtable, std::size_t count)
        : m_vtable(vtable),
          m_storage(
              static_cast<char*>(aligned_alloc(vtable.align(), vtable.align() * count))),
          m_size(count),
          m_capacity(count) {

        if (!m_storage) {
            throw std::bad_alloc{};
        }

        if (!m_vtable.is_trivially_default_constructible()) {
            std::size_t itemsize = m_vtable.align();
            char* data = m_storage;
            for (std::size_t ix = 0; ix < size(); ++ix) {
                vtable.default_construct(data);
                data += itemsize;
            }
        }
    }

    template<typename T>
    inline any_vector(const any_ref_vtable& vtable, std::size_t count, const T& value)
        : m_vtable(vtable),
          m_storage(static_cast<char*>(
              std::aligned_alloc(vtable.align(), vtable.align() * count))),
          m_size(count),
          m_capacity(count) {

        if (!m_storage) {
            throw std::bad_alloc{};
        }

        typecheck(value);

        std::size_t itemsize = m_vtable.align();
        char* data = m_storage;
        for (std::size_t ix = 0; ix < size(); ++ix) {
            if constexpr (std::is_same_v<T, any_ref> || std::is_same_v<T, any_cref>) {
                m_vtable.copy_construct(data, value.addr());
            }
            else {
                m_vtable.copy_construct(data, std::addressof(value));
            }
            data += itemsize;
        }
    }

    inline any_vector(const any_vector& cpfrom)
        : m_vtable(cpfrom.m_vtable),
          m_storage(static_cast<char*>(
              std::aligned_alloc(cpfrom.m_vtable.align(),
                                 cpfrom.m_vtable.align() * cpfrom.size()))),
          m_size(cpfrom.size()),
          m_capacity(cpfrom.size()) {

        if (!m_storage) {
            throw std::bad_alloc{};
        }

        if (!m_vtable.is_trivially_copy_constructible()) {
            std::memcpy(m_storage, cpfrom.m_storage, m_vtable.align() * size());
        }
        else {
            std::size_t itemsize = m_vtable.align();
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

    inline any_vector& operator=(any_vector&& mvfrom) noexcept {
        swap(mvfrom);
        return *this;
    }

    void swap(any_vector& other) noexcept {
        std::swap(m_vtable, other.m_vtable);
        std::swap(m_storage, other.m_storage);
        std::swap(m_size, other.m_size);
        std::swap(m_capacity, other.m_capacity);
    }

    inline ~any_vector() {
        clear();
        if (m_storage) {
            std::free(m_storage);
        }
    }

    using reference = any_ref;
    using const_reference = any_cref;
    using iterator = generic_iterator<char*, any_ref, any_cref>;
    using reverse_iterator = iterator;
    using const_iterator = generic_iterator<const char*, any_cref, any_cref>;
    using const_reverse_iterator = const_iterator;

    inline iterator begin() {
        return {m_storage, static_cast<std::int64_t>(m_vtable.align()), m_vtable};
    }

    inline const_iterator begin() const {
        return {m_storage, static_cast<std::int64_t>(m_vtable.align()), m_vtable};
    }

    inline iterator end() {
        return {m_storage + size() * m_vtable.align(),
                static_cast<std::int64_t>(m_vtable.size()),
                m_vtable};
    }

    inline const_iterator end() const {
        return {m_storage + size() * m_vtable.align(),
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
            std::size_t itemsize = m_vtable.align();
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
        typecheck(value);

        if (size() == capacity()) {
            realloc(capacity() * 2);
        }

        char* addr = m_storage + size();
        if constexpr (std::is_same_v<T, any_ref> || std::is_same_v<T, any_cref>) {
            m_vtable.copy_construct(addr, value.addr());
        }
        else {
            m_vtable.copy_construct(addr, std::addressof(value));
        }

        ++m_size;
    }

    template<typename T>
    void push_back(T&& value) {
        typecheck(value);

        if (size() == capacity()) {
            realloc(capacity() * 2);
        }

        char* addr = m_storage + size();
        if constexpr (std::is_same_v<T, any_ref> || std::is_same_v<T, any_cref>) {
            m_vtable.move_construct(addr, value.addr());
        }
        else {
            m_vtable.move_construct(addr, std::addressof(value));
        }

        ++m_size;
    }

    inline void pop_back() {
        m_vtable.destruct(m_storage + size() - 1);
        --m_size;
    }

    inline void* data() noexcept {
        return m_storage;
    }

    inline const void* data() const noexcept {
        return m_storage;
    }

    inline const any_ref_vtable& vtable() const {
        return m_vtable;
    }
};
}  // namespace py
