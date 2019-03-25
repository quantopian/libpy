#pragma once

#include <stdexcept>
#include <vector>

#include "libpy/any.h"

namespace py {
class any_vector final {
private:
    py::any_ref_vtable m_vtable;
    std::vector<char> m_storage;

    std::ptrdiff_t pos_to_index(std::ptrdiff_t pos) const {
        std::ptrdiff_t ix;
        if (__builtin_mul_overflow(pos, m_vtable.size(), &ix)) {
            throw std::overflow_error("pos * m_strides overflows std::ptrdiff_t");
        }

        return ix;
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

    template<typename R, typename C>
    struct generic_iterator {
    protected:
        friend any_vector;

        char* m_ptr;
        std::size_t m_ix;
        std::int64_t m_stride;
        any_ref_vtable m_vtable;

        generic_iterator(char* buffer,
                         std::size_t ix,
                         std::int64_t stride,
                         any_ref_vtable vtable)
            : m_ptr(buffer), m_ix(ix), m_stride(stride), m_vtable(vtable) {}

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
            m_ix += 1;
            return *this;
        }

        generic_iterator operator++(int) {
            generic_iterator out = *this;
            m_ptr += m_stride;
            m_ix += 1;
            return out;
        }

        generic_iterator& operator+=(difference_type n) {
            m_ptr += m_stride * n;
            m_ix += n;
            return *this;
        }

        generic_iterator operator+(difference_type n) const {
            return {m_ptr + n * m_stride, m_ix + n, m_stride, m_vtable};
        }

        generic_iterator& operator--() {
            m_ptr -= m_stride;
            m_ix -= 1;
            return *this;
        }

        generic_iterator operator--(int) {
            generic_iterator out = *this;
            m_ptr -= m_stride;
            m_ix -= 1;
            return out;
        }

        generic_iterator& operator-=(difference_type n) {
            m_ptr -= m_stride * n;
            m_ix -= n;
            return *this;
        }

        difference_type operator-(const generic_iterator& other) const {
            return m_ix - other.m_ix;
        }

        bool operator!=(const generic_iterator& other) const {
            return !(m_ix == other.m_ix && m_ptr == other.m_ptr);
        }

        bool operator==(const generic_iterator& other) const {
            return m_ix == other.m_ix && m_ptr == other.m_ptr;
        }

        bool operator<(const generic_iterator& other) const {
            return m_ix < other.m_ix;
        }

        bool operator<=(const generic_iterator& other) const {
            return m_ix <= other.m_ix;
        }

        bool operator>(const generic_iterator& other) const {
            return m_ix > other.m_ix;
        }

        bool operator>=(const generic_iterator& other) const {
            return m_ix >= other.m_ix;
        }
    };


public:
    any_vector() = delete;
    inline any_vector(const any_ref_vtable& vtable) : m_vtable(vtable) {}
    inline any_vector(const any_ref_vtable& vtable, std::size_t size)
        : m_vtable(vtable), m_storage(size * vtable.size()) {
        for (std::size_t pos = 0; pos < this->size(); ++pos) {
            vtable.default_construct(&m_storage[pos_to_index(pos)]);
        }
    }

    template<typename T>
    inline any_vector(const any_ref_vtable& vtable, std::size_t size, const T& value)
        : m_vtable(vtable), m_storage(size * vtable.size()) {
        typecheck(value);
        for (std::size_t pos = 0; pos < this->size(); ++pos) {
            auto addr = &m_storage[pos_to_index(pos)];
            if constexpr (std::is_same_v<T, any_ref> || std::is_same_v<T, any_cref>) {
                m_vtable.copy_construct(addr, value.addr());
            }
            else {
                m_vtable.copy_construct(addr, std::addressof(value));
            }
        }
    }

    inline ~any_vector() {
        std::size_t itemsize = m_vtable.size();
        char* data = m_storage.data();
        for (std::ptrdiff_t ix = 0; ix < static_cast<std::ptrdiff_t>(size()); ++ix) {
            m_vtable.destruct(data);
            data += itemsize;
        }
    }

    using reference = any_ref;
    using const_reference = any_cref;
    using iterator = generic_iterator<any_ref, any_cref>;
    using reverse_iterator = iterator;
    using const_iterator = generic_iterator<any_cref, any_cref>;
    using const_reverse_iterator = const_iterator;

    inline iterator begin() {
        return {m_storage.data(), 0, static_cast<std::int64_t>(m_vtable.size()), m_vtable};
    }

    inline const_iterator begin() const {
        return {const_cast<char*>(m_storage.data()),
                0,
                static_cast<std::int64_t>(m_vtable.size()),
                m_vtable};
    }

    inline iterator end() {
        return {m_storage.data(), size(), static_cast<std::int64_t>(m_vtable.size()), m_vtable};
    }

    inline const_iterator end() const {
        return {const_cast<char*>(m_storage.data()),
                size(),
                static_cast<std::int64_t>(m_vtable.size()),
                m_vtable};
    }

    inline std::size_t size() const {
        return m_storage.size() / m_vtable.size();
    }

    inline reference operator[](std::ptrdiff_t pos) {
        return {&m_storage[pos_to_index(pos)], m_vtable};
    }

    inline reference at(std::ptrdiff_t pos) {
        if (pos < 0 || static_cast<size_t>(pos) >= m_storage.size() / m_vtable.size()) {
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
        m_storage.clear();
    }

    template<typename T>
    void push_back(const T& value) {
        typecheck(value);
        void* addr = &m_storage.back();
        m_storage.insert(m_storage.end(), m_vtable.size(), ' ');
        if constexpr (std::is_same_v<T, any_ref> || std::is_same_v<T, any_cref>) {
            m_vtable.copy_construct(addr, value.addr());
        }
        else {
            m_vtable.copy_construct(addr, std::addressof(value));
        }
    }

    template<typename T>
    void push_back(T&& value) {
        typecheck(value);
        void* addr = &m_storage.back();
        m_storage.insert(m_storage.end(), m_vtable.size(), ' ');
        if constexpr (std::is_same_v<T, any_ref> || std::is_same_v<T, any_cref>) {
            m_vtable.move_construct(addr, value.addr());
        }
        else {
            m_vtable.move_construct(addr, std::addressof(value));
        }
    }

    inline void pop_back() {
        for (std::size_t b = 0; b < m_vtable.size(); ++b) {
            m_storage.pop_back();
        }
    }

    inline void* data() noexcept {
        return m_storage.data();
    }

    inline const void* data() const noexcept {
        return m_storage.data();
    }

    inline const any_ref_vtable& vtable() const {
        return m_vtable;
    }
};
}  // namespace py
