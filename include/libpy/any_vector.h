#pragma once

#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "libpy/any.h"
#include "libpy/meta.h"
#include "libpy/numpy_utils.h"

namespace py {
class any_vector {
private:
    py::any_vtable m_vtable;
    std::byte* m_storage;
    std::size_t m_size;
    std::size_t m_capacity;

    inline std::ptrdiff_t pos_to_index(std::ptrdiff_t pos) const {
        return pos * m_vtable.size();
    }

    template<typename T>
    void typecheck() const {
        if (any_vtable::make<T>() != m_vtable) {
            throw std::bad_any_cast{};
        }
    }

    template<typename T>
    void typecheck(const T&) const {
        typecheck<T>();
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
    void map_old_new(F&& f, std::byte* new_data, std::byte* old_data) {
        std::size_t itemsize = m_vtable.size();
        for (std::size_t ix = 0; ix < size(); ++ix) {
            f(new_data, old_data);
            new_data += itemsize;
            old_data += itemsize;
        }
    }

    template<typename T, typename F>
    void construct_into_buffer(std::byte* buf, std::size_t ix, T&& value, F&& construct) {
        std::byte* addr = buf + pos_to_index(ix);
        using raw = py::meta::remove_cvref<T>;
        if constexpr (std::is_same_v<raw, any_ref> || std::is_same_v<raw, any_cref>) {
            construct(addr, value.addr());
        }
        else {
            construct(addr, std::addressof(value));
        }
    }

    /** Grow the vector in place, `count` must be greater than `capacity`.
     */
    template<typename T, typename F>
    void grow(std::size_t count, T&& value, F&& construct) {
        if (!count) {
            count = 1;
        }

        std::byte* old_data = m_storage;
        std::byte* new_data = m_vtable.alloc(count);

        /* NOTE: We must copy the value before freeing the old buffer. If the
           new value is a reference into `*this`, then freeing the old buffer
           will invalidate the reference and we will use the value after it is
           destroyed and freed.

           If the value is noexcept move, then we still need to copy the value
           before moving out of the values in the old buffer. If the new value
           is a reference into `*this`, then if we move first we will copy the
           state of the moved-from value, which is not the proper semantics for
           `push_back`.
        */
        try {
            construct_into_buffer(new_data,
                                  size(),
                                  std::forward<T>(value),
                                  std::forward<F>(construct));
        }
        catch (...) {
            // free the new array and re-raise the exception without modifying
            // our state (`m_storage` nor `m_capacity`).
            m_vtable.free(new_data);
            throw;
        }

        if (m_vtable.is_trivially_copyable()) {
            if (size()) {
                // note: because it is UB to call `std::memcpy` with a `nullptr`
                // src even if size is 0 so we need to explicitly guard for our
                // first growth.
                std::memcpy(new_data, old_data, size() * m_vtable.size());
            }
        }
        else {
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
                std::byte* new_end;
                try {
                    map_old_new(
                        [&](std::byte* new_, std::byte* old) {
                            new_end = new_;
                            m_vtable.copy_construct(new_, old);
                        },
                        new_data,
                        old_data);
                }
                catch (...) {
                    // if the copying throws an exception, unwind the new vector but leave
                    // our original storage and capacity alone
                    for (std::byte* p = new_data; p != new_end; p += m_vtable.size()) {
                        m_vtable.destruct(p);
                    }

                    // free the new array and re-raise the exception without modifying our
                    // state (`m_storage` nor `m_capacity`).
                    m_vtable.free(new_data);
                    throw;
                }

                // now that we have successfully copied everything to the `new_data`, we
                // can go through and destruct the elements of `old_data`
                std::byte* p = old_data;
                std::size_t itemsize = m_vtable.size();
                for (std::size_t ix = 0; ix < size(); ++ix) {
                    m_vtable.destruct(p);
                    p += itemsize;
                }
            }
        }

        m_vtable.free(old_data);

        // if we make it here, we have successfully initialized `new_data` and unwound
        // `old_data`, now we can swap our new state over to the new array
        m_storage = new_data;
        m_capacity = count;
    }

    template<typename T, typename F>
    void push_back_impl(T&& value, F&& construct) {
        typecheck(value);

        if (size() == capacity()) {
            grow(capacity() * 2, std::forward<T>(value), std::forward<F>(construct));
        }
        else {
            construct_into_buffer(m_storage,
                                  size(),
                                  std::forward<T>(value),
                                  std::forward<F>(construct));
        }

        ++m_size;
    }

public:
    any_vector() = delete;

    /** Create an empty vector with the given vtable.

        @param vtable The vtable for the vector.
     */
    inline any_vector(const any_vtable& vtable)
        : m_vtable(vtable), m_storage(nullptr), m_size(0), m_capacity(0) {}

    /** Initialize the vector with `count` default constructed elements from the given
        vtable.

        @param vtable The vtable for the vector.
        @param count The number of elements to default construct.
     */
    inline any_vector(const any_vtable& vtable, std::size_t count)
        : m_vtable(vtable),
          m_storage(vtable.default_construct_alloc(count)),
          m_size(count),
          m_capacity(count) {}

private:
    template<typename T>
    void fill_constant_trivially_copyable(std::size_t count, const T& value) {
        const std::byte* addr;
        std::size_t size;
        if constexpr (std::is_same_v<T, any_ref> || std::is_same_v<T, any_cref>) {
            addr = reinterpret_cast<const std::byte*>(value.addr());
            size = value.vtable().size();
        }
        else {
            addr = reinterpret_cast<const std::byte*>(std::addressof(value));
            size = sizeof(value);
        }

        bool constant_bitpattern = true;
        for (std::size_t ix = 1; ix < size; ++ix) {
            if (addr[ix] != addr[0]) {
                constant_bitpattern = false;
                break;
            }
        }

        std::size_t itemsize = m_vtable.size();

        if (constant_bitpattern) {
            // memset is much more efficient for bulk copying
            std::memset(m_storage, static_cast<int>(addr[0]), count * itemsize);
        }
        else {
            std::byte* end = m_storage + count * itemsize;
            auto memcpy_optimized = [&](std::size_t itemsize) {
                for (std::byte* data = m_storage; data < end; data += itemsize) {
                    std::memcpy(data, addr, itemsize);
                }
            };

            switch (itemsize) {
            case 2:
                memcpy_optimized(2);
                break;
            case 4:
                memcpy_optimized(4);
                break;
            case 8:
                memcpy_optimized(8);
                break;
            default:
                memcpy_optimized(itemsize);
            }
        }
    }

    template<typename AnyRefType>
    void fill_anyref(std::size_t count, const AnyRefType& value) {
        if (m_vtable.is_trivially_copy_constructible()) {
            fill_constant_trivially_copyable(count, value);
        }
        else {
            std::size_t itemsize = m_vtable.size();
            std::byte* data = m_storage;
            std::byte* end = data + count * itemsize;

            try {
                for (; data < end; data += itemsize) {
                    m_vtable.copy_construct(data, value.addr());
                }
            }
            catch (...) {
                // if an exception occurs default constructing the vector, be sure to
                // unwind the partially initialized state
                for (std::byte* p = m_storage; p < data; p += itemsize) {
                    m_vtable.destruct(p);
                }
                m_vtable.free(m_storage);
                throw;
            }
        }
    }

public:
    /** Initialize an any vector with `count` copies of `value`.

        @param vtable The vtable for the vector.
        @param count The number of elements to fill.
        @param value The value to copy from to fill the vector.
     */
    template<typename T>
    inline any_vector(const any_vtable& vtable, std::size_t count, const T& value)
        : m_vtable(vtable),
          m_storage(vtable.alloc(count)),
          m_size(count),
          m_capacity(count) {
        try {
            typecheck(value);

            if constexpr (std::is_same_v<T, any_ref> || std::is_same_v<T, any_cref>) {
                fill_anyref(count, value);
            }
            else {
                std::size_t itemsize = m_vtable.size();
                std::byte* data = m_storage;
                std::byte* end = data + count * itemsize;
                try {
                    for (; data < end; data += itemsize) {
                        new (data) T(value);
                    }
                }
                catch (...) {
                    for (std::byte* p = m_storage; p < data; p += itemsize) {
                        m_vtable.destruct(p);
                    }
                    throw;
                }
            }
        }
        catch (...) {
            m_vtable.free(m_storage);
            throw;
        }
    }

    inline any_vector(const any_vector& cpfrom)
        : m_vtable(cpfrom.m_vtable),
          m_storage(cpfrom.vtable().alloc(cpfrom.size())),
          m_size(cpfrom.size()),
          m_capacity(cpfrom.size()) {
        if (m_vtable.is_trivially_copy_constructible()) {
            std::memcpy(m_storage, cpfrom.m_storage, m_vtable.size() * size());
        }
        else {
            std::size_t itemsize = m_vtable.size();
            std::byte* new_data = m_storage;
            std::byte* old_data = cpfrom.m_storage;
            std::size_t ix;

            try {
                for (ix = 0; ix < size(); ++ix) {
                    m_vtable.copy_construct(new_data, old_data);
                    new_data += itemsize;
                    old_data += itemsize;
                }
            }
            catch (...) {
                for (std::size_t unwind_ix = 0; unwind_ix < ix; ++unwind_ix) {
                    m_vtable.destruct(m_storage + itemsize * unwind_ix);
                }
                m_vtable.free(m_storage);
                throw;
            }
        }
    }

    inline any_vector(any_vector&& mvfrom) noexcept
        : m_vtable(mvfrom.m_vtable),
          m_storage(mvfrom.m_storage),
          m_size(mvfrom.size()),
          m_capacity(mvfrom.capacity()) {
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

    /** Swap the contents of this vector with another. This can change the `vtable`
        of the two vectors.

        @param other The vector to swap contents with.
     */
    inline void swap(any_vector& other) noexcept {
        std::swap(m_vtable, other.m_vtable);
        std::swap(m_storage, other.m_storage);
        std::swap(m_size, other.m_size);
        std::swap(m_capacity, other.m_capacity);
    }

    inline ~any_vector() {
        clear();
        m_vtable.free(m_storage);
    }

    using reference = any_ref;
    using const_reference = any_cref;
    using iterator = generic_iterator<std::byte*, any_ref, any_cref>;
    using reverse_iterator = iterator;
    using const_iterator = generic_iterator<const std::byte*, any_cref, any_cref>;
    using const_reverse_iterator = const_iterator;

    inline iterator begin() {
        return {m_storage, static_cast<std::int64_t>(m_vtable.size()), m_vtable};
    }

    inline const_iterator begin() const {
        return {m_storage, static_cast<std::int64_t>(m_vtable.size()), m_vtable};
    }

    inline const_iterator cbegin() const {
        return begin();
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

    inline const_iterator cend() const {
        return end();
    }

    /** Get the number of elements in this vector.
     */
    inline std::size_t size() const {
        return m_size;
    }

    /** Get the number of elements this vector can store before requiring a resize.
     */
    inline std::size_t capacity() const {
        return m_capacity;
    }

    inline reference operator[](std::ptrdiff_t pos) {
        return {&m_storage[pos_to_index(pos)], m_vtable};
    }

    /** Runtime bounds checked accessor for the array.

        @param pos The index into the array to access.
        @return An any reference to the element.
        @throws std::out_of_range when `pos` is out of bounds for the array.
     */
    inline reference at(std::ptrdiff_t pos) {
        if (pos < 0 || static_cast<size_t>(pos) >= size()) {
            throw std::out_of_range("pos out of bounds");
        }

        return (*this)[pos];
    }

    inline const_reference operator[](std::ptrdiff_t pos) const {
        return {&m_storage[pos_to_index(pos)], m_vtable};
    }

    /** Runtime bounds checked accessor for the array.

        @param pos The index into the array to access.
        @return A const any reference to the element.
        @throws std::out_of_range when `pos` is out of bounds for the array.
     */
    inline const_reference at(std::ptrdiff_t pos) const {
        if (pos < 0 || static_cast<size_t>(pos) >= size()) {
            throw std::out_of_range("pos out of bounds");
        }

        return (*this)[pos];
    }

    /** Get an any reference to the first element of this vector.

        @pre `size() > 0`
     */
    inline reference front() {
        return (*this)[0];
    }

    /** Get a const any reference to the first element of this vector.

        @pre `size() > 0`
     */
    inline const_reference front() const {
        return (*this)[0];
    }

    /** Get an any reference to the last element of this vector.

        @pre `size() > 0`
     */
    inline reference back() {
        return (*this)[size() - 1];
    }

    /** Get a const any reference to the last element of this vector.

        @pre `size() > 0`
     */
    inline const_reference back() const {
        return (*this)[size() - 1];
    }

    /** Clear all of the elements in the vector. This destroys all of the contained
        objects.

        @post `size() == 0`
        @post Invalidates all iterators.
     */
    inline void clear() {
        if (!m_vtable.is_trivially_destructible()) {
            std::size_t itemsize = m_vtable.size();
            std::byte* data = m_storage;
            for (std::size_t ix = 0; ix < size(); ++ix) {
                m_vtable.destruct(data);
                data += itemsize;
            }
        }
        m_size = 0;
    }

    /** Add an element to this vector.

        @param value The value to copy into the vector.
        @post Invalidates iterators if `size() == capacity()` on entry.
        @note This function has a strong exception guarantee.
     */
    template<typename T>
    void push_back(const T& value) {
        push_back_impl(value, [this](void* new_, const void* old) {
            m_vtable.copy_construct(new_, old);
        });
    }

    /** Add an element to this vector.

        @param value The value to move into the vector.
        @post Invalidates iterators if `size() == capacity()` on entry.
        @note This function has a strong exception guarantee.
     */
    template<typename T>
    void push_back(T&& value) {
        using raw = py::meta::remove_cvref<T>;
        if constexpr (std::is_same_v<raw, any_ref> || std::is_same_v<raw, any_cref>) {
            push_back_impl(value, [this](void* new_, const void* old) {
                m_vtable.copy_construct(new_, old);
            });
        }
        else {
            push_back_impl(std::forward<T>(value), [this](void* new_, void* old) {
                m_vtable.move_construct(new_, old);
            });
        }
    }

    /** Remove and destroy the last element from the vector.

        @pre `size() > 0`.
     */
    inline void pop_back() {
        m_vtable.destruct(m_storage + pos_to_index(size() - 1));
        --m_size;
    }

    /** Get the underlying buffer for this vector.

        @see py::array_view::array_view(const py::any_vector&)

        This pointer cannot be cast to a `T[]` or a `T*` and used as an array. Pointer
        arithmetic may be done on this buffer to access individual elements as a `T*`.

        Correct:

        \code
        // first element
        T& ref = *reinterpret_cast<T*>(vec.buffer());

        // nth element
        T& ref = *reinterpret_cast<T*>(vec.buffer()[sizeof(T) * n]);
        \endcode

        Do all indexing on the `std::byte*` because `buffer()` returns a pointer to
        the first element of a `std::byte[]`.

        Incorrect:

        \code
        T& ref = reinterpret_cast<T*>(vec.buffer())[n];
        \endcode

        `operator[]` can only be used with a `T*` when the pointer points to a
        member of a `T[]`. `buffer()` returns the array of bytes that
        _provides storage_ for the elements, but no `T[]` exists.
     */
    inline std::byte* buffer() noexcept {
        return m_storage;
    }

    /** Get the underlying buffer for this vector.

        @see py::array_view::array_view(const py::any_vector&)

        This pointer cannot be cast to a `T[]` or a `T*` and used as an array. Pointer
        arithmetic may be done on this buffer to access individual elements as a
        `T*`.

        Correct:

        \code
        // first element
        const T& ref = *reinterpret_cast<const T*>(vec.buffer());

        // nth element
        const T& ref = *reinterpret_cast<const T*>(vec.buffer()[sizeof(T) * n]);
        \endcode

        Do all indexing on the `std::byte*` because `buffer()` returns a pointer to
        the first element of a `std::byte[]`.

        Incorrect:

        \code
        T& ref = reinterpret_cast<T*>(vec.buffer())[n];
        \endcode

        `operator[]` can only be used with a `T*` when the pointer points to a
        member of a `T[]`. `buffer()` returns the array of bytes that
        _provides storage_ for the elements, but no `T[]` exists.
     */
    inline const std::byte* buffer() const noexcept {
        return m_storage;
    }

    /** Get the vtable for this vector.
     */
    inline const any_vtable& vtable() const {
        return m_vtable;
    }
};

inline owned_ref<> move_to_numpy_array(py::any_vector&& values) {

    auto descr = values.vtable().new_dtype();
    if (!descr) {
        return nullptr;
    }
    return py::move_to_numpy_array<py::any_vector, 1>(std::move(values),
                                                      std::move(descr),
                                                      {values.size()},
                                                      {static_cast<std::int64_t>(
                                                          values.vtable().size())});
}
}  // namespace py
