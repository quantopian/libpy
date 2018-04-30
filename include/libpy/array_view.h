#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <iterator>
#include <stdexcept>
#include <string_view>
#include <type_traits>

namespace py {
/** A struct to wrap an array of type T whose shape is not known until runtime.

    @tparam T The type of the elements in the array.
    @tparam ndim The rank of the array.
 */
template<typename T, std::size_t ndim, bool = ndim != 1>
class ndarray_view {
protected:
    std::array<std::size_t, ndim> m_shape;
    std::array<std::int64_t, ndim> m_strides;
    char* m_buffer;

    std::size_t pos_to_index(const std::array<std::size_t, ndim>& pos) const {
        std::size_t ix = 0;

        for (std::size_t n = 0; n < ndim; ++n) {
            std::size_t along_axis;
            if (__builtin_mul_overflow(pos[n], m_strides[n], &along_axis)) {
                throw std::overflow_error("pos * m_strides overflows std::size_t");
            }
            if (__builtin_add_overflow(ix, along_axis, &ix)) {
                throw std::overflow_error("ix + along_axis overflows std::size_t");
            }
        }

        return ix;
    }

public:
    // expose member types to look more like a `std::array`.
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference = value_type&;
    using const_reference = const value_type&;
    using pointer = value_type*;
    using const_pointer = const value_type*;

    static constexpr std::size_t npos = -1;

    /** Create a virtual array of length ``size`` holding the scalar ``value``.

        @param value The value to fill the array with.
        @param size The size of the array.
        @return The new mutable array view.
     */
    static ndarray_view virtual_array(T& value, std::size_t size) {
        return {reinterpret_cast<char*>(std::addressof(value)), size, 0};
    }

    /** Default constructor creates an empty view over nothing.
     */
    ndarray_view() : m_shape({0}), m_strides({0}), m_buffer(nullptr) {}

    /** Take a view over `buffer`.

        @param buffer The buffer to take a view over.
        @param shape The number of elements in the buffer along each axis.
        @param stride The number of bytes between elements along each axis.
     */
    ndarray_view(char* buffer,
                 const std::array<std::size_t, ndim> shape,
                 const std::array<std::int64_t, ndim>& strides)
        : m_shape(shape), m_strides(strides), m_buffer(buffer) {
        // assert the pointer is aligned
        assert(reinterpret_cast<std::size_t>(buffer) % alignof(T) == 0);
    }

    ndarray_view(const ndarray_view& cpfrom)
        : m_shape(cpfrom.m_shape), m_strides(cpfrom.m_strides), m_buffer(cpfrom.m_buffer) {}

    ndarray_view& operator=(const ndarray_view& cpfrom) {
        m_shape = cpfrom.m_shape;
        m_strides = cpfrom.m_strides;
        m_buffer = cpfrom.m_buffer;
        return *this;
    }

    /** Access the element at the given index with bounds checking.

        @param pos The index to lookup.
        @return A view of the string at the given index.
     */
    template<typename... Ixs>
    const T& at(Ixs... pos) const {
        return at({pos...});
    }

    const T& at(const std::array<std::size_t, ndim>& ixs) const {
        for (std::size_t n = 0; n < ndim; ++n) {
            if (ixs[n] >= m_shape[n]) {
                throw std::out_of_range("pos exceeds the length of the array");
            }
        }

        return (*this)(ixs);
    }

    /** Access the element at the given index with bounds checking.

        @param pos The index to lookup.
        @return A view of the string at the given index.
     */
    template<typename... Ixs>
    T& at(Ixs... pos) {
        return at({pos...});
    }

    T& at(const std::array<std::size_t, ndim>& ixs) {
        for (std::size_t n = 0; n < ndim; ++n) {
            if (ixs[n] >= m_shape[n]) {
                throw std::out_of_range("pos exceeds the length of the array");
            }
        }

        return (*this)(ixs);
    }

    template<typename... Ixs>
    const T& operator()(Ixs... ixs) const {
        return (*this)({ixs...});
    }

    const T& operator()(const std::array<std::size_t, ndim>& ixs) const {
        return *reinterpret_cast<T*>(&m_buffer[pos_to_index(ixs)]);
    }

    template<typename... Ixs>
    T& operator()(Ixs... ixs) {
        return (*this)({ixs...});
    }

    T& operator()(const std::array<std::size_t, ndim>& ixs) {
        return *reinterpret_cast<T*>(&m_buffer[pos_to_index(ixs)]);
    }

    /** Create a view over a subsection of the viewed memory.

        @param start The start index of the slice.
        @param stop The stop index of the slice, exclusive.
        @param step The value to increment each index by.
        @return A view over a subset of the memory.
     */
    ndarray_view slice(std::size_t start, std::size_t stop = npos, std::size_t step = 1) {
        std::array<std::size_t, ndim> new_shape;
        std::array<std::int64_t, ndim> new_strides;

        new_shape[0] = (stop == npos) ? m_shape[0] - start : stop - start;
        new_strides[0] = m_strides[0] * step;

        for (std::size_t ix = 1; ix < ndim; ++ix) {
            new_shape[ix - 1] = m_shape[ix];
            new_strides[ix - 1] = m_strides[ix];
        }

        return ndarray_view(&m_buffer[pos_to_index(start)], new_shape, new_strides);
    }

    /** Create a view over a subsection of the viewed memory.

        @param start The start index of the slice.
        @param stop The stop index of the slice, exclusive.
        @param step The value to increment each index by.
        @return A view over a subset of the memory.
     */
    const ndarray_view
    slice(std::size_t start, std::size_t stop = npos, std::size_t step = 1) const {
        return const_cast<ndarray_view*>(this)->slice(start, stop, step);
    }

    /** The number of elements in this array.
     */
    std::size_t size() const {
        return m_shape[0];
    }

    constexpr static std::size_t rank() {
        return ndim;
    }

    /** The number of bytes to go from one element to the next.
     */
    const std::array<std::size_t, ndim>& shape() const {
        return m_strides;
    }

    /** The number of bytes to go from one element to the next.
     */
    const std::array<std::int64_t, ndim>& strides() const {
        return m_strides;
    }

    /** The underlying buffer for this array view.
     */
    T* data() {
        return m_buffer;
    }

    /** The underlying buffer of characters for this string array.
     */
    const T* data() const {
        return m_buffer;
    }
};

/** A struct to wrap an array of type T whose shape is not known until runtime.

    @tparam T The type of the elements in the array.
    @tparam ndim The rank of the array.
 */
template<typename T>
class ndarray_view<T, 1, false> : public ndarray_view<T, 1, true> {
private:
    using generic_ndarray_impl = ndarray_view<T, 1, true>;

    template<typename V>
    struct generic_iterator {
    private:
        char* m_ptr;
        std::size_t m_stride;

    protected:
        friend ndarray_view;

        generic_iterator(char* buffer, std::size_t stride)
            : m_ptr(buffer), m_stride(stride) {}

    public:
        using difference_type = std::size_t;
        using value_type = V;
        using pointer = value_type*;
        using const_pointer = const value_type*;
        using reference = value_type&;
        using iterator_category = std::random_access_iterator_tag;

        reference operator*() {
            return *reinterpret_cast<V*>(m_ptr);
        }

        const reference operator*() const {
            return *reinterpret_cast<V*>(m_ptr);
        }

        reference operator[](difference_type ix) {
            return *(*this + ix);
        }

        const reference operator[](difference_type ix) const {
            return *(*this + ix);
        }

        pointer operator->() {
            return reinterpret_cast<V*>(m_ptr);
        }

        const_pointer operator->() const {
            return reinterpret_cast<const V*>(m_ptr);
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
            return generic_iterator(m_ptr + n * m_stride, m_stride);
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

public:
    using iterator = generic_iterator<typename generic_ndarray_impl::value_type>;
    using const_iterator =
        generic_iterator<const typename generic_ndarray_impl::value_type>;
    using reverse_iterator = iterator;
    using const_reverse_iterator = const_iterator;

    /** Create a view over an arbitrary contiguous container of `T`s.

        @param contiguous_container The container to take a view of.
     */
    template<typename C,
             typename =
                 std::enable_if_t<std::is_same_v<decltype(std::declval<C>().data()), T*>>>
    ndarray_view(C& contiguous_container)
        : generic_ndarray_impl(reinterpret_cast<char*>(contiguous_container.data()),
                               {contiguous_container.size()},
                               {sizeof(T)}) {}

    iterator begin() {
        return iterator(this->m_buffer, this->m_strides[0]);
    }

    const_iterator cbegin() {
        return const_iterator(this->m_buffer, this->m_strides[0]);
    }

    const_iterator begin() const {
        return const_iterator(this->m_buffer, this->m_strides[0]);
    }

    iterator end() {
        return iterator(this->m_buffer + this->pos_to_index(this->m_shape),
                        this->m_strides[0]);
    }

    const_iterator end() const {
        return const_iterator(this->m_buffer + this->pos_to_index(this->m_shape),
                              this->m_strides[0]);
    }

    const_iterator cend() {
        return const_iterator(this->m_buffer + this->pos_to_index(this->m_shape),
                              this->m_strides[0]);
    }

    reverse_iterator rbegin() {
        return reverse_iterator(this->m_buffer + this->pos_to_index({this->size() - 1}),
                                -this->m_strides[0]);
    }

    const_reverse_iterator crbegin() {
        return const_reverse_iterator(this->m_buffer +
                                          this->pos_to_index({this->size() - 1}),
                                      -this->m_strides[0]);
    }

    const_reverse_iterator rbegin() const {
        return const_reverse_iterator(this->m_buffer +
                                          this->pos_to_index({this->size() - 1}),
                                      -this->m_strides[0]);
    }

    reverse_iterator rend() {
        auto stride = -this->m_strides[0];
        return reverse_iterator(this->m_buffer + stride, stride);
    }

    const_reverse_iterator rend() const {
        auto stride = -this->m_strides[0];
        return const_reverse_iterator(this->m_buffer + stride, stride);
    }

    const_reverse_iterator crend() {
        auto stride = -this->m_strides[0];
        return const_reverse_iterator(this->m_buffer + stride, stride);
    }

    /**  Access the element at the given index without bounds checking.

         @param pos The index to lookup.
         @return A view of the string at the given index. Undefined if `pos` is
                 out of bounds.
     */
    const T& operator[](std::size_t pos) const {
        return *reinterpret_cast<T*>(&this->m_buffer[this->pos_to_index({pos})]);
    }

    /**  Access the element at the given index without bounds checking.

         @param pos The index to lookup.
         @return A view of the string at the given index. Undefined if `pos` is
                 out of bounds.
     */
    T& operator[](std::size_t pos) {
        return *reinterpret_cast<T*>(&this->m_buffer[this->pos_to_index({pos})]);
    }

};

template<typename T>
using array_view = ndarray_view<T, 1>;
}  // namespace py
