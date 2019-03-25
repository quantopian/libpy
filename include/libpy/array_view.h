#pragma once

#include <algorithm>
#include <any>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <iterator>
#include <stdexcept>
#include <string_view>
#include <type_traits>

#include "libpy/any.h"

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

    std::ptrdiff_t pos_to_index(const std::array<std::size_t, ndim>& pos) const {
        std::ptrdiff_t ix = 0;

        for (std::size_t n = 0; n < ndim; ++n) {
            std::ptrdiff_t along_axis;
            if (__builtin_mul_overflow(pos[n], m_strides[n], &along_axis)) {
                throw std::overflow_error("pos * m_strides overflows std::ptrdiff_t");
            }
            if (__builtin_add_overflow(ix, along_axis, &ix)) {
                throw std::overflow_error("ix + along_axis overflows std::ptrdiff_t");
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

    /** Create a virtual array of length ``size`` holding the scalar ``value``.

        @param value The value to fill the array with.
        @param shape The shape of the array.
        @return The new mutable array view.
     */
    static ndarray_view virtual_array(T& value,
                                      const std::array<std::size_t, ndim>& shape) {
        return {reinterpret_cast<char*>(std::addressof(value)), shape, {0}};
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
    char* buffer() {
        return m_buffer;
    }

    /** The underlying buffer of characters for this string array.
     */
    const char* buffer() const {
        return m_buffer;
    }

    /** Create a new immutable view over the same memory.

        @return The frozen view.
    */
    ndarray_view<const T, ndim, true> freeze() const {
        return {m_buffer, m_shape, m_strides};
    }

    /** Check if two views are exactly identical.

        @param other The view to compare to.
        @return Are these views identical?
     */
    bool operator==(const ndarray_view& other) const {
        return m_buffer == other.m_buffer && m_shape == other.m_shape &&
               m_strides == other.m_strides;
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

    /** Iterator type to implement forward, const, reverse, and const reverse iterators.

        This type cannot be implemented with just a pointer and stride because stride may
        be zero, so an index is needed to count the number of iterations in that case.
     */
    template<typename V>
    struct generic_iterator {
    private:
        char* m_ptr;
        std::size_t m_ix;
        std::int64_t m_stride;

    protected:
        friend ndarray_view;

        generic_iterator(char* buffer, std::size_t ix, std::int64_t stride)
            : m_ptr(buffer), m_ix(ix), m_stride(stride) {}

    public:
        using difference_type = std::int64_t;
        using value_type = V;
        using pointer = value_type*;
        using const_pointer = const value_type*;
        using reference = value_type&;
        using const_reference = const value_type&;
        using iterator_category = std::random_access_iterator_tag;

        reference operator*() {
            return *reinterpret_cast<V*>(m_ptr);
        }

        const_reference operator*() const {
            return *reinterpret_cast<V*>(m_ptr);
        }

        reference operator[](difference_type ix) {
            return *(*this + ix);
        }

        const_reference operator[](difference_type ix) const {
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
            return generic_iterator(m_ptr + n * m_stride, m_ix + n, m_stride);
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
    using iterator = generic_iterator<typename generic_ndarray_impl::value_type>;
    using const_iterator =
        generic_iterator<const typename generic_ndarray_impl::value_type>;
    using reverse_iterator = iterator;
    using const_reverse_iterator = const_iterator;

    static constexpr std::size_t npos = -1;

    // Re-use constructor from the generic impl.
    using ndarray_view<T, 1, true>::ndarray_view;

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

    /** Create a virtual array of length ``size`` holding the scalar ``value``.

        @param value The value to fill the array with.
        @param shape The shape of the array.
        @return The new mutable array view.
     */
    static ndarray_view virtual_array(T& value, const std::array<std::size_t, 1>& shape) {
        return {reinterpret_cast<char*>(std::addressof(value)), shape, {0}};
    }

    /** Create a virtual array of length ``size`` holding the scalar ``value``.

        @param value The value to fill the array with.
        @param size The size of the array.
        @return The new mutable array view.
     */
    static ndarray_view virtual_array(T& value, std::size_t size) {
        return {reinterpret_cast<char*>(std::addressof(value)), {size}, {0}};
    }

    iterator begin() {
        return iterator(this->m_buffer, 0, this->m_strides[0]);
    }

    const_iterator cbegin() {
        return const_iterator(this->m_buffer, 0, this->m_strides[0]);
    }

    const_iterator begin() const {
        return const_iterator(this->m_buffer, 0, this->m_strides[0]);
    }

    iterator end() {
        return iterator(this->m_buffer + this->pos_to_index(this->m_shape),
                        this->size(),
                        this->m_strides[0]);
    }

    const_iterator end() const {
        return const_iterator(this->m_buffer + this->pos_to_index(this->m_shape),
                              this->size(),
                              this->m_strides[0]);
    }

    const_iterator cend() {
        return const_iterator(this->m_buffer + this->pos_to_index(this->m_shape),
                              this->size(),
                              this->m_strides[0]);
    }

    reverse_iterator rbegin() {
        return reverse_iterator(this->m_buffer + this->pos_to_index({this->size() - 1}),
                                0,
                                -this->m_strides[0]);
    }

    const_reverse_iterator crbegin() {
        return const_reverse_iterator(this->m_buffer +
                                          this->pos_to_index({this->size() - 1}),
                                      0,
                                      -this->m_strides[0]);
    }

    const_reverse_iterator rbegin() const {
        return const_reverse_iterator(this->m_buffer +
                                          this->pos_to_index({this->size() - 1}),
                                      0,
                                      -this->m_strides[0]);
    }

    reverse_iterator rend() {
        auto stride = -this->m_strides[0];
        return reverse_iterator(this->m_buffer + stride, this->size(), stride);
    }

    const_reverse_iterator rend() const {
        auto stride = -this->m_strides[0];
        return const_reverse_iterator(this->m_buffer + stride, this->size(), stride);
    }

    const_reverse_iterator crend() {
        auto stride = -this->m_strides[0];
        return const_reverse_iterator(this->m_buffer + stride, this->size(), stride);
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

    /** Access the first element of this array. The array must be non-empty.
     */
    const T& front() const {
        return (*this)[0];
    }

    /** Access the first element of this array. The array must be non-empty.
     */
    T& front() {
        return (*this)[0];
    }

    /** Access the last element of this array. The array must be non-empty.
     */
    const T& back() const {
        return (*this)[this->size() - 1];
    }

    /** Access the last element of this array. The array must be non-empty.
     */
    T& back() {
        return (*this)[this->size() - 1];
    }

    /** Create a view over a subsection of the viewed memory.

        @param start The start index of the slice.
        @param stop The stop index of the slice, exclusive.
        @param step The value to increment each index by.
        @return A view over a subset of the memory.
    */
    ndarray_view slice(std::size_t start, std::size_t stop = npos, std::size_t step = 1) {
        std::size_t size = (stop == npos) ? this->m_shape[0] - start : stop - start;
        std::int64_t stride = this->m_strides[0] * step;
        return ndarray_view(this->m_buffer + this->pos_to_index({start}),
                            {size},
                            {stride});
    }

    /** Create a view over a subsection of the viewed memory.

        @param start The start index of the slice.
        @param stop The stop index of the slice, exclusive.
        @param step The value to increment each index by.
        @return A view over a subset of the memory.
     */
    ndarray_view<const T, 1, false>
    slice(std::size_t start, std::size_t stop = npos, std::size_t step = 1) const {
        std::size_t size = (stop == npos) ? this->m_shape[0] - start : stop - start;
        std::int64_t stride = this->m_strides[0] * step;
        return ndarray_view<const T, 1, false>(this->m_buffer +
                                                   this->pos_to_index({start}),
                                               {size},
                                               {stride});
    }

    /** Create a new immutable view over the same memory.

        @return The frozen view.
     */
    ndarray_view<const T, 1, false> freeze() const {
        return {this->m_buffer, this->m_shape, this->m_strides};
    }
};

template<typename T>
using array_view = ndarray_view<T, 1>;

namespace detail {
/** A struct to wrap an array whose type and shape is not known until runtime.

    @tparam T The type of the elements in the array.
    @tparam ndim The rank of the array.
 */
template<std::size_t ndim, typename T, bool higher_dimensional = ndim != 1>
class any_ref_ndarray_view {
protected:
    std::array<std::size_t, ndim> m_shape;
    std::array<std::int64_t, ndim> m_strides;
    char* m_buffer;
    any_ref_assign_func m_assign;

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
    using value_type = any_cref;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference = T;
    using const_reference = any_cref;
    using pointer = value_type*;
    using const_pointer = const value_type*;

    /** Create a virtual array of length ``size`` holding the scalar ``value``.

        @param value The value to fill the array with.
        @param shape The shape of the array.
        @return The new mutable array view.
     */
    template<typename U>
    static any_ref_ndarray_view
    virtual_array(U& value, const std::array<std::size_t, ndim>& shape) {
        return {reinterpret_cast<char*>(std::addressof(value)),
                shape,
                {0},
                &typeid(T),
                sizeof(T)};
    }

    /** Default constructor creates an empty view over nothing.
     */
    any_ref_ndarray_view()
        : m_shape({0}), m_strides({0}), m_buffer(nullptr), m_assign(nullptr) {}

    /** Take a view over `buffer`.

        @param buffer The buffer to take a view over.
        @param shape The number of elements in the buffer along each axis.
        @param stride The number of bytes between elements along each axis.
     */
    any_ref_ndarray_view(char* buffer,
                         const std::array<std::size_t, ndim> shape,
                         const std::array<std::int64_t, ndim>& strides,
                         any_ref_assign_func assign)
        : m_shape(shape), m_strides(strides), m_buffer(buffer), m_assign(assign) {}

    /** Access the element at the given index with bounds checking.

        @param pos The index to lookup.
        @return A view of the string at the given index.
     */
    template<typename... Ixs>
    const_reference at(Ixs... pos) const {
        return at({pos...});
    }

    const_reference at(const std::array<std::size_t, ndim>& ixs) const {
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
    reference at(Ixs... pos) {
        return at({pos...});
    }

    reference at(const std::array<std::size_t, ndim>& ixs) {
        for (std::size_t n = 0; n < ndim; ++n) {
            if (ixs[n] >= m_shape[n]) {
                throw std::out_of_range("pos exceeds the length of the array");
            }
        }

        return (*this)(ixs);
    }

    template<typename... Ixs>
    const_reference operator()(Ixs... ixs) const {
        return (*this)({ixs...});
    }

    const_reference operator()(const std::array<std::size_t, ndim>& ixs) const {
        return {&m_buffer[pos_to_index(ixs)], m_assign};
    }

    template<typename... Ixs>
    reference operator()(Ixs... ixs) {
        return (*this)({ixs...});
    }

    reference operator()(const std::array<std::size_t, ndim>& ixs) {
        return {&m_buffer[pos_to_index(ixs)], m_assign};
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

    /** Cast the array to a static type.
     */
    template<typename U>
    ndarray_view<U, ndim, higher_dimensional> cast() {
        if (&any_ref_assign<U> != m_assign) {
            throw std::bad_any_cast{};
        }

        return {m_buffer, m_shape, m_strides};
    }

    /** Cast the array to a static type.
     */
    template<typename U>
    ndarray_view<const U, ndim, higher_dimensional> cast() const {
        if (&any_ref_assign<U> != m_assign) {
            throw std::bad_any_cast{};
        }

        return {m_buffer, m_shape, m_strides};
    }

    /** The underlying buffer for this array view.
     */
    char* buffer() {
        return m_buffer;
    }

    /** The underlying buffer of characters for this string array.
     */
    const char* buffer() const {
        return m_buffer;
    }

    /** Check if two views are exactly identical.

        @param other The view to compare to.
        @return Are these views identical?
     */
    bool operator==(const any_ref_ndarray_view& other) const {
        return m_buffer == other.m_buffer && m_shape == other.m_shape &&
               m_strides == other.m_strides && m_assign == other.m_assign;
    }
};

template<typename T>
class any_ref_ndarray_view<1, T, false> : public any_ref_ndarray_view<1, T, true> {
private:
    using generic_ndarray_impl = any_ref_ndarray_view<1, T, true>;

    /** Iterator type to implement forward, const, reverse, and const reverse iterators.

        This type cannot be implemented with just a pointer and stride because stride may
        be zero, so an index is needed to count the number of iterations in that case.
     */
    template<typename R, typename C>
    struct generic_iterator {
    private:
        char* m_ptr;
        std::size_t m_ix;
        std::int64_t m_stride;
        any_ref_assign_func m_assign;

    protected:
        friend any_ref_ndarray_view;

        generic_iterator(char* buffer,
                         std::size_t ix,
                         std::int64_t stride,
                         any_ref_assign_func assign)
            : m_ptr(buffer), m_ix(ix), m_stride(stride), m_assign(assign) {}

    public:
        using difference_type = std::int64_t;
        using value_type = C;
        using pointer = C*;
        using reference = R;
        using const_reference = C;
        using iterator_category = std::random_access_iterator_tag;

        reference operator*() {
            return {m_ptr, m_assign};
        }

        const_reference operator*() const {
            return {m_ptr, m_assign};
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
            return {m_ptr + n * m_stride, m_ix + n, m_stride, m_assign};
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
    using reference = typename generic_ndarray_impl::reference;
    using const_reference = typename generic_ndarray_impl::const_reference;

    using iterator = generic_iterator<reference, const_reference>;
    using const_iterator = generic_iterator<const_reference, const_reference>;
    using reverse_iterator = iterator;
    using const_reverse_iterator = const_iterator;

    static constexpr std::size_t npos = -1;

    // Re-use constructor from the generic impl.
    using any_ref_ndarray_view<1, T, true>::any_ref_ndarray_view;

    any_ref_ndarray_view(const any_ref_ndarray_view& cpfrom)
        : any_ref_ndarray_view<1, T, true>::any_ref_ndarray_view(cpfrom) {}

    /** Create a view over an arbitrary contiguous container of `T`s.

        @param contiguous_container The container to take a view of.
     */
    template<typename C, typename U = typename C::value_type>
    any_ref_ndarray_view(C& contiguous_container)
        : generic_ndarray_impl(reinterpret_cast<char*>(contiguous_container.data()),
                               {contiguous_container.size()},
                               {sizeof(U)},
                               &any_ref_assign<U>) {}

    /** Create a virtual array of length ``size`` holding the scalar ``value``.

        @param value The value to fill the array with.
        @param shape The shape of the array.
        @return The new mutable array view.
     */
    template<typename U>
    static any_ref_ndarray_view virtual_array(U& value,
                                              const std::array<std::size_t, 1>& shape) {
        return {reinterpret_cast<char*>(std::addressof(value)),
                shape,
                {0},
                &any_ref_assign<U>};
    }

    /** Create a virtual array of length ``size`` holding the scalar ``value``.

        @param value The value to fill the array with.
        @param size The size of the array.
        @return The new mutable array view.
     */
    template<typename U>
    static any_ref_ndarray_view virtual_array(U& value, std::size_t size) {
        return {reinterpret_cast<char*>(std::addressof(value)),
                {size},
                {0},
                &any_ref_assign<U>};
    }

    iterator begin() {
        return {this->m_buffer, 0, this->m_strides[0], this->m_assign};
    }

    const_iterator cbegin() {
        return {this->m_buffer, 0, this->m_strides[0], this->m_assign};
    }

    const_iterator begin() const {
        return {this->m_buffer, 0, this->m_strides[0], this->m_assign};
    }

    iterator end() {
        return {this->m_buffer + this->pos_to_index(this->m_shape),
                this->size(),
                this->m_strides[0],
                this->m_assign};
    }

    const_iterator end() const {
        return {this->m_buffer + this->pos_to_index(this->m_shape),
                this->size(),
                this->m_strides[0],
                this->m_assign};
    }

    const_iterator cend() {
        return {this->m_buffer + this->pos_to_index(this->m_shape),
                this->size(),
                this->m_strides[0],
                this->m_assign};
    }

    reverse_iterator rbegin() {
        return {this->m_buffer + this->pos_to_index({this->size() - 1}),
                0,
                -this->m_strides[0],
                this->m_assign};
    }

    const_reverse_iterator crbegin() {
        return {this->m_buffer + this->pos_to_index({this->size() - 1}),
                0,
                -this->m_strides[0],
                this->m_assign};
    }

    const_reverse_iterator rbegin() const {
        return {this->m_buffer + this->pos_to_index({this->size() - 1}),
                0,
                -this->m_strides[0],
                this->m_assign};
    }

    reverse_iterator rend() {
        auto stride = -this->m_strides[0];
        return {this->m_buffer + stride, this->size(), stride, this->m_assign};
    }

    const_reverse_iterator rend() const {
        auto stride = -this->m_strides[0];
        return {this->m_buffer + stride, this->size(), stride, this->m_assign};
    }

    const_reverse_iterator crend() {
        auto stride = -this->m_strides[0];
        return {this->m_buffer + stride, this->size(), stride, this->m_assign};
    }

    /**  Access the element at the given index without bounds checking.

         @param pos The index to lookup.
         @return A view of the string at the given index. Undefined if `pos` is
                 out of bounds.
     */
    const_reference operator[](std::size_t pos) const {
        return {&this->m_buffer[this->pos_to_index({pos})], this->m_assign};
    }

    /**  Access the element at the given index without bounds checking.

         @param pos The index to lookup.
         @return A view of the string at the given index. Undefined if `pos` is
                 out of bounds.
     */
    reference operator[](std::size_t pos) {
        return {&this->m_buffer[this->pos_to_index({pos})], this->m_assign};
    }

    /** Access the first element of this array. The array must be non-empty.
     */
    const T& front() const {
        return (*this)[0];
    }

    /** Access the first element of this array. The array must be non-empty.
     */
    T& front() {
        return (*this)[0];
    }

    /** Access the last element of this array. The array must be non-empty.
     */
    const T& back() const {
        return (*this)[this->size() - 1];
    }

    /** Access the last element of this array. The array must be non-empty.
     */
    T& back() {
        return (*this)[this->size() - 1];
    }

    /** Cast the array to a static type.
     */
    template<typename U>
    ndarray_view<U, 1, false> cast() {
        if (&any_ref_assign<U> != this->m_assign) {
            throw std::bad_any_cast{};
        }

        return {this->m_buffer, this->m_shape, this->m_strides};
    }

    /** Cast the array to a static type.
     */
    template<typename U>
    ndarray_view<const U, 1, false> cast() const {
        if (&any_ref_assign<U> != this->m_assign) {
            throw std::bad_any_cast{};
        }

        return {this->m_buffer, this->m_shape, this->m_strides};
    }
};
}  // namespace detail

template<>
class ndarray_view<any_cref, 1, false>
    : public detail::any_ref_ndarray_view<1, any_cref, false> {
private:
    using generic_ndarray_impl = detail::any_ref_ndarray_view<1, any_cref, false>;

public:
    static constexpr std::size_t npos = generic_ndarray_impl::npos;

    // Re-use constructor from the generic impl.
    using any_ref_ndarray_view<1, any_cref, false>::any_ref_ndarray_view;

    /** Create a view over a subsection of the viewed memory.

    @param start The start index of the slice.
    @param stop The stop index of the slice, exclusive.
    @param step The value to increment each index by.
    @return A view over a subset of the memory.
 */
    ndarray_view slice(std::size_t start, std::size_t stop = npos, std::size_t step = 1) {
        std::size_t size = (stop == npos) ? this->m_shape[0] - start : stop - start;
        std::int64_t stride = this->m_strides[0] * step;
        return {this->m_buffer + this->pos_to_index({start}),
                {size},
                {stride},
                this->m_assign};
    }

    /** Create a view over a subsection of the viewed memory.

        @param start The start index of the slice.
        @param stop The stop index of the slice, exclusive.
        @param step The value to increment each index by.
        @return A view over a subset of the memory.
     */
    ndarray_view<any_cref, 1, false>
    slice(std::size_t start, std::size_t stop = npos, std::size_t step = 1) const {
        std::size_t size = (stop == npos) ? this->m_shape[0] - start : stop - start;
        std::int64_t stride = this->m_strides[0] * step;
        return {this->m_buffer + this->pos_to_index({start}),
                {size},
                {stride},
                this->m_assign};
    }

    /** Create a new immutable view over the same memory.

        @return The frozen view.
     */
    ndarray_view<any_cref, 1, false> freeze() const {
        return {this->m_buffer, this->m_shape, this->m_strides, this->m_assign};
    }
};

template<>
class ndarray_view<any_ref, 1, false>
    : public detail::any_ref_ndarray_view<1, any_ref, false> {
private:
    using generic_ndarray_impl = detail::any_ref_ndarray_view<1, any_ref, false>;

public:
    static constexpr std::size_t npos = generic_ndarray_impl::npos;

    // Re-use constructor from the generic impl.
    using any_ref_ndarray_view<1, any_ref, false>::any_ref_ndarray_view;

    /** Create a view over a subsection of the viewed memory.

    @param start The start index of the slice.
    @param stop The stop index of the slice, exclusive.
    @param step The value to increment each index by.
    @return A view over a subset of the memory.
 */
    ndarray_view slice(std::size_t start, std::size_t stop = npos, std::size_t step = 1) {
        std::size_t size = (stop == npos) ? this->m_shape[0] - start : stop - start;
        std::int64_t stride = this->m_strides[0] * step;
        return {this->m_buffer + this->pos_to_index({start}),
                {size},
                {stride},
                this->m_assign};
    }

    /** Create a view over a subsection of the viewed memory.

        @param start The start index of the slice.
        @param stop The stop index of the slice, exclusive.
        @param step The value to increment each index by.
        @return A view over a subset of the memory.
     */
    ndarray_view<any_cref, 1, false>
    slice(std::size_t start, std::size_t stop = npos, std::size_t step = 1) const {
        std::size_t size = (stop == npos) ? this->m_shape[0] - start : stop - start;
        std::int64_t stride = this->m_strides[0] * step;
        return {this->m_buffer + this->pos_to_index({start}),
                {size},
                {stride},
                this->m_assign};
    }

    /** Create a new immutable view over the same memory.

        @return The frozen view.
     */
    ndarray_view<any_cref, 1, false> freeze() const {
        return {this->m_buffer, this->m_shape, this->m_strides, this->m_assign};
    }
};

template<std::size_t ndim, bool higher_dimensional>
class ndarray_view<any_cref, ndim, higher_dimensional>
    : public detail::any_ref_ndarray_view<ndim, any_cref, higher_dimensional> {
private:
    using generic_ndarray_impl =
        detail::any_ref_ndarray_view<ndim, any_cref, higher_dimensional>;

public:
    static constexpr std::size_t npos = generic_ndarray_impl::npos;

    // Re-use constructor from the generic impl.
    using detail::any_ref_ndarray_view<ndim, any_cref, higher_dimensional>::
        any_ref_ndarray_view;

    /** Create a new immutable view over the same memory.

        @return The frozen view.
     */
    ndarray_view<any_cref, ndim, higher_dimensional> freeze() const {
        return {this->m_buffer, this->m_shape, this->m_strides, this->m_assign};
    }
};

template<std::size_t ndim, bool higher_dimensional>
class ndarray_view<any_ref, ndim, higher_dimensional>
    : public detail::any_ref_ndarray_view<ndim, any_ref, higher_dimensional> {
private:
    using generic_ndarray_impl =
        detail::any_ref_ndarray_view<ndim, any_ref, higher_dimensional>;

public:
    // Re-use constructor from the generic impl.
    using detail::any_ref_ndarray_view<ndim, any_ref, higher_dimensional>::
        any_ref_ndarray_view;

    /** Create a new immutable view over the same memory.

        @return The frozen view.
     */
    ndarray_view<any_cref, ndim, higher_dimensional> freeze() const {
        return {this->m_buffer, this->m_shape, this->m_strides, this->m_assign};
    }
};
}  // namespace py
