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
#include "libpy/any_vector.h"
#include "libpy/exception.h"

namespace py {

namespace detail {
struct buffer_free {
    void operator()(Py_buffer* view) {
        if (view) {
            PyBuffer_Release(view);
            delete view;
        }
    }
};
}  // namespace detail

using buffer = std::unique_ptr<Py_buffer, detail::buffer_free>;

namespace detail {
template<typename T>
constexpr char buffer_format = '\0';

template<>
constexpr char buffer_format<char> = 'c';

template<>
constexpr char buffer_format<signed char> = 'b';

template<>
constexpr char buffer_format<unsigned char> = 'B';

template<>
constexpr char buffer_format<bool> = '?';

template<>
constexpr char buffer_format<short> = 'h';

template<>
constexpr char buffer_format<unsigned short> = 'H';

template<>
constexpr char buffer_format<int> = 'i';

template<>
constexpr char buffer_format<unsigned int> = 'I';

template<>
constexpr char buffer_format<long> = 'l';

template<>
constexpr char buffer_format<unsigned long> = 'L';

template<>
constexpr char buffer_format<long long> = 'q';

template<>
constexpr char buffer_format<unsigned long long> = 'Q';

template<>
constexpr char buffer_format<float> = 'f';

template<>
constexpr char buffer_format<double> = 'd';
}  // namespace detail

/** A struct to wrap an array of type T whose shape is not known until runtime.

    @tparam T The type of the elements in the array.
    @tparam ndim The rank of the array.
 */
template<typename T, std::size_t ndim, bool = ndim != 1>
class ndarray_view {
public:
    using buffer_type =
        std::conditional_t<std::is_const_v<T>, const std::byte*, std::byte*>;

protected:
    // allow any `ndarray_view` specialization to call the protected constructor
    // of other `ndarray_view` specializations
    template<typename, std::size_t, bool>
    friend class ndarray_view;

    std::array<std::size_t, ndim> m_shape;
    std::array<std::int64_t, ndim> m_strides;
    buffer_type m_buffer;

    std::ptrdiff_t pos_to_index(const std::array<std::size_t, ndim>& pos) const {
        std::ptrdiff_t ix = 0;
        for (std::size_t n = 0; n < ndim; ++n) {
            ix += pos[n] * m_strides[n];
        }
        return ix;
    }

    ndarray_view(buffer_type buffer,
                 const std::array<std::size_t, ndim> shape,
                 const std::array<std::int64_t, ndim>& strides)
        : m_shape(shape), m_strides(strides), m_buffer(buffer) {}

public:
    // expose member types to look more like a `std::array`.
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference = value_type&;
    using const_reference = const value_type&;
    using pointer = value_type*;
    using const_pointer = const value_type*;

    template<typename = std::enable_if_t<detail::buffer_format<T> != '\0'>>
    static std::tuple<ndarray_view<T, ndim>, py::buffer> from_buffer_protocol(PyObject* ob) {
        py::buffer buf(new Py_buffer);
        int flags = PyBUF_ND | PyBUF_STRIDED | PyBUF_FORMAT;
        if (!std::is_const_v<T>) {
            flags |= PyBUF_WRITABLE;
        }
        if (PyObject_GetBuffer(ob, buf.get(), flags)) {
            throw py::exception{};
        }

        char fmt;
        if (!buf->format) {
            fmt = 'B';
        }
        else if (std::strlen(buf->format) != 1) {
            throw py::exception(PyExc_TypeError,
                                "cannot adapt buffer of format=",
                                buf->format,
                                " to an ndarray_view of ",
                                util::type_name<T>().get());
        }
        else {
            fmt = *buf->format;
        }

        if (fmt != detail::buffer_format<T>) {
            throw py::exception(PyExc_TypeError,
                                "cannot adapt buffer of format=",
                                fmt,
                                " to an ndarray_view of ",
                                util::type_name<T>().get());
        }

        if (buf->ndim != ndim) {
            throw py::exception(
                PyExc_TypeError, "buffer ndim=", buf->ndim, " != expected_ndim=", ndim);
        }

        std::array<std::size_t, ndim> shape;
        std::array<std::int64_t, ndim> strides;
        for (int ix = 0; ix < buf->ndim; ++ix) {
            shape[ix] = static_cast<std::size_t>(buf->shape[ix]);
            strides[ix] = static_cast<std::int64_t>(buf->strides[ix]);
        }

        return {ndarray_view<T, ndim>{reinterpret_cast<buffer_type>(buf->buf),
                                      shape,
                                      strides},
                std::move(buf)};
    }

    template<typename = std::enable_if_t<detail::buffer_format<T> != '\0'>>
    static std::tuple<ndarray_view, py::buffer>
    from_buffer_protocol(const py::scoped_ref<>& ob) {
        return from_buffer_protocol(ob.get());
    }

    ndarray_view& operator=(const ndarray_view<const T, ndim, false>& cpfrom) {
        m_shape = cpfrom.shape();
        m_strides = cpfrom.strides();
        m_buffer = cpfrom.buffer();
        return *this;
    }

    /** Create a virtual array of length ``size`` holding the scalar ``value``.

        @param value The value to fill the array with.
        @param shape The shape of the array.
        @return The new mutable array view.
     */
    static ndarray_view virtual_array(T& value,
                                      const std::array<std::size_t, ndim>& shape) {
        return {std::addressof(value), shape, {0}};
    }

    /** Default constructor creates an empty view over nothing.
     */
    ndarray_view() noexcept : m_shape({0}), m_strides({0}), m_buffer(nullptr) {}

    /** Take a view over `buffer`.

        @param buffer The buffer to take a view over.
        @param shape The number of elements in the buffer along each axis.
        @param stride The number of bytes between elements along each axis.
     */
    ndarray_view(T* buffer,
                 const std::array<std::size_t, ndim> shape,
                 const std::array<std::int64_t, ndim>& strides)
        : ndarray_view(reinterpret_cast<buffer_type>(buffer), shape, strides) {}

    /** Access the element at the given index with bounds checking.

        @param pos The index to lookup.
        @return A view of the string at the given index.
     */
    template<typename... Ixs>
    T& at(Ixs... pos) const {
        return at({pos...});
    }

    /** Access the element at the given index with bounds checking.

        @param ixs The index to lookup.
        @return A view of the string at the given index.
     */
    T& at(const std::array<std::size_t, ndim>& ixs) const {
        for (std::size_t n = 0; n < ndim; ++n) {
            if (ixs[n] >= m_shape[n]) {
                throw std::out_of_range("pos exceeds the length of the array");
            }
        }

        return (*this)(ixs);
    }

    template<typename... Ixs>
    T& operator()(Ixs... ixs) const {
        return (*this)({ixs...});
    }

    T& operator()(const std::array<std::size_t, ndim>& ixs) const {
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
        return m_shape;
    }

    /** The number of bytes to go from one element to the next.
     */
    const std::array<std::int64_t, ndim>& strides() const {
        return m_strides;
    }

    /** The underlying buffer of characters for this string array.
     */
    buffer_type buffer() const {
        return m_buffer;
    }

    /** Create a new immutable view over the same memory.

        @return The frozen view.
    */
    ndarray_view<const T, ndim> freeze() const {
        return ndarray_view<const T, ndim>{m_buffer, m_shape, m_strides};
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

protected:
    // allow any `ndarray_view` specialization to call the protected constructor
    // of other `ndarray_view` specializations
    template<typename, std::size_t, bool>
    friend class ndarray_view;

public:
    using buffer_type = typename generic_ndarray_impl::buffer_type;

private:
    /** Iterator type to implement forward, const, reverse, and const reverse iterators.

        This type cannot be implemented with just a pointer and stride because stride may
        be zero, so an index is needed to count the number of iterations in that case.
     */
    template<typename V>
    struct generic_iterator {
    private:
        buffer_type m_ptr;
        std::size_t m_ix;
        std::int64_t m_stride;

    protected:
        friend ndarray_view;

        generic_iterator(buffer_type buffer, std::size_t ix, std::int64_t stride)
            : m_ptr(buffer), m_ix(ix), m_stride(stride) {}

    public:
        using difference_type = std::int64_t;
        using value_type = V;
        using pointer = value_type*;
        using const_pointer = const value_type*;
        using reference = value_type&;
        using const_reference = const value_type&;
        using iterator_category = std::random_access_iterator_tag;

        reference operator*() const {
            return *reinterpret_cast<V*>(m_ptr + m_ix * m_stride);
        }

        reference operator[](difference_type ix) const {
            return *(*this + ix);
        }

        pointer operator->() const {
            return reinterpret_cast<const V*>(m_ptr + m_ix * m_stride);
        }

        generic_iterator& operator++() {
            m_ix += 1;
            return *this;
        }

        generic_iterator operator++(int) {
            generic_iterator out = *this;
            m_ix += 1;
            return out;
        }

        generic_iterator& operator+=(difference_type n) {
            m_ix += n;
            return *this;
        }

        generic_iterator operator+(difference_type n) const {
            return generic_iterator(m_ptr, m_ix + n, m_stride);
        }

        generic_iterator& operator--() {
            m_ix -= 1;
            return *this;
        }

        generic_iterator operator--(int) {
            generic_iterator out = *this;
            m_ix -= 1;
            return out;
        }

        generic_iterator& operator-=(difference_type n) {
            m_ix -= n;
            return *this;
        }

        difference_type operator-(const generic_iterator& other) const {
            return m_ix - other.m_ix;
        }

        bool operator!=(const generic_iterator& other) const {
            return m_ix != other.m_ix;
        }

        bool operator==(const generic_iterator& other) const {
            return m_ix == other.m_ix;
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
    template<
        typename C,
        typename = std::enable_if_t<
            std::is_same_v<decltype(std::declval<C>().data()), std::remove_const_t<T>*>>>
    ndarray_view(C& contiguous_container)
        : generic_ndarray_impl(contiguous_container.data(),
                               {contiguous_container.size()},
                               {sizeof(T)}) {}

    /** Create a view over an arbitrary contiguous container of `T`s.

        @param contiguous_container The container to take a view of.
     */
    template<typename C,
             typename = std::enable_if_t<
                 std::is_const_v<T> && std::is_same_v<decltype(std::declval<C>().data()),
                                                      std::remove_const_t<T>*>>>
    ndarray_view(const C& contiguous_container)
        : generic_ndarray_impl(contiguous_container.data(),
                               {contiguous_container.size()},
                               {sizeof(T)}) {}

    /** Create a virtual array of length ``size`` holding the scalar ``value``.

        @param value The value to fill the array with.
        @param shape The shape of the array.
        @return The new mutable array view.
     */
    static ndarray_view virtual_array(T& value, const std::array<std::size_t, 1>& shape) {
        return {std::addressof(value), shape, {0}};
    }

    /** Create a virtual array of length ``size`` holding the scalar ``value``.

        @param value The value to fill the array with.
        @param size The size of the array.
        @return The new mutable array view.
     */
    static ndarray_view virtual_array(T& value, std::size_t size) {
        return {std::addressof(value), {size}, {0}};
    }

    iterator begin() const {
        return iterator(this->m_buffer, 0, this->m_strides[0]);
    }

    const_iterator cbegin() const {
        return const_iterator(this->m_buffer, 0, this->m_strides[0]);
    }

    iterator end() const {
        return iterator(this->m_buffer + this->pos_to_index(this->m_shape),
                        this->size(),
                        this->m_strides[0]);
    }

    const_iterator cend() const {
        return const_iterator(this->m_buffer + this->pos_to_index(this->m_shape),
                              this->size(),
                              this->m_strides[0]);
    }

    reverse_iterator rbegin() const {
        return reverse_iterator(this->m_buffer + this->pos_to_index({this->size() - 1}),
                                0,
                                -this->m_strides[0]);
    }

    const_reverse_iterator crbegin() const {
        return const_reverse_iterator(this->m_buffer +
                                          this->pos_to_index({this->size() - 1}),
                                      0,
                                      -this->m_strides[0]);
    }

    reverse_iterator rend() const {
        auto stride = -this->m_strides[0];
        return reverse_iterator(this->m_buffer + stride, this->size(), stride);
    }

    const_reverse_iterator crend() const {
        auto stride = -this->m_strides[0];
        return const_reverse_iterator(this->m_buffer + stride, this->size(), stride);
    }

    /**  Access the element at the given index without bounds checking.

         @param pos The index to lookup.
         @return A view of the string at the given index. Undefined if `pos` is
                 out of bounds.
     */
    T& operator[](std::size_t pos) const {
        return *reinterpret_cast<T*>(&this->m_buffer[this->pos_to_index({pos})]);
    }

    /** Access the first element of this array. The array must be non-empty.
     */
    T& front() const {
        return (*this)[0];
    }

    /** Access the last element of this array. The array must be non-empty.
     */
    T& back() const {
        return (*this)[this->size() - 1];
    }

    /** Create a view over a subsection of the viewed memory.

        @param start The start index of the slice.
        @param stop The stop index of the slice, exclusive.
        @param step The value to increment each index by.
        @return A view over a subset of the memory.
     */
    ndarray_view<T, 1, false>
    slice(std::size_t start, std::size_t stop = npos, std::size_t step = 1) const {
        std::size_t size = (stop == npos) ? this->m_shape[0] - start : stop - start;
        std::int64_t stride = this->m_strides[0] * step;
        return ndarray_view(this->m_buffer + this->pos_to_index({start}),
                            {size},
                            {stride});
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
public:
    using buffer_type =
        std::conditional_t<std::is_same_v<T, py::any_cref>, const std::byte*, std::byte*>;

protected:
    // allow any `any_ref_ndarray_view` specialization to call the protected constructor
    // of other `any_ref_ndarray_view` specializations
    template<std::size_t, typename, bool>
    friend class any_ref_ndarray_view;

    std::array<std::size_t, ndim> m_shape;
    std::array<std::int64_t, ndim> m_strides;
    buffer_type m_buffer;
    any_vtable m_vtable;

    std::ptrdiff_t pos_to_index(const std::array<std::size_t, ndim>& pos) const {
        std::ptrdiff_t ix = 0;
        for (std::size_t n = 0; n < ndim; ++n) {
            ix += pos[n] * m_strides[n];
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

    template<typename = std::enable_if_t<detail::buffer_format<T> != '\0'>>
    static std::tuple<any_ref_ndarray_view<ndim, T>, py::buffer>
    from_buffer_protocol(PyObject* ob) {
        py::buffer buf(new Py_buffer);
        int flags = PyBUF_ND | PyBUF_STRIDED | PyBUF_FORMAT;
        if (!std::is_const_v<T>) {
            flags |= PyBUF_WRITABLE;
        }
        if (PyObject_GetBuffer(ob, buf.get(), flags)) {
            throw py::exception{};
        }

        char fmt;
        if (!buf->format) {
            fmt = 'B';
        }
        else if (std::strlen(buf->format) != 1) {
            throw py::exception(PyExc_TypeError,
                                "cannot adapt buffer of format=",
                                buf->format,
                                " to an ndarray_view");
        }
        else {
            fmt = *buf->format;
        }

        py::any_vtable vtable;

        switch (fmt) {
        case detail::buffer_format<char>:
            vtable = py::any_vtable::make<char>();
            break;
        case detail::buffer_format<signed char>:
            vtable = py::any_vtable::make<signed char>();
            break;
        case detail::buffer_format<unsigned char>:
            vtable = py::any_vtable::make<unsigned char>();
            break;
        case detail::buffer_format<bool>:
            vtable = py::any_vtable::make<bool>();
            break;
        case detail::buffer_format<short>:
            vtable = py::any_vtable::make<short>();
            break;
        case detail::buffer_format<unsigned short>:
            vtable = py::any_vtable::make<unsigned short>();
            break;
        case detail::buffer_format<int>:
            vtable = py::any_vtable::make<int>();
            break;
        case detail::buffer_format<unsigned int>:
            vtable = py::any_vtable::make<unsigned int>();
            break;
        case detail::buffer_format<long>:
            vtable = py::any_vtable::make<long>();
            break;
        case detail::buffer_format<unsigned long>:
            vtable = py::any_vtable::make<unsigned long>();
            break;
        case detail::buffer_format<long long>:
            vtable = py::any_vtable::make<long long>();
            break;
        case detail::buffer_format<unsigned long long>:
            vtable = py::any_vtable::make<unsigned long long>();
            break;
        case detail::buffer_format<float>:
            vtable = py::any_vtable::make<float>();
            break;
        case detail::buffer_format<double>:
            vtable = py::any_vtable::make<double>();
            break;
        default:
            throw py::exception(PyExc_TypeError,
                                "cannot adapt buffer of format=",
                                fmt,
                                " to an ndarray_view");
        }

        if (buf->ndim != ndim) {
            throw py::exception(
                PyExc_TypeError, "buffer ndim=", buf->ndim, " != expected_ndim=", ndim);
        }

        std::array<std::size_t, ndim> shape;
        std::memcpy(shape.data(), buf->shape, ndim);

        std::array<std::int64_t, ndim> strides;
        std::memcpy(strides.data(), buf->strides, ndim);

        return {any_ref_ndarray_view{vtable,
                                     reinterpret_cast<buffer_type>(buf->buf),
                                     shape,
                                     strides},
                std::move(buf)};
    }

    template<typename = std::enable_if_t<detail::buffer_format<T> != '\0'>>
    static std::tuple<any_ref_ndarray_view, py::buffer>
    from_buffer_protocol(const py::scoped_ref<>& ob) {
        return from_buffer_protocol(ob.get());
    }

    /** Create a virtual array of length ``size`` holding the scalar ``value``.

        @param value The value to fill the array with.
        @param shape The shape of the array.
        @return The new mutable array view.
     */
    template<typename U>
    static any_ref_ndarray_view
    virtual_array(U& value, const std::array<std::size_t, ndim>& shape) {
        return {reinterpret_cast<buffer_type>(std::addressof(value)),
                shape,
                {0},
                &typeid(T),
                sizeof(T)};
    }

    /** Create a virtual array of length ``size`` holding the scalar ``value``.

        @param value The value to fill the array with.
        @param shape The shape of the array.
        @return The new mutable array view.
     */
    template<typename U, typename = std::enable_if_t<std::is_same_v<T, any_cref>>>
    static any_ref_ndarray_view
    virtual_array(const U& value, const std::array<std::size_t, ndim>& shape) {
        return {reinterpret_cast<buffer_type>(std::addressof(value)),
                shape,
                {0},
                &typeid(T),
                sizeof(T)};
    }

    any_ref_ndarray_view(buffer_type buffer,
                         const std::array<std::size_t, ndim> shape,
                         const std::array<std::int64_t, ndim>& strides,
                         const py::any_vtable& vtable)
        : m_shape(shape), m_strides(strides), m_buffer(buffer), m_vtable(vtable) {}

    /** Default constructor creates an empty view over nothing.
     */
    any_ref_ndarray_view() : m_shape({0}), m_strides({0}), m_buffer(nullptr) {}

    /** Take a view over `buffer`.

        @param buffer The buffer to take a view over.
        @param shape The number of elements in the buffer along each axis.
        @param stride The number of bytes between elements along each axis.
     */
    template<typename U>
    any_ref_ndarray_view(U* buffer,
                         const std::array<std::size_t, ndim> shape,
                         const std::array<std::int64_t, ndim>& strides)
        : any_ref_ndarray_view(reinterpret_cast<buffer_type>(buffer),
                               shape,
                               strides,
                               py::any_vtable::make<U>()) {}

    /** Access the element at the given index with bounds checking.

        @param pos The index to lookup.
        @return A view of the string at the given index.
     */
    template<typename... Ixs>
    reference at(Ixs... pos) const {
        return at({pos...});
    }

    /** Access the element at the given index with bounds checking.

        @param ixs The index to lookup.
        @return A view of the string at the given index.
     */
    reference at(const std::array<std::size_t, ndim>& ixs) const {
        for (std::size_t n = 0; n < ndim; ++n) {
            if (ixs[n] >= m_shape[n]) {
                throw std::out_of_range("pos exceeds the length of the array");
            }
        }

        return (*this)(ixs);
    }

    template<typename... Ixs>
    reference operator()(Ixs... ixs) const {
        return (*this)({ixs...});
    }

    reference operator()(const std::array<std::size_t, ndim>& ixs) const {
        return {&m_buffer[pos_to_index(ixs)], m_vtable};
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
    ndarray_view<U, ndim, higher_dimensional> cast() const {
        if (any_vtable::make<U>() != m_vtable) {
            throw std::bad_any_cast{};
        }

        return {m_buffer, m_shape, m_strides};
    }

    /** The underlying buffer of characters for this string array.
     */
    auto buffer() const {
        return m_buffer;
    }

    /** Check if two views are exactly identical.

        @param other The view to compare to.
        @return Are these views identical?
     */
    bool operator==(const any_ref_ndarray_view& other) const {
        return m_buffer == other.m_buffer && m_shape == other.m_shape &&
               m_strides == other.m_strides && m_vtable == other.m_vtable;
    }

    /** Check if two views are not exactly identical.

        @param other The view to compare to.
        @return Are these views not identical?
     */
    bool operator!=(const any_ref_ndarray_view& other) const {
        return !(*this == other);
    }

    const any_vtable& vtable() const {
        return m_vtable;
    }
};

template<typename T>
class any_ref_ndarray_view<1, T, false> : public any_ref_ndarray_view<1, T, true> {
private:
    using generic_ndarray_impl = any_ref_ndarray_view<1, T, true>;

protected:
    // allow any `any_ref_ndarray_view` specialization to call the protected constructor
    // of other `any_ref_ndarray_view` specializations
    template<std::size_t, typename, bool>
    friend class any_ref_ndarray_view;

public:
    using buffer_type = typename generic_ndarray_impl::buffer_type;

private:
    /** Iterator type to implement forward, const, reverse, and const reverse iterators.

        This type cannot be implemented with just a pointer and stride because stride may
        be zero, so an index is needed to count the number of iterations in that case.
     */
    template<typename R, typename C>
    struct generic_iterator {
    private:
        buffer_type m_ptr;
        std::size_t m_ix;
        std::int64_t m_stride;
        any_vtable m_vtable;

    protected:
        friend any_ref_ndarray_view;

        generic_iterator(buffer_type buffer,
                         std::size_t ix,
                         std::int64_t stride,
                         const any_vtable& vtable)
            : m_ptr(buffer), m_ix(ix), m_stride(stride), m_vtable(vtable) {}

    public:
        using difference_type = std::int64_t;
        using value_type = C;
        using pointer = C*;
        using reference = R;
        using const_reference = C;
        using iterator_category = std::random_access_iterator_tag;

        reference operator*() const {
            return {m_ptr + m_ix * m_stride, m_vtable};
        }

        reference operator[](difference_type ix) const {
            return *(*this + ix);
        }

        generic_iterator& operator++() {
            m_ix += 1;
            return *this;
        }

        generic_iterator operator++(int) {
            generic_iterator out = *this;
            m_ix += 1;
            return out;
        }

        generic_iterator& operator+=(difference_type n) {
            m_ix += n;
            return *this;
        }

        generic_iterator operator+(difference_type n) const {
            return {m_ptr, m_ix + n, m_stride, m_vtable};
        }

        generic_iterator& operator--() {
            m_ix -= 1;
            return *this;
        }

        generic_iterator operator--(int) {
            generic_iterator out = *this;
            m_ix -= 1;
            return out;
        }

        generic_iterator& operator-=(difference_type n) {
            m_ix -= n;
            return *this;
        }

        difference_type operator-(const generic_iterator& other) const {
            return m_ix - other.m_ix;
        }

        bool operator!=(const generic_iterator& other) const {
            return m_ix != other.m_ix;
        }

        bool operator==(const generic_iterator& other) const {
            return m_ix == other.m_ix;
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
        : generic_ndarray_impl(contiguous_container.data(),
                               {contiguous_container.size()},
                               {sizeof(U)}) {}

    /** Create a view over an arbitrary contiguous container of `T`s.

        @param contiguous_container The container to take a view of.
     */
    template<typename C,
             typename U = typename C::value_type,
             typename = std::enable_if_t<std::is_same_v<T, any_cref>>>
    any_ref_ndarray_view(const C& contiguous_container)
        : generic_ndarray_impl(contiguous_container.data(),
                               {contiguous_container.size()},
                               {sizeof(U)}) {}

    /** Create a view over an any_vector.

        @param any_vector The `any_vector` to view.
     */
    any_ref_ndarray_view(py::any_vector& any_vector)
        : generic_ndarray_impl(any_vector.data(),
                               {any_vector.size()},
                               {static_cast<std::int64_t>(any_vector.vtable().size())},
                               any_vector.vtable()) {}

    /** Create a view over an any_vector.

        @param any_vector The `any_vector` to view.
     */
    template<typename = std::enable_if_t<std::is_same_v<T, any_cref>>>
    any_ref_ndarray_view(const py::any_vector& any_vector)
        : generic_ndarray_impl(any_vector.data(),
                               {any_vector.size()},
                               {static_cast<std::int64_t>(any_vector.vtable().size())},
                               any_vector.vtable()) {}

    /** Create a virtual array of length ``size`` holding the scalar ``value``.

        @param value The value to fill the array with.
        @param shape The shape of the array.
        @return The new mutable array view.
     */
    template<typename U>
    static any_ref_ndarray_view virtual_array(U& value,
                                              const std::array<std::size_t, 1>& shape) {
        return {reinterpret_cast<buffer_type>(std::addressof(value)),
                shape,
                {0},
                any_vtable::make<U>()};
    }

    /** Create a virtual array of length ``size`` holding the scalar ``value``.

        @param value The value to fill the array with.
        @param size The size of the array.
        @return The new mutable array view.
     */
    template<typename U>
    static any_ref_ndarray_view virtual_array(U& value, std::size_t size) {
        return {reinterpret_cast<buffer_type>(std::addressof(value)),
                {size},
                {0},
                any_vtable::make<U>()};
    }

    iterator begin() const {
        return {this->m_buffer, 0, this->m_strides[0], this->m_vtable};
    }

    const_iterator cbegin() const {
        return {this->m_buffer, 0, this->m_strides[0], this->m_vtable};
    }

    iterator end() const {
        return {this->m_buffer + this->pos_to_index(this->m_shape),
                this->size(),
                this->m_strides[0],
                this->m_vtable};
    }

    const_iterator cend() const {
        return {this->m_buffer + this->pos_to_index(this->m_shape),
                this->size(),
                this->m_strides[0],
                this->m_vtable};
    }

    reverse_iterator rbegin() const {
        return {this->m_buffer + this->pos_to_index({this->size() - 1}),
                0,
                -this->m_strides[0],
                this->m_vtable};
    }

    const_reverse_iterator crbegin() const {
        return {this->m_buffer + this->pos_to_index({this->size() - 1}),
                0,
                -this->m_strides[0],
                this->m_vtable};
    }

    const_reverse_iterator rend() const {
        auto stride = -this->m_strides[0];
        return {this->m_buffer + stride, this->size(), stride, this->m_vtable};
    }

    const_reverse_iterator crend() {
        auto stride = -this->m_strides[0];
        return {this->m_buffer + stride, this->size(), stride, this->m_vtable};
    }

    /**  Access the element at the given index without bounds checking.

         @param pos The index to lookup.
         @return A view of the string at the given index. Undefined if `pos` is
                 out of bounds.
     */
    reference operator[](std::size_t pos) const {
        return {&this->m_buffer[this->pos_to_index({pos})], this->m_vtable};
    }

    /** Access the first element of this array. The array must be non-empty.
     */
    reference front() const {
        return (*this)[0];
    }

    /** Access the last element of this array. The array must be non-empty.
     */
    reference back() const {
        return (*this)[this->size() - 1];
    }

    /** Cast the array to a static type.
     */
    template<typename U>
    ndarray_view<U, 1, false> cast() const {
        if (any_vtable::make<U>() != this->m_vtable) {
            throw std::bad_any_cast{};
        }

        return {reinterpret_cast<U*>(this->m_buffer), this->m_shape, this->m_strides};
    }
};
}  // namespace detail

template<>
class ndarray_view<any_cref, 1, false>
    : public detail::any_ref_ndarray_view<1, any_cref, false> {
private:
    using generic_ndarray_impl = detail::any_ref_ndarray_view<1, any_cref, false>;

protected:
    friend class ndarray_view<any_ref, 1, false>;

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
    ndarray_view
    slice(std::size_t start, std::size_t stop = npos, std::size_t step = 1) const {
        std::size_t size = (stop == npos) ? this->m_shape[0] - start : stop - start;
        std::int64_t stride = this->m_strides[0] * step;
        return {this->m_buffer + this->pos_to_index({start}),
                {size},
                {stride},
                this->m_vtable};
    }

    /** Create a new immutable view over the same memory.

        @return The frozen view.
     */
    ndarray_view<any_cref, 1, false> freeze() const {
        return {this->m_buffer, this->m_shape, this->m_strides, this->m_vtable};
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
    ndarray_view
    slice(std::size_t start, std::size_t stop = npos, std::size_t step = 1) const {
        std::size_t size = (stop == npos) ? this->m_shape[0] - start : stop - start;
        std::int64_t stride = this->m_strides[0] * step;
        return ndarray_view{this->m_buffer + this->pos_to_index({start}),
                            {size},
                            {stride},
                            this->m_vtable};
    }

    /** Create a new immutable view over the same memory.

        @return The frozen view.
     */
    ndarray_view<any_cref, 1, false> freeze() const {
        return {this->m_buffer, this->m_shape, this->m_strides, this->m_vtable};
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
        return {this->m_buffer, this->m_shape, this->m_strides, this->m_vtable};
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
        return {this->m_buffer, this->m_shape, this->m_strides, this->m_vtable};
    }
};
}  // namespace py
